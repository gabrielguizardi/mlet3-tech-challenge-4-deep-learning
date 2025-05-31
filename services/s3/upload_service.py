import uuid
import torch
import os
import pickle
import json

from services.s3.base_service import S3BaseService

class S3UploadService(S3BaseService):
    def __init__(self, model, scaler, metadata: dict):
        super().__init__()
        self.model = model
        self.scaler = scaler
        self.metadata = metadata
       

    def execute(self):
        id = str(uuid.uuid4())
        train_path = self.__train_path(id)

        model_path, scaler_path, metadata_path = self.__save_files(train_path)
        model_s3_path, scaler_s3_path, metadata_s3_path = self.__upload_files(model_path, scaler_path, metadata_path, id)

        self.__exclude_files(train_path)

        return id, model_s3_path, scaler_s3_path, metadata_s3_path

    def __train_path(self, id):
        return f"models/repository/{id}"
    
    def __save_files(self, train_path):
        os.makedirs(train_path, exist_ok=True)

        model_path = self.__save_model(train_path)
        scaler_path = self.__save_scaler(train_path)
        metadata_path = self.__save_metadata(train_path)      
        
        return model_path, scaler_path, metadata_path
    
    def __save_model(self, train_path):
        model_path = os.path.join(train_path, "model.pth")
        torch.save(self.model, model_path)

        return model_path
    
    def __save_scaler(self, train_path):
        if self.scaler is not None:
            scaler_path = os.path.join(train_path, "scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            return scaler_path
        
        return None
    
    def __save_metadata(self, train_path):
        metadata_path = os.path.join(train_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f)

        return metadata_path
    
    def __upload_files(self, model_path, scaler_path, metadata_path, id):
        model_s3_path = self.__upload_to_s3(model_path, id)
        metadata_s3_path = self.__upload_to_s3(metadata_path, id)

        scaler_s3_path = None
        if scaler_path is not None:
            scaler_s3_path = self.__upload_to_s3(scaler_path, id)

        return model_s3_path, scaler_s3_path, metadata_s3_path

    def __upload_to_s3(self, file_path, id):
        s3_key = f"models/{id}/{os.path.basename(file_path)}"
        self.s3_client.upload_file(file_path, self.bucket_name, s3_key)

        return f"s3://{self.bucket_name}/{s3_key}"

    def __exclude_files(self, train_path):
        if os.path.exists(train_path):
            for root, dirs, files in os.walk(train_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(train_path)