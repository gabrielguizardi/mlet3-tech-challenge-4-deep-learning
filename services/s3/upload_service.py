import uuid
import torch
import os
import pickle
import boto3
import json

class S3UploadService:
    def __init__(self, model, scaler, metadata: dict):
        self.model = model
        self.scaler = scaler
        self.metadata = metadata
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
            region_name='us-east-1'
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME')

    def execute(self):
        id = str(uuid.uuid4())
        train_path = self.__train_path(id)

        model_path, scaler_path, metadata_path = self.__save_files(train_path)
        model_s3_path, scaler_s3_path, metadata_s3_path = self.__upload_files(model_path, scaler_path, metadata_path, id)

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
