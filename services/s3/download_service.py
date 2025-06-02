import torch
import io
import pickle
import json

from services.s3.base_service import S3BaseService


class S3DownloadService(S3BaseService):
    """
    Service class for downloading trained models and associated files from AWS S3.

    This service handles the retrieval of trained models, scalers, and metadata from S3
    using a model ID. It manages the downloading and instantiation of all model components.

    Attributes:
        id (str): The unique identifier of the model to download.

    Raises:
        FileNotFoundError: If the model files are not found in S3.
    """

    def __init__(self, id):
        """
        Initialize the S3DownloadService.

        Args:
            id (str): The unique identifier of the model to download.
        """
        super().__init__()
        self.id = id
    
    def execute(self):
        s3_path = self.__s3_path()
        files = self.__find_path(s3_path)

        model_file_path = self.__find_file(files, "model.pth", required=True)
        scaler_file_path = self.__find_file(files, "scaler.pkl")
        metadata_file_path = self.__find_file(files, "metadata.json", required=True)

        model = self.__instantiate_model(model_file_path)
        scaler = self.__instantiate_scaler(scaler_file_path)
        metadata = self.__instantiate_metadata(metadata_file_path)

        return model, scaler, metadata

    def __s3_path(self):
        return f"models/{self.id}"

    def __find_path(self, s3_path):
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=s3_path
        )

        if 'Contents' in response:
            return response['Contents']
        else:
            raise FileNotFoundError("Model Not Found")
        
    def __find_file(self, files, file_name, required=False):
        for file in files:
            if file['Key'].endswith(file_name):
                return file['Key']
            
        if required:
            raise FileNotFoundError(f"{file_name} Not Found in {self.id}")

        return None
        
    def __instantiate_model(self, model_file_path):
        file = self.__get_file(model_file_path)
        buffer = io.BytesIO(file['Body'].read())
        return torch.load(buffer, map_location=torch.device('cpu'), weights_only=False)

    def __get_file(self, file_path):
        return self.s3_client.get_object(Bucket=self.bucket_name, Key=file_path)
        
    def __instantiate_scaler(self, scaler_file_path):
        if scaler_file_path is None:
            return None

        file = self.__get_file(scaler_file_path)
        return pickle.load(file['Body'])

    def __instantiate_metadata(self, metadata_file_path):
        file = self.__get_file(metadata_file_path)
        return json.load(file['Body'])