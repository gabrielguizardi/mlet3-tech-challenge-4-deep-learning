import boto3
import os

class S3BaseService:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
            region_name='us-east-1'
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME')