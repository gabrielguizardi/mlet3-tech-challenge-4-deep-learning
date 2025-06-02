import boto3
import os

class S3BaseService:
    """
    Base service class for AWS S3 operations.

    This class provides basic S3 client setup using AWS credentials from environment variables.
    It serves as a base class for more specific S3 operations like upload and download.

    Attributes:
        s3_client: boto3 S3 client instance.
        bucket_name (str): Name of the S3 bucket to use.
    """

    def __init__(self):
        """
        Initialize the S3BaseService.

        Sets up the S3 client using AWS credentials from environment variables.
        Environment variables required:
            - AWS_ACCESS_KEY_ID
            - AWS_SECRET_ACCESS_KEY
            - AWS_SESSION_TOKEN
            - S3_BUCKET_NAME
        """
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
            region_name='us-east-1'
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
