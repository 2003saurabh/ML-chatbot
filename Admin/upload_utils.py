import boto3
import os
from botocore.exceptions import BotoCoreError, ClientError
from logger import setup_logger
from error_handler import handle_errors

logger = setup_logger(__name__)

# Initialize S3 client globally (reuse connection)
try:
    s3 = boto3.client("s3")
except Exception as e:
    logger.error(f"Failed to initialize S3 client: {str(e)}")
    raise

@handle_errors
def validate_pdf(file_path: str) -> bool:
    """
    Validate if the file is a PDF and exists locally.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if valid PDF, False otherwise.
    """
    if not file_path.lower().endswith(".pdf"):
        logger.warning(f"File does not have a .pdf extension: {file_path}")
        return False
    if not os.path.exists(file_path):
        logger.warning(f"File does not exist: {file_path}")
        return False
    return True

@handle_errors
def upload_to_s3(file_path: str, bucket: str, s3_key: str) -> None:
    """
    Upload a valid PDF file to S3.

    Args:
        file_path (str): Local path of the PDF file.
        bucket (str): Target S3 bucket name.
        s3_key (str): S3 key (path within bucket).
    """
    if not validate_pdf(file_path):
        logger.error("Upload aborted due to invalid file.")
        return

    try:
        logger.info(f"Uploading {file_path} to s3://{bucket}/{s3_key}")
        s3.upload_file(file_path, bucket, s3_key)
        logger.info("Upload successful.")
    except (BotoCoreError, ClientError) as e:
        logger.error(f"S3 upload failed: {str(e)}")
        raise
