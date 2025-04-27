import boto3
from langchain_aws import BedrockLLM
from config.settings import AWS_REGION, LLM_MODEL_ID
from core.logger import setup_logger

logger = setup_logger(__name__)

def get_llm():
    try:
        client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        logger.info("Bedrock LLM client initialized.")
        return BedrockLLM(
            client=client,
            model_id=LLM_MODEL_ID,
            model_kwargs={"temperature": 0.7, "max_tokens": 1024}
        )
    except Exception as e:
        logger.exception("Failed to initialize LLM.")
        raise e