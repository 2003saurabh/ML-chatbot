import boto3
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_aws import BedrockEmbeddings
from config.settings import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, EMBEDDING_MODEL_ID, AWS_REGION
from core.logger import setup_logger

logger = setup_logger(__name__)

def get_vectorstore():
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        embedding = BedrockEmbeddings(
            client=boto3.client("bedrock-runtime", region_name=AWS_REGION),
            model_id=EMBEDDING_MODEL_ID
        )
        logger.info("Vectorstore and embedding initialized.")
        return QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION,
            embedding=embedding
        )
    except Exception as e:
        logger.exception("Failed to initialize vectorstore.")
        raise e