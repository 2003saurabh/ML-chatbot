import boto3
import time
import random
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws.embeddings import BedrockEmbeddings
from vectorstore_utils import get_bedrock_client
from logger import setup_logger
from error_handler import handle_errors

# Logging setup
logger = setup_logger(__name__)

# Initialize Bedrock client only once
bedrock_client = get_bedrock_client()

def retry_with_backoff(func, *args, **kwargs):
    """Retry a function with exponential backoff."""
    retries = 5
    delay = 1  # Initial delay in seconds
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == retries - 1:
                raise
            backoff_delay = delay * (2 ** attempt) + random.uniform(0, 1)
            logger.info(f"Retrying in {backoff_delay:.2f} seconds...")
            time.sleep(backoff_delay)

@handle_errors
def extract_text_chunks(file_path: str):
    """Extract text chunks from a PDF file."""
    logger.info(f"Loading PDF file: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,separators=["\n\n", "\n", ".", " "])
    chunks = text_splitter.split_documents(documents)

    logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
    return chunks

@handle_errors
def generate_embeddings(chunks):
    """Generate embeddings using AWS Bedrock."""
    bedrock_embeddings = BedrockEmbeddings(
        client=bedrock_client,
        model_id="amazon.titan-embed-text-v2:0"
    )

    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = bedrock_embeddings.embed_documents(texts)
    logger.info(f"Generated {len(embeddings)} embeddings")
    return embeddings

def generate_embeddings_in_batches(chunks, batch_size=100):
    """Generate embeddings for chunks in batches to avoid timeouts and API throttling."""
    logger.info(f"Generating embeddings in batches of {batch_size}...")
    embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} of {(len(chunks) + batch_size - 1) // batch_size}")
        try:
            embeddings_batch = retry_with_backoff(generate_embeddings, batch)
            embeddings.extend(embeddings_batch)
        except Exception as e:
            logger.error(f"Error generating embeddings for batch {i // batch_size + 1}: {str(e)}")
            continue

    return embeddings
