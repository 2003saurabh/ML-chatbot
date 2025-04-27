import os
import time
import threading
import dotenv
import boto3
import requests
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import BedrockEmbeddings
from logger import setup_logger
from error_handler import handle_errors

# Load & validate environment
dotenv.load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# Optional upload timeout env var (seconds)
QDRANT_UPLOAD_TIMEOUT = float(os.getenv("QDRANT_UPLOAD_TIMEOUT", 120.0))
if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("Missing QDRANT_URL or QDRANT_API_KEY environment variable")

# Logging
logger = setup_logger(__name__)

# Thread-safe singletons
default_timeout = float(os.getenv("QDRANT_CLIENT_TIMEOUT", 60.0))
_qdrant_lock = threading.Lock()
_bedrock_lock = threading.Lock()
_qdrant_client = None
_bedrock_client = None


def get_qdrant_client(retries=3, backoff=2.0, timeout=default_timeout):
    """Thread-safe singleton Qdrant client with retry logic."""
    global _qdrant_client
    if _qdrant_client is None:
        with _qdrant_lock:
            if _qdrant_client is None:
                for attempt in range(1, retries + 1):
                    try:
                        logger.info(f"[Qdrant] Init attempt {attempt}")
                        client = QdrantClient(
                            url=QDRANT_URL,
                            api_key=QDRANT_API_KEY,
                            timeout=timeout
                        )
                        client.get_collections()
                        _qdrant_client = client
                        logger.info("Qdrant client initialized successfully")
                        break
                    except Exception as e:
                        logger.error(f"Qdrant init failed ({e}), retrying in {backoff}s...")
                        time.sleep(backoff)
                else:
                    raise ConnectionError("Unable to initialize Qdrant client after retries")
    return _qdrant_client


def get_bedrock_client(region="ap-south-1"):
    """Thread-safe singleton Bedrock client."""
    global _bedrock_client
    if _bedrock_client is None:
        with _bedrock_lock:
            if _bedrock_client is None:
                try:
                    logger.info("Initializing Bedrock client...")
                    _bedrock_client = boto3.client("bedrock-runtime", region_name=region)
                except Exception:
                    logger.exception("Failed to initialize Bedrock client")
                    raise
    return _bedrock_client

qdrant_client = get_qdrant_client()
bedrock_client = get_bedrock_client()

@handle_errors
def create_collection_if_not_exists(
    collection_name: str,
    overwrite: bool = False,
    vector_dim: int = 1024,
    distance: rest_models.Distance = Distance.COSINE
):
    """Ensure Qdrant collection exists or recreate if overwrite=True."""
    if not isinstance(collection_name, str) or not collection_name.strip():
        raise ValueError("collection_name must be a non-empty string")
    client = qdrant_client
    try:
        existing = [c.name for c in client.get_collections().collections]
        exists = collection_name in existing
    except Exception as e:
        logger.warning(f"Could not fetch collections list: {e}")
        exists = client.collection_exists(collection_name)

    params = VectorParams(size=vector_dim, distance=distance)
    if exists:
        if overwrite:
            client.recreate_collection(collection_name, vectors_config=params)
            logger.info(f"Recreated collection '{collection_name}'")
        else:
            logger.info(f"Leaving existing collection '{collection_name}' intact (append mode)")
    else:
        client.recreate_collection(collection_name, vectors_config=params)
        logger.info(f"Created collection '{collection_name}'")

@handle_errors
def store_embeddings_in_qdrant(
    collection_name: str,
    chunks: list,
    embeddings: list,
    overwrite: bool = False,
    upload_retries: int = 3,
    backoff: float = 2.0
):
    """
    Store text chunks + embeddings into Qdrant, handling surrogates.
    """
    if not isinstance(chunks, list) or not isinstance(embeddings, list):
        raise TypeError("chunks and embeddings must be lists")

    # Filter and sanitize text/embedding pairs
    pairs = []
    for idx, chunk in enumerate(chunks):
        if hasattr(chunk, "page_content") and isinstance(chunk.page_content, str):
            text = chunk.page_content.strip()
            if text and idx < len(embeddings):
                emb = embeddings[idx]
                if hasattr(emb, "__len__"):
                    # Remove surrogate code points
                    safe_text = ''.join(
                        c for c in text
                        if not (0xD800 <= ord(c) <= 0xDFFF)
                    )
                    if len(safe_text) < len(text):
                        logger.warning("Removed invalid Unicode surrogates from chunk %d", idx)
                    pairs.append((idx, safe_text, emb))
                else:
                    logger.warning(f"Skipping invalid embedding at index {idx}")
    if not pairs:
        logger.warning("No valid (chunk, embedding) pairs to store")
        return []

    dropped = len(chunks) - len(pairs)
    if dropped:
        logger.warning(f"Dropped {dropped} invalid or mismatched chunk(s)")

    # Ensure collection
    create_collection_if_not_exists(collection_name, overwrite=overwrite)

    # Prepare upsert points
    points = []
    for idx, text, emb in pairs:
        pid = str(uuid.uuid4())
        points.append(
            rest_models.PointStruct(
                id=pid,
                vector=emb,
                payload={"source": f"chunk_{idx}", "text": text}
            )
        )

    # Upload with retry
    for attempt in range(1, upload_retries + 1):
        try:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )
            logger.info(f"Uploaded {len(points)} embeddings to '{collection_name}'")
            break
        except (requests.exceptions.Timeout, UnexpectedResponse) as e:
            if attempt < upload_retries:
                logger.warning(f"Upsert attempt {attempt} failed: {e}, retrying in {backoff}s...")
                time.sleep(backoff)
            else:
                logger.exception("All upsert attempts failed")
                raise
    return [p.id for p in points]

@handle_errors
def get_qdrant_vectorstore(collection_name: str):
    """Return a QdrantVectorStore for querying the given collection."""
    if not isinstance(collection_name, str) or not collection_name.strip():
        raise ValueError("collection_name must be a non-empty string")
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            client=bedrock_client
        )
    )

@handle_errors
def count_vectors(collection_name: str) -> int:
    """Return the total number of vectors in a Qdrant collection."""
    if not isinstance(collection_name, str) or not collection_name.strip():
        raise ValueError("collection_name must be a non-empty string")
    resp = qdrant_client.count(collection_name=collection_name)
    return resp.count
