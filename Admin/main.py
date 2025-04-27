import streamlit as st
import os
import uuid
from datetime import datetime
from logger import setup_logger
from upload_utils import upload_to_s3
from embedding_utils import extract_text_chunks, generate_embeddings
from vectorstore_utils import (
    store_embeddings_in_qdrant,
    count_vectors,
    get_qdrant_client
)

# Initialize logger
logger = setup_logger(__name__)

# Streamlit UI setup
st.set_page_config(page_title="Admin Uploader", layout="wide")
st.title("📚 Admin Upload Interface")
st.markdown("Upload a PDF, extract its content, and store embeddings in Qdrant.")

# Cache the Qdrant client as a resource to prevent repeated instantiation
@st.cache_resource
def get_cached_client():
    return get_qdrant_client()

# Cache the collection list (refresh every 10 minutes)
@st.cache_data(ttl=600)
def list_collections():
    client = get_cached_client()
    try:
        col_objs = client.get_collections().collections
        logger.debug("Fetched Qdrant collections.")
        return [c.name for c in col_objs]
    except Exception as e:
        logger.error(f"Error fetching collections: {e}")
        return []

# Input fields
bucket_name = st.text_input("S3 Bucket Name")

# Fetch Qdrant collections using cached functions
collection_names = list_collections()
if not collection_names:
    st.warning("⚠️ Could not retrieve collections or no collections exist yet.")
existing = st.selectbox("Select Existing Collection (optional)", [""] + collection_names)

# Other inputs
custom_name = st.text_input("Or Enter New Qdrant Collection Name (overrides selection)")
overwrite = st.checkbox("Overwrite if collection already exists?", value=False)
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])


def process_pdf(uploaded_file, bucket_name, collection_name, overwrite):
    """
    Process the uploaded PDF file: save locally, upload to S3,
    extract text, generate embeddings, and store in Qdrant.
    """
    # Validate inputs
    if not uploaded_file:
        st.warning("⚠️ Please upload a PDF file.")
        logger.warning("Upload attempt without file.")
        return
    if not bucket_name:
        st.warning("⚠️ Please enter a valid S3 bucket name.")
        logger.warning("No S3 bucket name provided.")
        return
    if not collection_name:
        st.warning("⚠️ Please specify a Qdrant collection name.")
        logger.warning("No Qdrant collection name provided.")
        return

    unique_id = str(uuid.uuid4())
    original_filename = uploaded_file.name
    extension = os.path.splitext(original_filename)[1]
    local_filename = f"{unique_id}{extension}"
    s3_key = f"uploads/{local_filename}"

    try:
        # Save locally
        with open(local_filename, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success(f"✅ File saved locally as `{local_filename}`.")
        logger.info(f"Local file saved: {local_filename} (Original: {original_filename})")

        # Upload to S3
        upload_to_s3(local_filename, bucket_name, s3_key)
        st.success(f"✅ Uploaded to S3 at `{s3_key}`.")
        logger.info(f"Uploaded to S3 - Bucket: {bucket_name}, Key: {s3_key}")

        # Extract text
        chunks = extract_text_chunks(local_filename)
        if not chunks:
            st.error("❌ No text extracted from the PDF.")
            logger.error("Text extraction returned 0 chunks.")
            return
        st.success(f"✅ Extracted {len(chunks)} text chunks.")
        logger.info(f"Extracted {len(chunks)} text chunks from {local_filename}")

        # Generate embeddings
        embeddings = generate_embeddings(chunks)
        if not embeddings:
            st.error("❌ Failed to generate embeddings.")
            logger.error("Embedding generation returned empty.")
            return
        st.success(f"✅ Generated {len(embeddings)} embeddings.")
        logger.info(f"Generated {len(embeddings)} embeddings for file: {local_filename}")

        # Store embeddings (append or overwrite based on user choice)
        added_ids = store_embeddings_in_qdrant(
            collection_name,
            chunks,
            embeddings,
            overwrite=overwrite
        )

        # Show results
        if added_ids:
            action = "Overwritten" if overwrite else "Appended"
            st.success(f"🎉 {action} {len(added_ids)} embeddings in collection `{collection_name}`!")
            logger.info(f"{action} {len(added_ids)} embeddings to Qdrant collection `{collection_name}`.")
            # Display total count
            try:
                total = count_vectors(collection_name)
                st.info(f"📊 Total vectors in `{collection_name}`: {total}")
                logger.info(f"Total vectors in `{collection_name}`: {total}")
            except Exception:
                st.warning("⚠️ Could not fetch total vector count.")
        else:
            st.warning("⚠️ No embeddings were added.")

    except Exception as e:
        st.error("❌ An unexpected error occurred.")
        logger.exception(f"Exception in process_pdf: {e}")

    finally:
        # Cleanup
        if os.path.exists(local_filename):
            os.remove(local_filename)
            logger.info(f"Local file cleaned up: {local_filename}")

# Determine final collection name
collection_name = custom_name.strip() if custom_name.strip() else existing.strip()

# Trigger process
if st.button("🚀 Process PDF"):
    with st.spinner("Processing..."):
        process_pdf(uploaded_file, bucket_name, collection_name, overwrite)
