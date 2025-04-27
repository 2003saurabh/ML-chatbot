import streamlit as st
from core.chatbot import get_chat_response
from core.logger import setup_logger
from services.memory_service import clean_up_memory, get_memory
from datetime import datetime

# Setup logger
logger = setup_logger(__name__)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.message_count = 0
    st.session_state.memory = get_memory()

# Page config
st.set_page_config(
    page_title="ML Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("Chat Settings")
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.message_count = 0
        clean_up_memory(
            st.session_state.memory["short_term"],
            st.session_state.memory["long_term"],
            0
        )
        st.success("Chat history cleared!")

# Main chat interface
st.title("Machine Learning Assistant 🤖")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
query = st.chat_input("Ask me anything about Machine Learning...")

if query:
    st.session_state.message_count += 1
    logger.info(f"User input: {query}")

    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    try:
        # Get bot response
        with st.spinner("Thinking..."):
            response = get_chat_response(query)
        
        # Add bot response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        # Periodic memory cleanup
        if st.session_state.message_count % 10 == 0:
            clean_up_memory(
                st.session_state.memory["short_term"],
                st.session_state.memory["long_term"],
                st.session_state.message_count
            )
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        fallback = "I'm having trouble right now. Could you rephrase that?"
        st.session_state.messages.append({"role": "assistant", "content": fallback})
        with st.chat_message("assistant"):
            st.markdown(fallback)