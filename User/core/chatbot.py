from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableMap, RunnableLambda
from services.llm_service import get_llm
from services.vector_service import get_vectorstore
from services.memory_service import get_memory
from core.logger import setup_logger
from core.error_handler import handle_errors
from typing import Dict, Any
import streamlit as st

logger = setup_logger(__name__)

# Initialize core components
llm = get_llm()
retriever = get_vectorstore().as_retriever(search_kwargs={"k": 7})

# Prompt Template with dynamic summary handling
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are Llama 3, a helpful, concise, and knowledgeable AI assistant. 
You answer questions strictly using only the information provided in <context>. 
You do not use prior knowledge or make assumptions beyond the given context.

Instructions:
- If the answer is found in the context, answer concisely and accurately.
- If the answer is not found in the context, reply exactly with:
"I'm sorry, I could not find enough information to answer that."
- Do not mention your limitations or training data.
- Do not repeat the context in your answer.
- Stay strictly within the Machine Learning domain.
"""),

    MessagesPlaceholder(variable_name="chat_history"),

    ("user", """
<context>
{context}
</context>

Question:
{question}
""")
])

# Runnable Input Mapper
input_map = RunnableLambda(lambda x: {
    "context": x.get("context", ""),
    "question": x.get("question", ""),
    "chat_history": x.get("chat_history", [])
})

# Output cleanup (optional)
output_parser = RunnableLambda(lambda x: x.strip() if isinstance(x, str) else x)

logger.debug("final_input: %s", input_map | prompt)
# Final chain without summary
chain = input_map | prompt | llm | output_parser


@handle_errors
def get_chat_response(question: str) -> str:
    """Generate a chat response based on context and history."""
    logger.info(f"Processing question: {question}")

    if not question.strip():
        return "Please enter a valid question."

    try:
        memory = st.session_state.memory

        # Retrieve context
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs]) or "No relevant context found."
        logger.debug(f"Question: {question}")
        logger.debug(f"Number of docs retrieved: {len(docs)}")
        for idx, doc in enumerate(docs):
            logger.debug(f"Doc {idx+1}:\n{doc.page_content}\n---")
    
        # Log the retrieved context
        logger.info(f"Context passed to LLM:\n{context}")
        
        # Load short-term memory
        short_term_history = memory["short_term"].load_memory_variables({}).get("chat_history", [])
        logger.info(f"Short-term history: {short_term_history[-5:]}")

        # Prepare the final input with the retrieved context and short-term memory
        final_input: Dict[str, Any] = {
            "context": context,
            "question": question,
            "chat_history": short_term_history[-5:]  # Only the last 5 messages from short-term memory
        }

        # Render and log the final expanded prompt
        rendered_prompt = prompt.format_messages(
            context=final_input["context"],
            question=final_input["question"],
            chat_history=final_input["chat_history"]
        )
        logger.info("Final Prompt given to LLM:")
        for message in rendered_prompt:
            logger.info(f"{message.type.upper()}: {message.content}")

        # Get response
        response = chain.invoke(final_input)

        if not isinstance(response, str) or not response.strip():
            logger.warning("Invalid LLM response received")
            return "I apologize, but I couldn't generate a proper response."

        # Update short-term memory with the new messages
        memory["short_term"].chat_memory.add_user_message(question)
        memory["short_term"].chat_memory.add_ai_message(response)
        logger.info("Short-term memory updated.")

        logger.info("Response generated successfully")
        return response

    except Exception as e:
        logger.exception("Error in chat response generation")
        raise e
