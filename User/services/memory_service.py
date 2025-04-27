from typing import Dict, Any
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from services.llm_service import get_llm
from core.logger import setup_logger

logger = setup_logger(__name__)


def get_memory() -> Dict[str, Any]:
    """Initialize and return short-term and long-term memory instances."""
    llm = get_llm()

    short_term_memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        return_messages=True,
        max_token_limit=2000
    )

    long_term_memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history_summary",
        input_key="question",
        return_messages=True,
        max_token_limit=4000
    )

    logger.info("Memory initialized")

    return {
        "short_term": short_term_memory,
        "long_term": long_term_memory
    }


def clean_up_memory(short_term_memory: ConversationBufferMemory, 
                    long_term_memory: ConversationSummaryMemory, 
                    message_count: int) -> None:
    """Handle memory cleanup and summarization."""
    try:
        if message_count > 0 and message_count % 10 == 0:
            chat_history = short_term_memory.load_memory_variables({})
            summary = long_term_memory.predict_new_summary(
                chat_history.get("chat_history", []),
                ""
            )
            logger.info(f"Created summary: {summary[:100]}...")

            short_term_memory.clear()
            logger.info(f"Short-term memory cleaned at message {message_count}")

        elif message_count == 0:
            short_term_memory.clear()
            long_term_memory.clear()
            logger.info("All memories cleared")

    except Exception as e:
        logger.exception("Memory cleanup failed")
        raise e
