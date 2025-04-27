import traceback
from core.logger import setup_logger

logger = setup_logger("error_handler")

def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            logger.debug(f"Calling function: {func.__name__}")
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {e}")
            logger.debug(traceback.format_exc())
            return "Sorry, something went wrong. Please try again."
    return wrapper