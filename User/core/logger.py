import logging
from logging.handlers import RotatingFileHandler
from config.settings import LOG_DIR, LOG_FILE
import os

os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(name="user_logger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=5)
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger