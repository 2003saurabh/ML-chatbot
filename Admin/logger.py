import logging

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.hasHandlers():  # ✅ Prevent duplicate handlers
        logger.setLevel(logging.INFO)

        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler('admin.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger