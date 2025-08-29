import logging
import os
from utils.config import LOG_DIR

os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name="delhi_rent_forecasting"):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler = logging.FileHandler(os.path.join(LOG_DIR, "pipeline.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger
