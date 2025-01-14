"""Logging setup."""

import logging
import os
from datetime import datetime


def setup_logging(
    level: int = logging.ERROR, name_appendix: str = "", dir_path: str = "logs"
) -> logging.Logger:
    """Set up and return a logger with both terminal and file output."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)  # pragma: no cover
    logger = logging.getLogger()
    logger.setLevel(level)
    current_time = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_file_path = os.path.join(dir_path, f"{current_time}_{name_appendix}.log")
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(log_file_path),
    ]
    formatter = logging.Formatter(f"%(asctime)s - {name_appendix} - %(levelname)s - %(message)s")
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
