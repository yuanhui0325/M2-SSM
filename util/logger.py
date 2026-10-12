"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
import logging
import sys


def setup_logger(name: str = "jpeg_encoder", level: int = logging.INFO) -> logging.Logger:
    """
    Set up and configure a logger for the JPEG encoder.

    Args:
        name: Name of the logger
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if logger already exists
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Default logger instance
logger = setup_logger()
