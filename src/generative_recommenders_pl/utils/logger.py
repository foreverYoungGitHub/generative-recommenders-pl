import logging


def configure_logger(name: str):
    # Get the root logger
    logger = logging.getLogger(name)
    # Set the logging level
    logger.setLevel(logging.INFO)
    return logger
