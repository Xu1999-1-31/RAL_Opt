import logging
from typing import Union

def setup_logging(logger: logging.Logger, level: Union[str, int] = "INFO") -> None:
    """Setup logger format and logging level"""
    if isinstance(level, str):
        level = level.upper()
        level = getattr(logging, level, logging.INFO)
    logger.setLevel(level)
    
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )