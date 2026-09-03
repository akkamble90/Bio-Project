import logging
import sys
from src.common.config import settings

def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a thread-safe, structured console logger.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        level_name = settings.log_level.upper()
        log_level = getattr(logging, level_name, logging.INFO)
        logger.setLevel(log_level)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    logger.propagate = False
    return logger