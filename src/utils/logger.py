import logging
import sys 
from pathlib import Path

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get a configured logger instance.
    Args:
        name (str): The name of the logger.
        level (str): The logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    #File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "app.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger