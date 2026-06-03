import logging
import sys
from pathlib import Path
 
 
def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Create and configure a logger.
 
    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
 
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
 
    if logger.handlers:
        return logger
 
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
 
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
 
    # Console handler — dùng utf-8 để tránh UnicodeEncodeError trên Windows (cp1252)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.stream = open(
        sys.stdout.fileno(),
        mode="w",
        encoding="utf-8",
        buffering=1,
        closefd=False,
    )
    logger.addHandler(console_handler)
 
    # File handler — utf-8
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
 
    return logger