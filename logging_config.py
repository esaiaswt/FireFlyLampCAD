"""
Shared logging configuration for the Funnel STL Viewer application.

Provides a centralized logging setup with rotating file handler and console output.
All modules should use get_logger(name) to obtain their logger instance.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler

# Log directory and file path
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# RotatingFileHandler settings
MAX_BYTES = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

_logging_configured = False


def setup_logging(level: int = logging.DEBUG) -> None:
    """
    Configure the root logger with a RotatingFileHandler and a StreamHandler.

    - RotatingFileHandler: writes to logs/app.log, max 5MB per file, 3 backups.
    - StreamHandler: outputs to stdout for development convenience.
    - Format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    - Default level: DEBUG

    This function is idempotent; calling it multiple times will not add
    duplicate handlers.
    """
    global _logging_configured
    if _logging_configured:
        return

    # Ensure the logs directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler (stdout) for development
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger instance.

    Ensures logging is configured before returning the logger.
    All modules should call this to get their logger:

        from logging_config import get_logger
        logger = get_logger(__name__)

    Args:
        name: The logger name, typically __name__ of the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    setup_logging()
    return logging.getLogger(name)
