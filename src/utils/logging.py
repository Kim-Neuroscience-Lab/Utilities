# src/utils/logging.py
"""
Logging utilities for the application.

Methods:
    setup_logging: Configure logging for the application.
    get_logger: Get a logger instance.
"""

# Standard Library Imports
import logging
from typing import Optional


def setup_logging(
    log_level: Optional[str] = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Logging level (e.g., "INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL").
        log_file: Path to the log file (optional).
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()) if log_level else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    """
    return logging.getLogger(name)
