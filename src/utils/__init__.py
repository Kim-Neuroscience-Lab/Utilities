# src/utils/__init__.py
"""
Utility functions and classes for the application.
"""

from src.utils.hardware import (
    detect_gpu_backend,
    get_memory_info,
    get_cpu_info,
    get_system_info,
)
from src.utils.logging import get_logger, setup_logging

__all__ = [
    # Hardware
    "detect_gpu_backend",
    "get_memory_info",
    "get_cpu_info",
    "get_system_info",
    # Logging
    "get_logger",
    "setup_logging",
]
