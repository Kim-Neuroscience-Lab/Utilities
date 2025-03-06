# src/utils/constants.py
"""
Constants for the application.

This module contains constants used throughout the application.
"""

# Standard Library Imports
from enum import Enum
from typing import Final

# Memory conversion constants
MB_IN_BYTES: Final[int] = 1024 * 1024
GB_IN_BYTES: Final[int] = MB_IN_BYTES * 1024
TB_IN_BYTES: Final[int] = GB_IN_BYTES * 1024

# Memory thresholds
MIN_MEMORY_THRESHOLD_BYTES: Final[float] = 1024 * 1024 * 100
MIN_MEMORY_THRESHOLD_MB: Final[float] = MIN_MEMORY_THRESHOLD_BYTES / MB_IN_BYTES
MIN_MEMORY_THRESHOLD_GB: Final[float] = MIN_MEMORY_THRESHOLD_MB / 1024
MIN_MEMORY_THRESHOLD_TB: Final[float] = MIN_MEMORY_THRESHOLD_GB / 1024


# Hardware constants
class GPUBackend(str, Enum):
    """GPU backend types supported by the application."""

    CUDA = "cuda"
    MPS = "mps"
    NONE = "none"
    UNKNOWN = "unknown"
    ERROR = "error"


class HardwareKeys(str, Enum):
    """Keys used in hardware information dictionaries."""

    GPU = "gpu"
    MEMORY = "memory"
    CPU = "cpu"
    BACKEND_TYPE = "backend_type"
    MEMORY_TOTAL = "memory_total"
    MEMORY_USED = "memory_used"
    MEMORY_FREE = "memory_free"
    AVAILABLE = "available"
    TOTAL = "total"


# Default values
DEFAULT_BUFFER_SIZE: Final[int] = MB_IN_BYTES
DEFAULT_MAX_BUFFER: Final[int] = GB_IN_BYTES
DEFAULT_MEMORY_FRACTION: Final[float] = 0.8
SAFETY_MARGIN: Final[float] = 0.9  # 90% of calculated optimal size
