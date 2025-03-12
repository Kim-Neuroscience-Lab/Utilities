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

# Memory constants
MEMORY_THRESHOLD_MB: Final[int] = 1024  # 1GB
DEFAULT_BUFFER_SIZE: Final[int] = 8 * MB_IN_BYTES
DEFAULT_MAX_BUFFER: Final[int] = 128 * MB_IN_BYTES
DEFAULT_MEMORY_FRACTION: Final[float] = 0.8
MAX_GPU_MEMORY_USAGE: Final[float] = 0.8  # Maximum fraction of GPU memory to use
GPU_MEMORY_FRACTION: Final[float] = 0.7  # Target fraction of GPU memory for processing
SAFETY_MARGIN: Final[float] = 0.1
BUFFER_SIZE: Final[int] = DEFAULT_BUFFER_SIZE


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


# Processing constants
BATCH_SIZE: Final[int] = 32  # Optimal batch size for general processing
GPU_BATCH_SIZE: Final[int] = 16  # Optimal batch size for GPU processing
ROI_SIZE_THRESHOLD: Final[int] = (
    500000  # Threshold for GPU vs fast method (reduced for better GPU utilization)
)
GPU_MEMORY_THRESHOLD: Final[int] = (
    1024 * 1024 * 1024
)  # 1GB minimum GPU memory for batch processing

GPU_BACKEND = GPUBackend
