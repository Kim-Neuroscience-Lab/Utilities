# src/utils/performance.py
"""
Performance utilities for the application.

Methods:
    batch_process: Process items in batches to reduce memory usage.
    timed_execution: Decorator to time function execution.
    get_optimal_batch_size: Calculate optimal batch size based on system resources.
"""

# Standard Library Imports
from typing import List, Any, Callable, Optional, Literal, Union
from logging import Logger
import math

# Internal Imports
from src.utils.logging import get_logger
from src.utils.hardware import get_system_info
from Utilities.src.core.exceptions.utils.hardware import NoGPUAvailable
from Utilities.src.core.exceptions.utils.performance import MemoryFractionError
from src.utils.constants import (
    MB_IN_BYTES,
    GB_IN_BYTES,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_MAX_BUFFER,
    DEFAULT_MEMORY_FRACTION,
    SAFETY_MARGIN,
    GPUBackend,
    HardwareKeys as HK,
)

# Initialize Logger
logger: Logger = get_logger(__name__)


def batch_process(
    items: List[Any],
    batch_size: int,
    process_func: Callable[[List[Any], Any], List[Any]],
    **kwargs: Any,
) -> List[Any]:
    """Process items in batches to reduce memory usage.

    Args:
        items: List of items to process
        batch_size: Number of items to process in each batch
        process_func: Function to process each batch
        **kwargs: Additional arguments to pass to process_func

    Returns:
        List of results from processing each batch
    """
    results: List[Any] = []
    for i in range(0, len(items), batch_size):
        batch: List[Any] = items[i : i + batch_size]
        batch_results: List[Any] = process_func(batch, **kwargs)  # type: ignore
        results.extend(batch_results)
    return results


def timed_execution(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to time function execution.

    Args:
        func: Function to time

    Returns:
        Wrapped function that logs execution time
    """
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time: float = time.time()
        result: Any = func(*args, **kwargs)
        end_time: float = time.time()
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper


def get_optimal_batch_size(
    items: List[Any],
    process_func: Callable[[List[Any], Any], List[Any]],
    device: Optional[Literal["cpu", "gpu"]] = None,
    buffer_size: Optional[int] = DEFAULT_BUFFER_SIZE,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
    **kwargs: Any,
) -> int:
    """Get the optimal batch size for processing items based on system resources.

    This function calculates the optimal batch size based on:
    1. Available system memory (CPU or GPU)
    2. Size of input items
    3. Target device (CPU or GPU)
    4. Buffer size for processing overhead

    Args:
        items: List of items to process
        process_func: Function to process each batch
        device: Target device for processing ("cpu" or "gpu")
        buffer_size: Size of buffer to reserve for processing overhead
        memory_fraction: Fraction of available memory to use (0.0 to 1.0)
        **kwargs: Additional arguments to pass to process_func

    Returns:
        int: Optimal batch size for processing

    Raises:
        MemoryFractionError: If memory_fraction is not between 0 and 1
        NoGPUAvailable: If device is "gpu" but no GPU is available
    """
    if not 0 < memory_fraction <= 1:
        raise MemoryFractionError("memory_fraction must be between 0 and 1")

    # Get system information
    system_info: dict = get_system_info()
    available_memory: int = 0

    # Set default device to CPU if not specified
    if device is None:
        device = "cpu"
        logger.info("No device specified, using CPU")

    # Calculate available memory based on device
    if device == "gpu":
        gpu_info = system_info[HK.HARDWARE.value][HK.GPU.value]
        if gpu_info[HK.BACKEND_TYPE.value] == GPUBackend.NONE.value:
            raise NoGPUAvailable("GPU specified but no GPU available")

        available_memory = gpu_info[HK.MEMORY_FREE.value]
        if not available_memory:
            logger.warning("Could not determine GPU memory, falling back to CPU")
            device = "cpu"

    if device == "cpu":
        memory_info = system_info[HK.HARDWARE.value][HK.MEMORY.value]
        available_memory = memory_info[HK.AVAILABLE.value]

    # Reserve buffer memory
    if buffer_size:
        available_memory = max(0, available_memory - buffer_size)

    # Calculate memory per item (sample first item)
    import sys

    sample_item = items[0] if items else None
    if sample_item is None:
        return 1  # Return minimum batch size for empty list

    try:
        import numpy as np

        if isinstance(sample_item, np.ndarray):
            item_size = sample_item.nbytes
        else:
            item_size = sys.getsizeof(sample_item)
    except ImportError:
        item_size = sys.getsizeof(sample_item)

    # Calculate batch size
    usable_memory = int(available_memory * memory_fraction)
    optimal_batch_size = max(
        1,
        min(
            len(items),  # Don't exceed list size
            usable_memory // max(1, item_size),  # Prevent division by zero
        ),
    )

    # Add safety margin
    optimal_batch_size = int(optimal_batch_size * SAFETY_MARGIN)

    logger.debug(
        f"Calculated optimal batch size: {optimal_batch_size} "
        f"(Device: {device}, Available Memory: {available_memory / MB_IN_BYTES:.2f}MB)"
    )

    return optimal_batch_size


def get_optimal_buffer_size(
    items: List[Any],
    process_func: Callable[[List[Any], Any], List[Any]],
    device: str = "cpu",
    min_buffer: int = DEFAULT_BUFFER_SIZE,
    max_buffer: int = DEFAULT_MAX_BUFFER,
    **kwargs: Any,
) -> int:
    """Calculate optimal buffer size based on system resources.

    Args:
        items: List of items to process
        process_func: Function to process items
        device: Target device for processing
        min_buffer: Minimum buffer size in bytes
        max_buffer: Maximum buffer size in bytes
        **kwargs: Additional arguments for process_func

    Returns:
        int: Optimal buffer size in bytes
    """
    system_info: dict = get_system_info()
    total_memory: int = 0

    # Get total memory based on device
    if device == "gpu":
        gpu_info = system_info[HK.HARDWARE.value][HK.GPU.value]
        total_memory = gpu_info.get(HK.MEMORY_TOTAL.value, 0)
        if not total_memory:
            logger.warning("Could not determine GPU memory, using CPU memory")
            device = "cpu"

    if device == "cpu":
        total_memory = system_info[HK.HARDWARE.value][HK.MEMORY.value][HK.TOTAL.value]

    # Calculate buffer as percentage of total memory (5% by default)
    buffer_size = int(total_memory * 0.05)

    # Clamp buffer size between min and max
    buffer_size = max(min_buffer, min(buffer_size, max_buffer))

    logger.debug(f"Calculated optimal buffer size: {buffer_size / MB_IN_BYTES:.2f}MB")

    return buffer_size
