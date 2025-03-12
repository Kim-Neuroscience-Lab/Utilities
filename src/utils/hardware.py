# src/utils/hardware.py
"""
Utilities for hardware detection and management.

This module provides functionality for detecting and gathering information about
system hardware components including GPU, CPU, and memory.

Constants:
    BYTES_TO_GB: Conversion factor from bytes to gigabytes
    MIN_MEMORY_THRESHOLD_MB: Minimum memory threshold for warnings
    GPU_BACKENDS: Supported GPU backend types

Methods:
    detect_gpu_backend: Detect available GPU backend based on platform and hardware.
    get_memory_info: Get system memory information.
    get_cpu_info: Get CPU information.
    get_gpu_info: Get GPU information.
    get_system_info: Get comprehensive system information.
"""

# Standard Library Imports
from logging import Logger
import platform
from typing import Tuple, Optional, Any, Dict, Union, List
from enum import Enum, auto

# Internal Imports
from src.utils.logging import get_logger
from src.core.exceptions.utils.hardware import *
from src.utils.constants import (
    GPU_MEMORY_FRACTION,
    MEMORY_THRESHOLD_MB,
    MB_IN_BYTES,
    GPU_BACKEND,
    HardwareKeys as HK,
)

# Initialize Logger
logger: Logger = get_logger(__name__)


def detect_gpu_backend() -> Tuple[Optional[str], Optional[Any]]:
    """Detect available GPU backend based on platform and hardware.

    Attempts to detect and initialize GPU backends in the following order:
    1. Metal Performance Shaders (MPS) for Apple Silicon
    2. CUDA (via CuPy) for NVIDIA GPUs

    Returns:
        Tuple[Optional[str], Optional[Any]]: A tuple containing:
            - backend_name (str): Name of the detected backend ('cupy' or 'mps')
            - backend_module: The imported backend module
            Returns (None, None) if no GPU is available

    Raises:
        NoGPUAvailable: If GPU detection fails unexpectedly
    """
    system: str = platform.system()
    machine: str = platform.machine()

    try:
        # Try Metal Performance Shaders first on Apple Silicon
        if system == "Darwin" and machine == "arm64":
            try:
                import torch

                if torch.backends.mps.is_built() and torch.backends.mps.is_available():
                    # Test MPS with a small tensor operation
                    try:
                        device = torch.device("mps")
                        test_tensor = torch.ones(1, device=device)
                        del test_tensor
                        torch.mps.empty_cache()  # Clean up
                        logger.info("Apple Silicon GPU detected via MPS")
                        return "mps", torch
                    except Exception as e:
                        logger.debug(f"MPS test failed: {str(e)}")
            except ImportError:
                logger.debug("PyTorch MPS not available")

        # Try CuPy for NVIDIA GPUs
        try:
            import cupy as cp

            logger.info("CUDA GPU detected via CuPy")
            return "cuda", cp
        except ImportError:
            logger.debug("CuPy not available")

        logger.info("No GPU backend detected")
        return None, None

    except Exception as e:
        logger.error(f"Unexpected error during GPU detection: {str(e)}")
        raise NoGPUAvailable(f"GPU detection failed: {str(e)}")


def get_memory_info() -> Dict[str, Union[int, float]]:
    """Get system memory information.

    Returns:
        Dict[str, Union[int, float]]: Dictionary containing:
            - total (int): Total physical memory in bytes
            - available (int): Available memory in bytes
            - percent_used (float): Percentage of memory used
            - used (int): Used memory in bytes
            - free (int): Free memory in bytes

    Raises:
        ImportError: If psutil is not available
    """
    try:
        import psutil

        memory = psutil.virtual_memory()

        # Check for low memory condition
        if memory.available < MIN_MEMORY_THRESHOLD_MB * MB_IN_BYTES:
            logger.warning(
                f"Available memory below {MIN_MEMORY_THRESHOLD_MB}MB threshold"
            )

        return {
            "total": memory.total,
            "available": memory.available,
            "percent_used": memory.percent,
            "used": memory.used,
            "free": memory.free,
            "total_gb": memory.total / GB_IN_BYTES,
            "available_gb": memory.available / GB_IN_BYTES,
        }
    except ImportError:
        logger.error("psutil not available, cannot retrieve memory information")
        return {}


def get_cpu_info() -> Dict[str, Any]:
    """Get CPU information.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - physical_cores (int): Number of physical CPU cores
            - logical_cores (int): Number of logical CPU cores
            - cpu_percent (List[float]): CPU usage percentage per core
            - cpu_freq (psutil._common.scpufreq): CPU frequency information
            - architecture (str): CPU architecture
            - machine (str): Machine hardware name

    Raises:
        ImportError: If psutil is not available
    """
    try:
        import psutil

        cpu_info: Dict[str, Any] = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "cpu_percent": psutil.cpu_percent(interval=0.1, percpu=True),
            "cpu_freq": psutil.cpu_freq(),
            "architecture": platform.processor(),
            "machine": platform.machine(),
        }

        if cpu_info["physical_cores"] is None:
            logger.warning("Could not determine physical core count")

        return cpu_info
    except ImportError:
        logger.error("psutil not available, cannot retrieve CPU information")
        return {
            "physical_cores": None,
            "logical_cores": None,
            "architecture": platform.processor(),
            "machine": platform.machine(),
        }


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - memory_total (int): Total GPU memory in bytes
            - memory_used (int): Used GPU memory in bytes
            - memory_free (int): Free GPU memory in bytes
            - backend_type (str): Type of GPU backend in use
            Additional fields may be present depending on the GPU backend

    Raises:
        RuntimeError: If GPU information cannot be retrieved
    """
    backend, module = detect_gpu_backend()

    if backend is None:
        logger.warning("No GPU backend available")
        return {HK.BACKEND_TYPE.value: GPUBackend.NONE.value}

    try:
        if backend == GPUBackend.CUDA.value:
            import cupy as cp  # type: ignore

            device_props = cp.cuda.runtime.getDeviceProperties(0)
            memory_pool = cp.get_default_memory_pool()

            return {
                HK.BACKEND_TYPE.value: GPUBackend.CUDA.value,
                HK.MEMORY_TOTAL.value: memory_pool.total_bytes,
                HK.MEMORY_USED.value: memory_pool.used_bytes,
                HK.MEMORY_FREE.value: memory_pool.free_bytes,
                "compute_capability": f"{device_props['major']}.{device_props['minor']}",
                "device_name": cp.cuda.runtime.getDeviceName(0),
            }

        elif backend == GPUBackend.MPS.value:
            import torch

            memory_total = torch.mps.recommended_max_memory()
            memory_used = torch.mps.current_allocated_memory()

            return {
                HK.BACKEND_TYPE.value: GPUBackend.MPS.value,
                HK.MEMORY_TOTAL.value: memory_total,
                HK.MEMORY_USED.value: memory_used,
                HK.MEMORY_FREE.value: memory_total - memory_used,
                "device_name": "Apple Silicon",
            }

        return {HK.BACKEND_TYPE.value: GPUBackend.UNKNOWN.value}

    except Exception as e:
        logger.error(f"Error getting GPU information: {str(e)}")
        return {HK.BACKEND_TYPE.value: GPUBackend.ERROR.value, "error": str(e)}


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - hardware (Dict): Hardware information including memory, GPU, and CPU details
            - platform (Dict): Platform-specific information
            - python (Dict): Python runtime information
    """
    return {
        "hardware": {
            "memory": get_memory_info(),
            "cpu": get_cpu_info(),
            "gpu": get_gpu_info(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
    }
