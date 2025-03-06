# src/core/exceptions/hardware.py
"""
Exceptions for hardware detection and management.

This module provides custom exceptions for hardware-related errors.

Classes:
    NoGPUAvailable: Exception raised when no GPU is available.
"""


class HardwareError(Exception):
    """Base exception for hardware-related errors."""

    pass


class NoGPUAvailable(HardwareError):
    """Exception raised when no GPU is available."""

    def __init__(self, message: str = "No GPU is available.") -> None:
        self.message: str = message
        super().__init__(self.message)
