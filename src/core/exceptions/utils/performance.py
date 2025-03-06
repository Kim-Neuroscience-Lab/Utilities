# src/core/exceptions/performance.py
"""
Exceptions for performance-related errors.

This module provides custom exceptions for performance-related errors.
"""


class PerformanceError(Exception):
    """Base exception for performance-related errors."""

    pass


class MemoryFractionError(PerformanceError):
    """Exception raised when memory fraction is not between 0 and 1."""

    def __init__(
        self, message: str = "Memory fraction must be between 0 and 1."
    ) -> None:
        self.message: str = message
        super().__init__(self.message)
