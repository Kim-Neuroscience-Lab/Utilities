# src/core/exceptions/region.py
"""
Exceptions for the region module.

Classes:
    RegionException: Base exception for the region module.
    SegmentNotFoundError: Exception raised when a segment is not found.
    NoSegmentsFoundError: Exception raised when no segments are found.
"""


class RegionException(Exception):
    """Base exception for the region module."""

    pass


class SegmentNotFoundError(RegionException):
    """Exception raised when a segment is not found."""

    def __init__(self, segment_id: str):
        self.segment_id = segment_id
        super().__init__(f"Segment {segment_id} not found.")


class NoSegmentsFoundError(RegionException):
    """Exception raised when no segments are found."""

    def __init__(self):
        super().__init__("No segments found.")
