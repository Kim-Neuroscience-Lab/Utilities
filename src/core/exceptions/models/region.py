# src/core/exceptions/models/region.py
"""
Exceptions for the region module.

Classes:
    RegionException: Base exception for the region module.
    SegmentNotFoundError: Exception raised when a segment is not found.
    NoSegmentsFoundError: Exception raised when no segments are found.
    RegionError: Base class for region-related exceptions.
    InvalidRegionIDError: Raised when a region ID is invalid.
    DuplicateRegionError: Raised when attempting to create a duplicate region.
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


class RegionError(Exception):
    """Base class for region-related exceptions."""

    pass


class InvalidRegionIDError(RegionError):
    """Raised when a region ID is invalid."""

    pass


class DuplicateRegionError(RegionError):
    """Raised when attempting to create a duplicate region."""

    def __init__(self, region_id: str):
        self.region_id = region_id
        super().__init__(f"Region {region_id} already exists.")
