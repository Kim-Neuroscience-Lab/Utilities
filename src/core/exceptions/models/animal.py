# src/core/exceptions/models/animal.py
"""
Exceptions for the animal module.
"""


class AnimalException(Exception):
    """Base exception for the animal module."""

    pass


class NoRegionsFoundError(AnimalException):
    """Exception raised when no regions are found."""

    def __init__(self):
        super().__init__("No regions found.")


class RegionNotFoundError(AnimalException):
    """Exception raised when a region is not found."""

    def __init__(self, region_id: str):
        self.region_id = region_id
        super().__init__(f"Region {region_id} not found.")


class RegionAlreadyExistsError(AnimalException):
    """Exception raised when a region already exists."""

    def __init__(self, region_id: str):
        self.region_id = region_id
        super().__init__(f"Region {region_id} already exists.")


class AnimalError(Exception):
    """Base class for animal-related exceptions."""

    pass


class InvalidAnimalIDError(AnimalError):
    """Raised when an animal ID is invalid."""

    pass


class DuplicateAnimalError(AnimalError):
    """Raised when attempting to create a duplicate animal."""

    pass
