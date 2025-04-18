"""Exceptions for the imputation module."""


class ImputationError(Exception):
    """Base class for imputation-related exceptions."""

    pass


class StrategyNotFoundError(ImputationError):
    """Raised when an imputation strategy is not found for a column."""

    pass


class ConvergenceError(ImputationError):
    """Raised when the MICE algorithm fails to converge."""

    pass
