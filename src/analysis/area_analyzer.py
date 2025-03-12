# src/analysis/area_analyzer.py
"""
Area Analyzer module for computing areas from pickle files.

This module provides functionality to analyze areas across multiple brain segments
and regions, following the structure of the Allen Brain Atlas.
"""

# Standard Library Imports
from typing import Dict, Any

# External Imports
from pydantic import BaseModel, Field, ConfigDict

# Internal Imports
from src.utils.constants import MB_IN_BYTES
from src.core.models.animal import Animal
from src.utils.logging import get_logger

# Initialize Logger
logger = get_logger(__name__)


class AreaAnalyzer(BaseModel):
    """Base class for area analysis.

    This class provides common functionality for analyzing areas from pickle files.
    It is meant to be subclassed by specific analyzers like ROIAreaAnalyzer.

    Attributes:
        input_dir: Directory containing pickle files
        max_workers: Maximum number of worker threads for parallel processing
        use_gpu: Whether to use GPU acceleration if available
    """

    input_dir: str = Field(..., description="Directory containing pickle files")
    max_workers: int = Field(None, description="Maximum number of worker threads")
    use_gpu: bool = Field(False, description="Whether to use GPU acceleration")

    model_config = ConfigDict(arbitrary_types_allowed=True)
