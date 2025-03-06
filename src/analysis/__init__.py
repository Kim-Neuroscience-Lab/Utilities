# src/analysis/__init__.py
"""
Analysis package for various data analysis operations.

This package provides a framework for different types of analysis,
with specific implementations for area analysis and ROI processing.
"""

from src.analysis.area.analyzer import AreaAnalyzer, ROIAreaAnalyzer
from src.analysis.area.config import AreaConfig, ROIAreaConfig
from src.analysis.area.results import AreaResult, ROIAreaResult
from src.analysis.utils.exceptions import AnalysisError, GPUNotAvailableError

__all__ = [
    "AreaAnalyzer",
    "ROIAreaAnalyzer",
    "AreaConfig",
    "ROIAreaConfig",
    "AreaResult",
    "ROIAreaResult",
    "AnalysisError",
    "GPUNotAvailableError",
]
