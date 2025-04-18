"""
K-Nearest Neighbors imputation implementation using sklearn.

This module provides a service for handling missing data imputation using the
KNN algorithm, leveraging sklearn's KNNImputer while maintaining a clean
service-oriented architecture.
"""

from typing import Optional, Union, Dict, List, Literal, Any
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from dataclasses import dataclass, field
from src.core.exceptions.data.imputation import ImputationError
from src.data.imputation.base_imputer import BaseImputer, BaseImputerConfig


@dataclass
class KNNImputerConfig(BaseImputerConfig):
    """Configuration for KNN imputation.

    Attributes:
        n_neighbors: Number of neighbors to use for imputation
        weights: Weight function used in prediction ('uniform', 'distance')
        metric: Distance metric for the tree ('euclidean', 'manhattan', etc.)
        random_state: Seed for reproducibility
        initial_strategy: Strategy for initial imputation ('mean', 'median', etc.)
        verbose: Whether to print progress messages
        exclude_columns: List of columns to exclude from imputation
        include_columns: List of columns to include in imputation
        class_column: Column indicating the class/group of each datapoint
        separate_imputation: Whether to impute each class separately
        preprocess_numeric: Whether to preprocess numeric columns
    """

    n_neighbors: int = 5
    weights: Literal["uniform", "distance"] = "uniform"
    metric: str = "nan_euclidean"


class KNNImputerService(BaseImputer):
    """Service class for performing KNN imputation."""

    def __init__(self, config: Optional[KNNImputerConfig] = None):
        """Initialize the KNN imputer service.

        Args:
            config: Configuration for KNN imputation
        """
        super().__init__(config or KNNImputerConfig())
        self.config: KNNImputerConfig

    def _create_imputer(self, **kwargs) -> KNNImputer:
        """Create and configure the KNNImputer instance.

        Returns:
            Configured KNNImputer instance
        """
        return KNNImputer(
            n_neighbors=self.config.n_neighbors,
            weights=self.config.weights,
            metric=self.config.metric,
            **kwargs
        )

    def _postprocess_imputed_data(self, imputed_data: pd.DataFrame) -> pd.DataFrame:
        """Post-process imputed data to ensure consistent data types.

        This ensures all numeric columns are properly converted to float values.

        Args:
            imputed_data: Imputed DataFrame

        Returns:
            Post-processed DataFrame
        """
        result = imputed_data.copy()

        # Get columns that should be numeric
        numeric_columns = []
        for col in result.columns:
            if any(prefix in col for prefix in ["vsv", "input"]):
                numeric_columns.append(col)

        # Convert numeric columns to float
        for col in numeric_columns:
            try:
                result[col] = pd.to_numeric(result[col], errors="coerce")
            except Exception as e:
                # If conversion fails, leave the column as is
                pass

        return result

    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in the input data.

        Overrides the base impute method to add post-processing step.

        Args:
            data: Input DataFrame with missing values

        Returns:
            DataFrame with imputed values
        """
        # Call the parent class impute method
        imputed_data = super().impute(data)

        # Apply post-processing
        imputed_data = self._postprocess_imputed_data(imputed_data)

        return imputed_data
