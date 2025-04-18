"""
Multiple Imputation by Chained Equations (MICE) implementation using sklearn.

This module provides a service for handling missing data imputation using the
MICE algorithm, leveraging sklearn's IterativeImputer while maintaining a clean
service-oriented architecture.
"""

from typing import Optional, Union, Dict, List, Literal, Any
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # Required import
from sklearn.impute import IterativeImputer
from dataclasses import dataclass, field
from src.core.exceptions.data.imputation import ImputationError
from src.data.imputation.base_imputer import BaseImputer, BaseImputerConfig


@dataclass
class MICEConfig(BaseImputerConfig):
    """Configuration for MICE imputation.

    Attributes:
        max_iterations: Maximum number of imputation rounds
        random_state: Seed for reproducibility
        initial_strategy: Strategy for initial imputation ('mean', 'median', etc.)
        n_nearest_features: Number of nearest features to use for imputation
        verbose: Whether to print progress messages
        exclude_columns: List of columns to exclude from imputation
        include_columns: List of columns to include in imputation
        class_column: Column indicating the class/group of each datapoint
        separate_imputation: Whether to impute each class separately
        preprocess_numeric: Whether to preprocess numeric columns
    """

    max_iterations: int = 10
    n_nearest_features: Optional[int] = None


class MICEService(BaseImputer):
    """Service class for performing MICE imputation."""

    def __init__(self, config: Optional[MICEConfig] = None):
        """Initialize the MICE service.

        Args:
            config: Configuration for MICE imputation
        """
        super().__init__(config or MICEConfig())
        self.config: MICEConfig
        # Initialize the imputer
        self._imputer = self._create_imputer()

    def _create_imputer(self, **kwargs) -> IterativeImputer:
        """Create and configure the IterativeImputer instance.

        Returns:
            Configured IterativeImputer instance
        """
        return IterativeImputer(
            max_iter=self.config.max_iterations,
            random_state=self.config.random_state,
            initial_strategy=self.config.initial_strategy,
            n_nearest_features=self.config.n_nearest_features,
            verbose=self.config.verbose,
            **kwargs,
        )

    def _validate_input(self, data: pd.DataFrame) -> None:
        """Validate input data.

        Args:
            data: Input DataFrame to validate

        Raises:
            ImputationError: If input validation fails
        """
        if not isinstance(data, pd.DataFrame):
            raise ImputationError("Input must be a pandas DataFrame")
        if data.empty:
            raise ImputationError("Input DataFrame is empty")

        # Validate excluded columns exist in the data
        invalid_columns = [
            col for col in self.config.exclude_columns if col not in data.columns
        ]
        if invalid_columns:
            raise ImputationError(
                f"Excluded columns not found in data: {invalid_columns}"
            )

    def _preprocess_numeric_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess numeric columns by removing commas and converting to float.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with preprocessed numeric columns
        """
        df = data.copy()

        # Identify numeric columns (excluding excluded columns)
        numeric_cols = df.select_dtypes(include=["object"]).columns
        numeric_cols = [
            col for col in numeric_cols if col not in self.config.exclude_columns
        ]

        for col in numeric_cols:
            try:
                # Remove commas and convert to float
                df[col] = df[col].replace({",": ""}, regex=True).astype(float)
            except (ValueError, TypeError):
                # If conversion fails, leave the column as is
                continue

        return df

    def _create_missing_mask(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create a mask of missing values.

        Args:
            data: Input DataFrame

        Returns:
            Boolean mask of missing values
        """
        return data.isna()

    def _separate_excluded_columns(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Separate excluded columns from the data.

        Args:
            data: Input DataFrame

        Returns:
            Tuple of (data without excluded columns, excluded columns data)
        """
        if not self.config.exclude_columns:
            return data, pd.DataFrame()

        excluded_data = data[self.config.exclude_columns].copy()
        remaining_data = data.drop(columns=self.config.exclude_columns)
        return remaining_data, excluded_data

    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform MICE imputation on the input data.

        Args:
            data: Input DataFrame with missing values

        Returns:
            DataFrame with imputed values

        Raises:
            ImputationError: If imputation fails
        """
        try:
            self._validate_input(data)

            # Create a copy to avoid modifying the original data
            df_copy = data.copy()

            # Preprocess numeric columns
            df_copy = self._preprocess_numeric_columns(df_copy)

            # Separate excluded columns
            df_to_impute, self._excluded_data = self._separate_excluded_columns(df_copy)

            # Create missing values mask for the data to be imputed
            self._missing_mask = self._create_missing_mask(df_to_impute)

            # Perform imputation
            imputed_values = self._imputer.fit_transform(df_to_impute)

            # Convert imputed values to DataFrame with original columns
            imputed_df = pd.DataFrame(imputed_values, columns=df_to_impute.columns)

            # Combine imputed data with excluded columns
            if not self._excluded_data.empty:
                imputed_df = pd.concat([imputed_df, self._excluded_data], axis=1)
                # Restore original column order
                imputed_df = imputed_df[data.columns]

            return imputed_df

        except Exception as e:
            raise ImputationError(f"Imputation failed: {str(e)}")

    def get_imputation_statistics(self) -> Dict:
        """Get statistics about the imputation process.

        Returns:
            Dictionary containing imputation statistics
        """
        if self._missing_mask is None:
            return {}

        stats = {
            "total_missing_values": self._missing_mask.sum().sum(),
            "missing_by_column": self._missing_mask.sum().to_dict(),
            "missing_percentage": (
                self._missing_mask.sum() / len(self._missing_mask) * 100
            ).to_dict(),
        }

        if self._excluded_data is not None and not self._excluded_data.empty:
            excluded_missing = self._excluded_data.isna().sum()
            stats["excluded_columns_missing"] = excluded_missing.to_dict()

        return stats
