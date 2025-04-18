"""
Base imputation module for handling missing data.

This module provides a base class for imputation services with common functionality
for handling missing data based on different strategies and configurations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, List, Literal, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from src.core.exceptions.data.imputation import ImputationError


@dataclass
class BaseImputerConfig:
    """Base configuration for imputation services.

    Attributes:
        random_state: Seed for reproducibility
        initial_strategy: Strategy for initial imputation ('mean', 'median', etc.)
        exclude_columns: List of columns to exclude from imputation
        include_columns: List of columns to include in imputation (if None, include all non-excluded columns)
        class_column: Column indicating the class/group of each datapoint (if None, don't group by class)
        separate_imputation: Whether to impute each class separately
        preprocess_numeric: Whether to preprocess numeric columns (convert strings with commas to floats)
        verbose: Whether to print progress messages
    """

    random_state: Optional[int] = 137
    initial_strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean"
    exclude_columns: List[str] = field(default_factory=list)
    include_columns: Optional[List[str]] = None
    class_column: Optional[str] = None
    separate_imputation: bool = False
    preprocess_numeric: bool = True
    verbose: bool = False


class BaseImputer(ABC):
    """Abstract base class for imputation services."""

    def __init__(self, config: Optional[BaseImputerConfig] = None):
        """Initialize the base imputer.

        Args:
            config: Configuration for imputation
        """
        self.config = config or BaseImputerConfig()
        self._missing_mask = None
        self._excluded_data = None
        self._class_data = None
        self._imputation_stats = {}

    @abstractmethod
    def _create_imputer(self, **kwargs) -> Any:
        """Create and configure the imputer instance.

        This method should be implemented by derived classes to create
        the specific imputer instance used for the imputation.

        Returns:
            Configured imputer instance
        """
        pass

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
        invalid_excluded = [
            col for col in self.config.exclude_columns if col not in data.columns
        ]
        if invalid_excluded:
            raise ImputationError(
                f"Excluded columns not found in data: {invalid_excluded}"
            )

        # Validate included columns exist in the data
        if self.config.include_columns:
            invalid_included = [
                col for col in self.config.include_columns if col not in data.columns
            ]
            if invalid_included:
                raise ImputationError(
                    f"Included columns not found in data: {invalid_included}"
                )

        # Validate class column exists in the data
        if self.config.class_column and self.config.class_column not in data.columns:
            raise ImputationError(
                f"Class column '{self.config.class_column}' not found in data"
            )

    def _preprocess_numeric_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess numeric columns by removing commas and converting to float.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with preprocessed numeric columns
        """
        if not self.config.preprocess_numeric:
            return data

        df = data.copy()

        # Identify potential numeric columns that are currently objects
        object_cols = df.select_dtypes(include=["object"]).columns

        # Filter by excluded and included columns
        numeric_cols = [
            col
            for col in object_cols
            if col not in self.config.exclude_columns
            and (
                self.config.include_columns is None
                or col in self.config.include_columns
            )
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

    def _get_columns_for_imputation(self, data: pd.DataFrame) -> List[str]:
        """Get the list of columns to be used for imputation.

        Args:
            data: Input DataFrame

        Returns:
            List of column names to impute
        """
        if self.config.include_columns:
            # If include_columns is specified, use those columns
            return [
                col
                for col in self.config.include_columns
                if col not in self.config.exclude_columns
            ]
        else:
            # Otherwise, use all columns except excluded ones
            return [
                col for col in data.columns if col not in self.config.exclude_columns
            ]

    def _separate_columns(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
        """Separate columns based on configuration.

        Args:
            data: Input DataFrame

        Returns:
            Tuple of (data for imputation, excluded data, class data if applicable)
        """
        # Get columns for imputation
        imputation_cols = self._get_columns_for_imputation(data)

        # Extract class column if specified
        class_data = None
        if self.config.class_column:
            class_data = data[self.config.class_column].copy()

        # Extract excluded columns
        excluded_cols = list(set(data.columns) - set(imputation_cols))
        excluded_data = data[excluded_cols].copy() if excluded_cols else pd.DataFrame()

        # Extract data for imputation
        imputation_data = data[imputation_cols].copy()

        return imputation_data, excluded_data, class_data

    def _impute_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in a single DataFrame.

        This method should be overridden by derived classes to implement
        the specific imputation algorithm.

        Args:
            data: DataFrame to impute

        Returns:
            Imputed DataFrame
        """
        # Create an imputer for this data
        imputer = self._create_imputer()

        # Impute the data
        imputed_values = imputer.fit_transform(data)

        # Convert back to DataFrame with original column names
        return pd.DataFrame(imputed_values, columns=data.columns, index=data.index)

    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in the input data.

        Args:
            data: Input DataFrame with missing values

        Returns:
            DataFrame with imputed values

        Raises:
            ImputationError: If imputation fails
        """
        try:
            self._validate_input(data)

            # Initialize imputation statistics
            self._imputation_stats = {}

            # Create a copy to avoid modifying the original data
            df_copy = data.copy()

            # Preprocess numeric columns
            df_copy = self._preprocess_numeric_columns(df_copy)

            # Separate columns
            imputation_data, self._excluded_data, self._class_data = (
                self._separate_columns(df_copy)
            )

            # Create missing values mask for the data to be imputed
            self._missing_mask = self._create_missing_mask(imputation_data)

            # Collect imputation statistics
            self._collect_statistics(imputation_data)

            # Impute missing values
            if self.config.separate_imputation and self.config.class_column:
                # Impute each class separately
                imputed_data = self._impute_by_class(imputation_data)
            else:
                # Impute the entire dataset at once
                imputed_data = self._impute_dataframe(imputation_data)

            # Combine imputed data with excluded data
            result = imputed_data.copy()
            if not self._excluded_data.empty:
                # Make sure indices match for concatenation
                excluded_data = self._excluded_data.loc[result.index]
                result = pd.concat([result, excluded_data], axis=1)

                # Restore original column order
                result = result[data.columns]

            return result

        except Exception as e:
            raise ImputationError(f"Imputation failed: {str(e)}")

    def _impute_by_class(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values separately for each class.

        Args:
            data: DataFrame to impute

        Returns:
            Imputed DataFrame
        """
        if self._class_data is None:
            raise ImputationError("Class data is missing for class-based imputation")

        # Create a new DataFrame to store the imputed results
        imputed_data = pd.DataFrame(index=data.index, columns=data.columns)

        # Impute each class separately
        for class_value in self._class_data.unique():
            # Get indices for this class
            class_indices = self._class_data[self._class_data == class_value].index

            # Extract data for this class
            class_data = data.loc[class_indices].copy()

            # Skip if no data or no missing values
            if class_data.empty or not class_data.isna().any().any():
                imputed_data.loc[class_indices] = class_data
                continue

            # Impute missing values for this class
            try:
                imputed_class_data = self._impute_dataframe(class_data)
                # Store imputed values for this class
                imputed_data.loc[class_indices] = imputed_class_data
            except Exception as e:
                if self.config.verbose:
                    print(f"Warning: Failed to impute class {class_value}: {str(e)}")
                # Fall back to using the original data for this class
                imputed_data.loc[class_indices] = class_data

        return imputed_data

    def _collect_statistics(self, data: pd.DataFrame) -> None:
        """Collect statistics about the imputation process.

        Args:
            data: DataFrame being imputed
        """
        if self._missing_mask is None:
            return

        # Calculate total missing values
        total_missing = self._missing_mask.sum().sum()
        missing_by_col = self._missing_mask.sum().to_dict()
        missing_pct = (
            self._missing_mask.sum() / len(self._missing_mask) * 100
        ).to_dict()

        # Store overall statistics
        self._imputation_stats = {
            "total_missing_values": total_missing,
            "missing_by_column": missing_by_col,
            "missing_percentage": missing_pct,
        }

        # Add class-specific statistics if applicable
        if self.config.separate_imputation and self._class_data is not None:
            class_stats = {}
            for class_value in self._class_data.unique():
                # Get indices for this class
                class_indices = self._class_data[self._class_data == class_value].index

                # Calculate missing values for this class
                class_mask = self._missing_mask.loc[class_indices]
                class_stats[class_value] = {
                    "total_missing_values": class_mask.sum().sum(),
                    "missing_by_column": class_mask.sum().to_dict(),
                    "missing_percentage": (
                        (class_mask.sum() / len(class_mask) * 100).to_dict()
                        if len(class_mask) > 0
                        else {}
                    ),
                }

            self._imputation_stats["class_statistics"] = class_stats

    def get_imputation_statistics(self) -> Dict:
        """Get statistics about the imputation process.

        Returns:
            Dictionary containing imputation statistics
        """
        return self._imputation_stats
