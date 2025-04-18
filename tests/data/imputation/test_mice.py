#!/usr/bin/env python3
"""Test script for MICE imputation with real data."""

import pandas as pd
import numpy as np
from src.data.imputation.mice_imputer import MICEService, MICEConfig


def main():
    """Run MICE imputation test on real data."""
    # Configure MICE with excluded columns
    config = MICEConfig(
        max_iterations=10,
        random_state=42,
        exclude_columns=["Age", "age_categorical", "Animal"],
        verbose=True,
    )
    service = MICEService(config)

    # Load test data
    data = pd.read_csv("tests/test_data/mice_data_vsv_h2b.csv")
    print("\nOriginal data shape:", data.shape)
    print("\nMissing values before imputation:")
    print(data.isna().sum()[data.isna().sum() > 0])

    # Perform imputation
    imputed_df = service.impute(data)

    # Get statistics
    stats = service.get_imputation_statistics()

    print("\nImputation Statistics:")
    print(f"Total missing values imputed: {stats['total_missing_values']}")
    print("\nMissing values by column:")
    for col, count in stats["missing_by_column"].items():
        if count > 0:
            print(f"{col}: {count}")

    print("\nMissing values after imputation:")
    print(imputed_df.isna().sum()[imputed_df.isna().sum() > 0])

    # Verify excluded columns remained unchanged
    for col in config.exclude_columns:
        if not data[col].equals(imputed_df[col]):
            print(f"\nWarning: Excluded column {col} was modified!")

    print("\nImputation completed successfully!")


if __name__ == "__main__":
    main()
