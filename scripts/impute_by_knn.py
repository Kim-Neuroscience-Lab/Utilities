# src/scripts/impute_by_knn.py
"""
KNN imputation script for neuroscience data.

This script demonstrates the use of KNN imputation on neuroscience datasets,
particularly focusing on handling missing values in mice brain imaging data.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.data.imputation import KNNImputerService, KNNImputerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="KNN imputation for neuroscience datasets"
    )

    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to input CSV file"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to output CSV file (default: input_imputed.csv)",
    )

    parser.add_argument(
        "--neighbors",
        "-n",
        type=int,
        default=5,
        help="Number of neighbors to use for KNN imputation (default: 5)",
    )

    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        choices=["uniform", "distance"],
        default="uniform",
        help="Weight function used in prediction (default: uniform)",
    )

    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        default="nan_euclidean",
        help="Distance metric for the tree (default: nan_euclidean)",
    )

    parser.add_argument(
        "--exclude",
        "-e",
        type=str,
        nargs="+",
        help="Columns to exclude from imputation (default: Age age_categorical Animal)",
    )

    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        help="Columns to include in imputation (if specified, only these columns will be imputed)",
    )

    parser.add_argument(
        "--class-column",
        "-c",
        type=str,
        help="Column to use for class-based imputation (e.g., age_categorical)",
    )

    parser.add_argument(
        "--separate-imputation",
        "-s",
        action="store_true",
        help="Perform separate imputation for each class",
    )

    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip preprocessing numeric columns",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=137,
        help="Random state for reproducibility (default: 137)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    return parser.parse_args()


def main():
    """Run the KNN imputation script."""
    # Parse command line arguments
    args = parse_args()

    # Set up default output path if not specified
    if not args.output:
        input_path = Path(args.input)
        args.output = str(
            input_path.parent / f"{input_path.stem}_imputed{input_path.suffix}"
        )

    # Set default excluded columns if not specified
    if args.exclude is None:
        args.exclude = ["Age", "age_categorical", "Animal"]

    # Load the input data
    try:
        logger.info(f"Loading data from {args.input}")
        data = pd.read_csv(args.input)
        logger.info(f"Loaded data with shape {data.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

    # Check for missing values
    missing_values = data.isna().sum().sum()
    if missing_values == 0:
        logger.warning("No missing values found in the dataset. Nothing to impute.")
        logger.info(f"Saving original data to {args.output}")
        data.to_csv(args.output, index=False)
        sys.exit(0)

    logger.info(f"Found {missing_values} missing values in the dataset")

    # Configure KNN imputation
    logger.info("Configuring KNN imputation")
    config = KNNImputerConfig(
        n_neighbors=args.neighbors,
        weights=args.weights,
        metric=args.metric,
        exclude_columns=args.exclude,
        include_columns=args.include,
        class_column=args.class_column,
        separate_imputation=args.separate_imputation,
        preprocess_numeric=not args.no_preprocess,
        random_state=args.random_state,
        verbose=args.verbose,
    )

    # Create the imputer service
    imputer = KNNImputerService(config)

    # Perform imputation
    try:
        logger.info("Performing KNN imputation")
        imputed_data = imputer.impute(data)
        logger.info("Imputation completed successfully")
    except Exception as e:
        logger.error(f"Error during imputation: {e}")
        sys.exit(1)

    # Check if all missing values were imputed
    remaining_missing = imputed_data.isna().sum().sum()
    if remaining_missing > 0:
        logger.warning(f"{remaining_missing} missing values remain after imputation")
    else:
        logger.info("All missing values were successfully imputed")

    # Get imputation statistics
    stats = imputer.get_imputation_statistics()
    logger.info(f"Total missing values: {stats.get('total_missing_values', 0)}")

    # Display class-specific statistics if applicable
    if args.class_column and args.separate_imputation and "class_statistics" in stats:
        logger.info("Class-specific imputation statistics:")
        for class_value, class_stats in stats["class_statistics"].items():
            logger.info(
                f"Class {class_value}: {class_stats['total_missing_values']} missing values"
            )

    # Save the imputed data
    try:
        logger.info(f"Saving imputed data to {args.output}")
        imputed_data.to_csv(args.output, index=False)
        logger.info("Imputed data saved successfully")
    except Exception as e:
        logger.error(f"Error saving imputed data: {e}")
        sys.exit(1)

    logger.info("KNN imputation completed successfully")


if __name__ == "__main__":
    main()
