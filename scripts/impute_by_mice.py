# src/scripts/impute_by_mice.py
"""
Script for performing Multiple Imputation by Chained Equations (MICE) on data files.

This script provides functionality for imputing missing values in CSV files using the
MICE algorithm. It supports various imputation strategies and configuration options.

Example usage:
    # Basic usage
    python impute_by_mice.py input.csv

    # With custom settings
    python impute_by_mice.py input.csv --max-iterations 100 --random-state 42 --strategy mean

    # Exclude specific columns and set output directory
    python impute_by_mice.py input.csv --exclude-columns col1 col2 --output-dir /path/to/output

    # With verbose output and custom logging
    python impute_by_mice.py input.csv --verbose --log-level DEBUG
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from src.data.imputation.mice_imputer import MICEService, MICEConfig
from src.utils.logging import get_logger

# Configure logging
logger = get_logger("mice_imputation")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Namespace containing the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Perform MICE imputation on CSV files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Input CSV file containing data with missing values",
    )
    parser.add_argument(
        "--exclude-columns",
        type=str,
        nargs="+",
        help="Columns to exclude from imputation",
        default=[],
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=137,
        help="Maximum number of imputation rounds",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=137,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--strategy",
        choices=["mean", "median", "most_frequent", "constant"],
        default="mean",
        help="Initial imputation strategy",
    )
    parser.add_argument(
        "--n-nearest-features",
        type=int,
        help="Number of nearest features to use for imputation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output during imputation",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Set the logging level",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for imputed data and statistics (optional)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip imputation if output files already exist",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the imputed data",
    )

    parser.add_argument(
        "--save-plot",
        type=str,
        help="Path to save the plot. If not specified, a plot will be saved in the output directory when --plot is used.",
    )

    return parser.parse_args()


def setup_output_directory(base_dir: Path, input_file: Path) -> Path:
    """Set up the output directory for imputation results.

    Args:
        base_dir: Base directory for output
        input_file: Input file being processed

    Returns:
        Path to the output directory
    """
    output_dir = base_dir / f"{input_file.stem}_imputed"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_imputation_results(
    output_dir: Path,
    original_data: pd.DataFrame,
    imputed_data: pd.DataFrame,
    stats: dict,
) -> None:
    """Save imputation results and statistics.

    Args:
        output_dir: Directory to save results
        original_data: Original DataFrame before imputation
        imputed_data: DataFrame after imputation
        stats: Dictionary containing imputation statistics
    """
    # Save imputed data
    imputed_data.to_csv(output_dir / "imputed_data.csv", index=False)

    # Save statistics
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(output_dir / "imputation_statistics.csv", index=False)

    # Save comparison report
    with open(output_dir / "imputation_report.txt", "w") as f:
        f.write("Imputation Report\n")
        f.write("=" * 50 + "\n\n")

        f.write("Dataset Information:\n")
        f.write(f"Total rows: {len(original_data)}\n")
        f.write(f"Total columns: {len(original_data.columns)}\n\n")

        f.write("Missing Values Summary:\n")
        f.write(f"Total missing values: {stats['total_missing_values']}\n\n")

        f.write("Missing Values by Column:\n")
        for col, count in stats["missing_by_column"].items():
            if count > 0:
                percentage = stats["missing_percentage"][col]
                f.write(f"{col}: {count} ({percentage:.2f}%)\n")


def plot_imputed_data(
    original_data: pd.DataFrame,
    imputed_data: pd.DataFrame,
    missing_mask: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot the imputed data with red outlines around groups of imputed values.

    Args:
        original_data: Original DataFrame with missing values
        imputed_data: DataFrame after imputation
        missing_mask: Boolean mask of missing values (True where data was missing)
        save_path: Path to save the plot. If None, the plot will be displayed.
    """
    # Get numeric columns
    numeric_cols = imputed_data.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        logger.warning("No numeric columns found for plotting")
        return

    # Create DataFrames with just numeric columns
    numeric_df = imputed_data[numeric_cols]

    # Scale data for better visualization (min-max scaling)
    scaled_df = (numeric_df - numeric_df.min(axis=0)) / (
        numeric_df.max(axis=0) - numeric_df.min(axis=0)
    )

    # Get mask for missing values in numeric columns only
    # Make sure we're using only the columns that exist in both the mask and numeric columns
    common_cols = [col for col in numeric_cols if col in missing_mask.columns]
    numeric_missing_mask = missing_mask[common_cols].copy()

    # Ensure the mask has the same shape as the data we're plotting
    # This prevents any misalignment issues
    if len(common_cols) < len(numeric_cols):
        for col in set(numeric_cols) - set(common_cols):
            numeric_missing_mask[col] = False

    numeric_missing_mask = numeric_missing_mask[numeric_cols]

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot the heatmap of the data
    cax = plt.imshow(scaled_df.to_numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(cax, label="Normalized Value")

    # Create a 2D array of the missing mask to identify connected regions
    mask_array = numeric_missing_mask.values

    # Find connected regions of imputed cells (horizontal, vertical and diagonal connections)
    visited = np.zeros_like(mask_array, dtype=bool)

    # Helper function to find connected imputed cells
    def find_connected_region(i, j, region=None):
        if region is None:
            region = []

        # Check bounds and if cell is imputed and not visited
        if (
            i < 0
            or i >= mask_array.shape[0]
            or j < 0
            or j >= mask_array.shape[1]
            or not mask_array[i, j]
            or visited[i, j]
        ):
            return region

        # Mark as visited and add to region
        visited[i, j] = True
        region.append((i, j))

        # Check all 8 adjacent cells (horizontal, vertical, diagonal)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                find_connected_region(i + di, j + dj, region)

        return region

    # Find all connected regions
    connected_regions = []
    for i in range(mask_array.shape[0]):
        for j in range(mask_array.shape[1]):
            if mask_array[i, j] and not visited[i, j]:
                region = find_connected_region(i, j)
                if region:
                    connected_regions.append(region)

    # Draw outlines around connected regions
    for region in connected_regions:
        # Find bounding rectangle for the region
        min_i = min(i for i, j in region)
        max_i = max(i for i, j in region)
        min_j = min(j for i, j in region)
        max_j = max(j for i, j in region)

        # Draw rectangle around the region
        rect = Rectangle(
            (min_j - 0.5, min_i - 0.5),
            max_j - min_j + 1,
            max_i - min_i + 1,
            fill=False,
            edgecolor="red",
            linewidth=1.5,
            linestyle="-",
        )
        plt.gca().add_patch(rect)

    # Add axis labels and title
    plt.xlabel("Features")
    plt.ylabel("Samples")
    plt.title("MICE Imputed Data Visualization (Red outline = imputed values)")

    # Add feature names on x-axis
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)

    # Add sample indices on y-axis if not too many
    if len(imputed_data) <= 30:
        plt.yticks(range(len(imputed_data)))

    plt.tight_layout()

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()


def run_imputation(args: argparse.Namespace) -> int:
    """Run the MICE imputation process.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Set logging level
        logger.setLevel(getattr(logging, args.log_level))

        input_file = Path(args.input_file)
        if not input_file.exists():
            logger.error(f"Input file does not exist: {input_file}")
            return 1

        # Set up output directory
        output_base = Path(args.output_dir) if args.output_dir else input_file.parent
        output_dir = setup_output_directory(output_base, input_file)

        # Check if output already exists
        if args.skip_existing and (output_dir / "imputed_data.csv").exists():
            logger.info(f"Output already exists in {output_dir}, skipping...")
            return 0

        # Load data
        logger.info(f"Loading data from {input_file}")
        data = pd.read_csv(input_file)

        # Configure and initialize MICE
        config = MICEConfig(
            max_iterations=args.max_iterations,
            random_state=args.random_state,
            initial_strategy=args.strategy,
            n_nearest_features=args.n_nearest_features,
            verbose=args.verbose,
            exclude_columns=args.exclude_columns,
        )
        service = MICEService(config)

        # Print initial statistics
        logger.info("\nInitial Data Statistics:")
        logger.info("-" * 20)
        logger.info(f"Total rows: {len(data)}")
        logger.info(f"Total columns: {len(data.columns)}")
        logger.info(f"Total missing values: {data.isna().sum().sum()}")
        if args.exclude_columns:
            logger.info(f"Excluded columns: {', '.join(args.exclude_columns)}")

        # Create a mask of missing values before imputation
        original_missing_mask = data.isna()

        # Perform imputation
        logger.info("\nPerforming MICE imputation...")
        imputed_df = service.impute(data)
        stats = service.get_imputation_statistics()

        # Save results
        save_imputation_results(output_dir, data, imputed_df, stats)
        logger.info(f"\nResults saved to: {output_dir}")

        # Plot the imputed data if requested
        if args.plot or args.save_plot:
            # Determine the plot save path
            plot_path = None
            if args.save_plot:
                plot_path = args.save_plot
            elif args.plot:
                # Save in the output directory by default when plotting is enabled
                plot_path = output_dir / "imputed_data_visualization.png"

            # Plot the data
            plot_imputed_data(data, imputed_df, original_missing_mask, plot_path)

            if plot_path:
                logger.info(f"Plot saved to: {plot_path}")

        # Print summary
        logger.info("\nImputation Summary:")
        logger.info("-" * 20)
        logger.info(f"Total missing values imputed: {stats['total_missing_values']}")
        logger.info(
            f"Number of columns with missing values: {len(stats['missing_by_column'])}"
        )

        return 0

    except KeyboardInterrupt:
        logger.info("\nImputation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error during imputation: {str(e)}", exc_info=True)
        return 1


def main() -> None:
    """Main entry point for the script."""
    args = parse_args()
    sys.exit(run_imputation(args))


if __name__ == "__main__":
    main()
