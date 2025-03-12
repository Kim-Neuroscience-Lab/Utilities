# src/scripts/roi_area_analysis.py
"""
Script for performing roi area analysis of cortical imaging data.

This script provides functionality for computing areas of ROIs from pickle files
across multiple directories. It supports GPU acceleration and parallel processing
for improved performance.

Example usage:
    # Basic usage
    python roi_area_analysis.py /path/to/data

    # With GPU acceleration and custom settings
    python roi_area_analysis.py /path/to/data --use-gpu --workers 8 --log-level INFO --output-dir /path/to/output

    # With custom batch size and skip existing results
    python roi_area_analysis.py /path/to/data --batch-size 64 --skip-existing
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.analysis.roi_area_analyzer import ROIAreaAnalyzer
from src.utils.logging import get_logger

# Configure logging
logger = get_logger("roi_area_analysis")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Namespace containing the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Analyze ROI areas from pickle files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dir", type=str, help="Directory containing ROI pickle files"
    )
    parser.add_argument(
        "--use-gpu", action="store_true", help="Use GPU acceleration if available"
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="Number of worker threads"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory for analysis results (optional)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing ROIs (default: 32)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip analysis if output directory already exists",
    )

    return parser.parse_args()


def setup_output_directory(base_dir: Path, animal_id: str) -> Path:
    """Set up the output directory for analysis results.

    Args:
        base_dir: Base directory for output
        animal_id: ID of the animal being analyzed

    Returns:
        Path to the output directory
    """
    output_dir = base_dir / f"{animal_id.lower()}_area_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_analysis(args: argparse.Namespace) -> int:
    """Run the ROI area analysis.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Set logging level
        logger.setLevel(getattr(logging, args.log_level))

        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return 1

        # Initialize analyzer
        analyzer = ROIAreaAnalyzer(
            input_dir=str(input_dir), max_workers=args.workers, use_gpu=args.use_gpu
        )

        # Print system info
        logger.info("\nSystem Information:")
        logger.info("-" * 20)
        sys_info = analyzer.get_system_info()
        for key, value in sys_info.items():
            if key != "computation_methods":
                logger.info(f"{key}: {value}")

        # Set up output directory if specified
        output_base = Path(args.output_dir) if args.output_dir else input_dir.parent

        # Run analysis on all directories
        logger.info("\nAnalyzing ROIs...")
        results: Dict[str, pd.DataFrame] = analyzer.analyze_all_directories()

        if not results:
            logger.warning("No ROIs found to analyze!")
            return 1

        # Print computation statistics
        logger.info("\nComputation Method Usage:")
        logger.info("-" * 20)
        for method, count in analyzer.method_counts.items():
            logger.info(f"{method.capitalize()} method: {count} ROIs")

        # Process and save results for each animal
        logger.info("\nAnalysis Results:")
        logger.info("-" * 20)

        for animal_id, df in results.items():
            logger.info(f"\nAnimal: {animal_id}")
            logger.info(f"Total ROIs analyzed: {len(df)}")
            logger.info(f"Number of unique regions: {df['region_name'].nunique()}")
            logger.info(f"Number of segments: {df['segment_id'].nunique()}")

            # Set up output directory for this animal
            output_dir = setup_output_directory(output_base, animal_id)

            # Generate and save summaries
            region_summary = analyzer.get_summary_by_region(df)
            segment_summary = analyzer.get_summary_by_segment(df)

            # Save results
            df.to_csv(output_dir / "detailed_results.csv", index=False)
            region_summary.to_csv(output_dir / "region_summary.csv", index=False)
            segment_summary.to_csv(output_dir / "segment_summary.csv", index=False)

            logger.info(f"Results saved to: {output_dir}")

        return 0

    except KeyboardInterrupt:
        logger.info("\nAnalysis interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return 1


def main() -> None:
    """Main entry point for the script."""
    args = parse_args()
    sys.exit(run_analysis(args))


if __name__ == "__main__":
    main()
