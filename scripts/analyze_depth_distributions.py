# scripts/analyze_depth_distributions.py
"""
Script to analyze depth distributions from pickle files.

This script processes pickle files containing ROI data and generates depth distribution
analysis for each region and animal.

Example usage:
    python analyze_depth_distributions.py /path/to/pickle/file --output /path/to/output
"""

import argparse
from pathlib import Path
import sys
from typing import Optional

from src.analysis.depth_analyzer import DepthAnalyzer
from src.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Analyze depth distributions from pickle files."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input pickle file containing ROI data",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save the analysis results",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=101,
        help="Number of bins for depth analysis (default: 101)",
    )

    return parser.parse_args()


def main() -> Optional[int]:
    """Main function to run the depth distribution analysis.

    Returns:
        Optional[int]: Exit code (0 for success, 1 for error)
    """
    # Parse command line arguments
    args = parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize analyzer
    analyzer = DepthAnalyzer(num_bins=args.num_bins)

    try:
        # Load and process the pickle file
        logger.info(f"Processing {args.input_file}")
        analyzer.load_pickle_file(args.input_file)

        # Process each age group and animal
        for age_group in analyzer._region_data:
            age_group_dir = output_dir / age_group
            age_group_dir.mkdir(exist_ok=True)

            for animal_id in analyzer._region_data[age_group]:
                try:
                    # Save distributions for this animal
                    analyzer.save_distributions(
                        age_group=age_group,
                        animal_id=animal_id,
                        output_dir=age_group_dir,
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing animal {animal_id} in age group {age_group}: {str(e)}"
                    )

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Error processing {args.input_file}: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
