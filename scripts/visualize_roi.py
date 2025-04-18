# scripts/visualize_roi.py
"""
Script for visualizing ROI data in original and transformed formats.

This script loads an ROI from a pickle file and visualizes it in both
its original format and the transformed 101x101 grid format.

Example usage:
    python visualize_roi.py /path/to/roi.pkl --output output/visualization.png
"""

import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.utils.visualization import (
    plot_roi_original_and_transformed,
    visualize_roi_with_axons,
)
from src.utils.logging import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)


# Define a simple placeholder axon detection function
# This would normally be replaced with your actual axon detection algorithm
def simple_axon_detection(image):
    """A simple placeholder for axon detection that just thresholds the image."""
    threshold = np.percentile(image[image > 0], 75)  # Use 75th percentile as threshold
    return (image > threshold).astype(np.uint8)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize ROI data in original and transformed formats",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("roi_file", type=str, help="Path to the ROI pickle file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the visualization image (optional)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["basic", "axons"],
        default="basic",
        help="Visualization mode: 'basic' for original/transformed, 'axons' to include axon detection",
    )
    parser.add_argument(
        "--cmap", type=str, default="viridis", help="Colormap to use for visualization"
    )
    return parser.parse_args()


def main():
    """Run the ROI visualization."""
    args = parse_args()

    # Ensure the input file exists
    roi_path = Path(args.roi_file)
    if not roi_path.exists():
        logger.error(f"ROI file not found: {roi_path}")
        return 1

    # Set up output path if provided
    output_path = args.output
    if output_path:
        # Create output directory if needed
        output_dir = Path(output_path).parent
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Log details
        logger.info(f"Visualizing ROI: {roi_path}")
        logger.info(f"Mode: {args.mode}")
        if output_path:
            logger.info(f"Output will be saved to: {output_path}")

        # Perform visualization based on mode
        if args.mode == "basic":
            fig = plot_roi_original_and_transformed(
                roi_path=str(roi_path), output_path=output_path, cmap=args.cmap
            )
        else:  # args.mode == "axons"
            fig = visualize_roi_with_axons(
                roi_path=str(roi_path),
                axon_mask_func=simple_axon_detection,
                output_path=output_path,
                cmap=args.cmap,
            )

        # Show the plot if no output file was specified
        if not output_path and fig is not None:
            plt.show()

        logger.info("Visualization completed successfully")

    except Exception as e:
        logger.error(f"Error visualizing ROI: {str(e)}")
        logger.exception(e)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
