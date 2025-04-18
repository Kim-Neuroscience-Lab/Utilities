# scripts/view_roi_simple.py
"""
Standalone script for visualizing ROI data in original and transformed formats.

This script loads an ROI from a pickle file and visualizes it in both
its original format and the transformed 101x101 grid format.

Example usage:
    python view_roi_simple.py /path/to/roi.pkl --output output/visualization.png
"""

import argparse
import os
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from datetime import datetime
from scipy.ndimage import binary_dilation, binary_fill_holes
from skimage.measure import find_contours

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("roi_visualizer")


def plot_roi_original_and_transformed(
    roi_path,
    output_path=None,
    fig_size=(12, 6),
    cmap="viridis",
    transform_size=(101, 101),
):
    """Plot original and transformed ROI side by side."""
    try:
        # Load the ROI pickle
        with open(roi_path, "rb") as f:
            package = pickle.load(f)
            intensity = package.get("roi", {})
            name = package.get("name", "Unknown")

        if not intensity:
            logger.error(f"No intensity data in {roi_path}")
            return None

        # Create the figure
        fig, axes = plt.subplots(1, 2, figsize=fig_size)

        # Get coordinates
        coords = list(intensity.keys())
        min_y = min(vert[0] for vert in coords)
        max_y = max(vert[0] for vert in coords)
        min_x = min(vert[1] for vert in coords)
        max_x = max(vert[1] for vert in coords)

        # Create original image
        original_shape = (max_y - min_y + 1, max_x - min_x + 1)
        original_img = np.zeros(original_shape, dtype=np.float32)

        # Fill in the intensity values in original space
        for (y, x), value in intensity.items():
            y_idx, x_idx = y - min_y, x - min_x
            if 0 <= y_idx < original_shape[0] and 0 <= x_idx < original_shape[1]:
                original_img[y_idx, x_idx] = value

        # Create binary mask for visualization
        binary_mask = (original_img > 0).astype(np.uint8)

        # Calculate ranges for transformation
        y_range = max_y - min_y
        x_range = max_x - min_x

        # Create target square grid dimensions
        height, width = transform_size
        transformed_img = np.zeros((height, width), dtype=np.float32)

        logger.info("Applying RSAT-style transformation...")

        # Direct point mapping as in original RSAT code
        # This maps each original point to a position in the square grid
        # based on its relative position in the original ROI
        for (y, x), value in intensity.items():
            # Skip zero values
            if value <= 0:
                continue

            # Calculate relative position in original space
            rel_y = y - min_y
            rel_x = x - min_x

            # Map to grid position using RSAT formula
            # The original RSAT code uses 0-100 range
            norm_y = int(rel_y / y_range * 100)
            norm_x = int(rel_x / x_range * 100)

            # Ensure within bounds (RSAT uses 0-100 inclusive)
            if 0 <= norm_y <= 100 and 0 <= norm_x <= 100:
                # Original RSAT sets binary values, but we'll use intensity values
                transformed_img[norm_y, norm_x] = value

        # Get common colormap normalization
        mask_orig = original_img > 0
        mask_trans = transformed_img > 0

        if np.any(mask_orig) and np.any(mask_trans):
            orig_min = float(np.min(original_img[mask_orig]))
            trans_min = float(np.min(transformed_img[mask_trans]))
            vmin = min(orig_min, trans_min)

            orig_max = float(np.max(original_img))
            trans_max = float(np.max(transformed_img))
            vmax = max(orig_max, trans_max)
        else:
            vmin, vmax = 0, 1

        # Plot original image
        im0 = axes[0].imshow(original_img, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title(f"Original ROI: {name}")
        axes[0].set_xlabel(f"X ({original_shape[1]} px)")
        axes[0].set_ylabel(f"Y ({original_shape[0]} px)")

        # Plot transformed image
        im1 = axes[1].imshow(transformed_img, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Transformed ROI: {name} (RSAT method)")
        axes[1].set_xlabel(f"X (101 px)")
        axes[1].set_ylabel(f"Y (101 px)")

        # Add colorbars
        cbar = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        cbar.set_label("Intensity")
        cbar = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("Intensity")

        # Add overall title
        plt.suptitle(f"ROI Visualization: {Path(roi_path).name}", fontsize=16)
        plt.tight_layout(rect=(0, 0, 1, 0.96))  # Adjust for suptitle

        # Save figure if output path is provided
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Figure saved to {output_path}")

        return fig

    except Exception as e:
        logger.error(f"Error visualizing ROI: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize ROI in original and transformed formats."
    )
    parser.add_argument("roi_path", help="Path to ROI pickle file.")
    parser.add_argument(
        "--output", "-o", help="Path to save the visualization (optional)."
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Colormap to use for visualization (default: viridis).",
    )
    return parser.parse_args()


def main():
    """Main function to run the visualization."""
    args = parse_args()

    # Check if file exists
    if not os.path.exists(args.roi_path):
        logger.error(f"ROI file not found: {args.roi_path}")
        return 1

    # Log start
    logger.info(f"Visualizing ROI: {args.roi_path}")
    if args.output:
        logger.info(f"Output will be saved to: {args.output}")

    # Visualize ROI
    try:
        fig = plot_roi_original_and_transformed(
            args.roi_path,
            output_path=args.output,
            cmap=args.cmap,
        )

        # Show the plot if output is not specified
        if fig and not args.output:
            plt.show()

        logger.info("Visualization completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
