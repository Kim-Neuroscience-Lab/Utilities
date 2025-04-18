"""Utilities for visualizing ROI data."""

# Standard Library Imports
from typing import Optional, Tuple, Dict, Any, Union
import os
from pathlib import Path

# Third Party Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Internal Imports
from src.core.models.spatial.roi import ROI
from src.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)


def plot_roi_original_and_transformed(
    roi_path: str,
    output_path: Optional[str] = None,
    fig_size: Tuple[int, int] = (12, 6),
    cmap: str = "viridis",
    transform_size: Tuple[int, int] = (101, 101),
) -> Optional[Figure]:
    """Plot the original ROI data and its transformed representation side by side.

    Args:
        roi_path: Path to the ROI pickle file
        output_path: Path to save the figure (optional)
        fig_size: Figure size (width, height) in inches
        cmap: Colormap to use for intensity visualization
        transform_size: Size of the transformed image (default: 101x101)

    Returns:
        Optional[Figure]: The created figure, or None if loading failed
    """
    # Load the ROI
    roi = ROI.from_file(roi_path)
    if roi is None:
        logger.error(f"Failed to load ROI from {roi_path}")
        return None

    if roi.intensity is None or not roi.intensity:
        logger.error(f"ROI has no intensity data: {roi_path}")
        return None

    # Get ROI name and filename for title
    roi_name = roi.roi or "Unknown"
    file_name = Path(roi_path).name

    # Create the figure
    fig, axes = plt.subplots(1, 2, figsize=fig_size)

    # Get original ROI bounds
    min_y = min(vert[0] for vert in roi.verts)
    max_y = max(vert[0] for vert in roi.verts)
    min_x = min(vert[1] for vert in roi.verts)
    max_x = max(vert[1] for vert in roi.verts)

    # Create original image
    original_shape = (max_y - min_y + 1, max_x - min_x + 1)
    original_img = np.zeros(original_shape, dtype=np.float32)

    # Fill in the intensity values
    for (y, x), intensity in roi.intensity.items():
        y_idx, x_idx = y - min_y, x - min_x
        if 0 <= y_idx < original_shape[0] and 0 <= x_idx < original_shape[1]:
            original_img[y_idx, x_idx] = intensity

    # Create transformed image
    height, width = transform_size
    transformed_img = np.zeros((height, width), dtype=np.float32)

    # Transform coordinates to square grid
    y_range = max_y - min_y
    x_range = max_x - min_x

    for (y, x), intensity in roi.intensity.items():
        # Normalize coordinates to square grid
        norm_y = int((y - min_y) / y_range * (height - 1))
        norm_x = int((x - min_x) / x_range * (width - 1))

        # Ensure within bounds
        if 0 <= norm_y < height and 0 <= norm_x < width:
            transformed_img[norm_y, norm_x] = intensity

    # Get common colormap normalization
    orig_min = float(np.min(original_img[original_img > 0]))
    trans_min = float(np.min(transformed_img[transformed_img > 0]))
    vmin = min(orig_min, trans_min)

    orig_max = float(np.max(original_img))
    trans_max = float(np.max(transformed_img))
    vmax = max(orig_max, trans_max)

    norm = Normalize(vmin=vmin, vmax=vmax)

    # Plot original image
    im0 = axes[0].imshow(original_img, cmap=cmap, norm=norm)
    axes[0].set_title(f"Original ROI: {roi_name}")
    axes[0].set_xlabel(f"X ({original_shape[1]} px)")
    axes[0].set_ylabel(f"Y ({original_shape[0]} px)")

    # Plot transformed image
    im1 = axes[1].imshow(transformed_img, cmap=cmap, norm=norm)
    axes[1].set_title(f"Transformed ROI: {roi_name}")
    axes[1].set_xlabel(f"X ({width} px)")
    axes[1].set_ylabel(f"Y ({height} px)")

    # Add colorbars
    cbar = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label("Intensity")
    cbar = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Intensity")

    # Add overall title
    plt.suptitle(f"ROI Visualization: {file_name}", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.96))  # Adjust for suptitle

    # Save figure if output path is provided
    if output_path:
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Figure saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving figure: {str(e)}")

    return fig


def visualize_roi_with_axons(
    roi_path: str,
    axon_mask_func: Any,
    output_path: Optional[str] = None,
    fig_size: Tuple[int, int] = (15, 5),
    cmap: str = "viridis",
    axon_color: str = "red",
) -> Optional[Figure]:
    """Visualize an ROI with detected axons.

    Args:
        roi_path: Path to the ROI pickle file
        axon_mask_func: Function to generate axon masks
        output_path: Path to save the figure (optional)
        fig_size: Figure size (width, height) in inches
        cmap: Colormap to use for intensity visualization
        axon_color: Color to use for axon overlay

    Returns:
        Optional[Figure]: The created figure, or None if loading failed
    """
    # Load the ROI
    roi = ROI.from_file(roi_path)
    if roi is None:
        logger.error(f"Failed to load ROI from {roi_path}")
        return None

    if roi.intensity is None or not roi.intensity:
        logger.error(f"ROI has no intensity data: {roi_path}")
        return None

    # Get ROI name and filename for title
    roi_name = roi.roi or "Unknown"
    file_name = Path(roi_path).name

    # Get original ROI bounds
    min_y = min(vert[0] for vert in roi.verts)
    max_y = max(vert[0] for vert in roi.verts)
    min_x = min(vert[1] for vert in roi.verts)
    max_x = max(vert[1] for vert in roi.verts)

    # Create original image
    original_shape = (max_y - min_y + 1, max_x - min_x + 1)
    original_img = np.zeros(original_shape, dtype=np.float32)
    mask = np.zeros(original_shape, dtype=np.uint8)

    # Fill in the intensity values and mask
    for (y, x), intensity in roi.intensity.items():
        y_idx, x_idx = y - min_y, x - min_x
        if 0 <= y_idx < original_shape[0] and 0 <= x_idx < original_shape[1]:
            original_img[y_idx, x_idx] = intensity
            mask[y_idx, x_idx] = 1

    # Get axon mask
    axon_mask = axon_mask_func(original_img)

    # Clean edges (pixels outside ROI)
    outside_points = np.argwhere(mask == 0)
    axon_mask = roi._correct_edges(outside_points, axon_mask)

    # Create transformed mask
    height, width = 101, 101
    transformed_img = np.zeros((height, width), dtype=np.float32)
    transformed_axons = np.zeros((height, width), dtype=np.uint8)

    # Transform coordinates to square grid
    y_range = max_y - min_y
    x_range = max_x - min_x

    for (y, x), intensity in roi.intensity.items():
        y_idx, x_idx = y - min_y, x - min_x

        # Normalize coordinates to square grid
        norm_y = int((y - min_y) / y_range * (height - 1))
        norm_x = int((x - min_x) / x_range * (width - 1))

        # Ensure within bounds
        if (
            0 <= y_idx < original_shape[0]
            and 0 <= x_idx < original_shape[1]
            and 0 <= norm_y < height
            and 0 <= norm_x < width
        ):
            transformed_img[norm_y, norm_x] = intensity
            if axon_mask[y_idx, x_idx] > 0:
                transformed_axons[norm_y, norm_x] = 1

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=fig_size)

    # Plot original image
    im0 = axes[0].imshow(original_img, cmap=cmap)
    axes[0].set_title(f"Original ROI")
    axes[0].set_xlabel(f"X ({original_shape[1]} px)")
    axes[0].set_ylabel(f"Y ({original_shape[0]} px)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot original with axon overlay
    im1 = axes[1].imshow(original_img, cmap=cmap)
    # Create masked array for axon overlay
    axon_overlay = np.ma.masked_where(axon_mask == 0, axon_mask)
    axes[1].imshow(
        axon_overlay, cmap=plt.cm.get_cmap("binary", 2), alpha=0.7, vmin=0, vmax=1
    )
    axes[1].set_title("Original with Axons")
    axes[1].set_xlabel(f"X ({original_shape[1]} px)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot transformed with axon overlay
    im2 = axes[2].imshow(transformed_img, cmap=cmap)
    # Create masked array for transformed axon overlay
    t_axon_overlay = np.ma.masked_where(transformed_axons == 0, transformed_axons)
    axes[2].imshow(
        t_axon_overlay, cmap=plt.cm.get_cmap("binary", 2), alpha=0.7, vmin=0, vmax=1
    )
    axes[2].set_title("Transformed with Axons")
    axes[2].set_xlabel(f"X (101 px)")
    axes[2].set_ylabel(f"Y (101 px)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Add overall title
    plt.suptitle(f"{roi_name} ({file_name})", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.96))  # Adjust for suptitle

    # Save figure if output path is provided
    if output_path:
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Figure saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving figure: {str(e)}")

    return fig
