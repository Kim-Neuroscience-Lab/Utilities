# scripts/view_roi_with_axons.py
"""
Standalone script for visualizing ROI data with axon detection.

This script loads an ROI from a pickle file, detects axons by
normalizing, thresholding, converting to binary, detecting, and then transforming.

Example usage:
    python view_roi_with_axons.py /path/to/roi.pkl --output output/axon_visualization.png
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
import cv2
from scipy.ndimage import binary_dilation, binary_closing, gaussian_filter
from skimage.morphology import remove_small_objects, disk
from matplotlib.colors import Normalize
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("roi_visualizer")


def normalize_image(image, mask=None):
    """Normalize image to 0-1 range.

    Args:
        image: Input image
        mask: Optional mask of valid pixels

    Returns:
        Normalized image
    """
    # Make a copy to avoid modifying the original
    normalized = image.copy()

    # Determine valid pixels
    if mask is not None:
        valid_pixels = (mask > 0) & (image > 0)
    else:
        valid_pixels = image > 0

    # Skip if no valid pixels
    if not np.any(valid_pixels):
        return normalized

    # Simple min-max normalization
    min_val = np.min(normalized[valid_pixels])
    max_val = np.max(normalized[valid_pixels])

    if max_val > min_val:
        # Apply normalization only to valid pixels
        normalized[valid_pixels] = (normalized[valid_pixels] - min_val) / (
            max_val - min_val
        )

    logger.info(f"Normalized image: min={min_val}, max={max_val}")
    return normalized


def separate_clumped_axons(binary_mask, min_distance=5, min_intensity=0.1):
    """Separate clumped axons using distance transform and watershed.

    Args:
        binary_mask: Binary mask where 1 represents axons
        min_distance: Minimum distance between peaks for watershed markers
        min_intensity: Minimum intensity threshold for peak detection

    Returns:
        Segmented mask with separated axons
    """
    # If no pixels in mask, return empty mask
    if np.sum(binary_mask) == 0:
        return binary_mask

    # Convert to correct dtype
    mask = binary_mask.astype(np.uint8)

    # Compute the distance transform
    distance = ndi.distance_transform_edt(mask)

    # Find local maxima in the distance transform - returns coordinates
    coordinates = peak_local_max(
        distance, min_distance=min_distance, threshold_abs=min_intensity, labels=mask
    )

    # Create markers for watershed
    markers = np.zeros_like(mask, dtype=np.int32)
    marker_count = 0

    # Check if we have any coordinates
    if coordinates.size > 0:
        for i, coord in enumerate(coordinates, start=1):
            markers[coord[0], coord[1]] = i
            marker_count = i  # Keep track of the number of markers

    logger.info(f"Created {marker_count} watershed markers for axon separation")

    # Apply watershed using the distance transform
    if marker_count > 0:
        segmented = watershed(-distance, markers, mask=mask)
        # Convert back to binary mask
        result = (segmented > 0).astype(np.uint8)
    else:
        # If no markers found, return the original mask
        result = mask

    # Count objects after segmentation
    labels, num_objects = ndi.label(result)
    logger.info(f"Separated {num_objects} axons from {marker_count} markers")

    return result


def detect_axons(
    image, threshold=0.5, min_size=1, apply_morphology=True, separate_axons=False
):
    """Detect axons using thresholding on normalized image.

    Args:
        image: Normalized ROI image (0-1 range)
        threshold: Threshold value (0.0-1.0)
        min_size: Minimum size of objects to keep
        apply_morphology: Whether to apply morphological operations
        separate_axons: Whether to separate clumped axons using watershed

    Returns:
        Binary mask where 1 represents detected axons
    """
    # Ensure we have a valid image
    if not np.any(image > 0):
        return np.zeros_like(image, dtype=np.uint8)

    # Apply thresholding to create binary mask
    binary = (image >= threshold).astype(np.uint8)
    logger.info(f"Pixels above threshold {threshold}: {np.sum(binary)}")

    # Return the raw binary mask if no pixels are above threshold
    if np.sum(binary) == 0:
        logger.warning("No pixels above threshold, returning empty mask")
        return binary

    # Apply morphological operations if requested
    if apply_morphology and np.sum(binary) > 0:
        # Save before morphology for comparison
        before_morphology = np.sum(binary)

        # Clean up the binary mask to remove noise and connect nearby regions
        # Apply dilation to connect nearby pixels
        binary = binary_dilation(binary, structure=disk(1))

        # Apply closing to fill small holes
        binary = binary_closing(binary, structure=disk(2))

        # Remove small objects
        if min_size > 1 and np.sum(binary) > min_size * 10:
            binary = remove_small_objects(binary.astype(bool), min_size=min_size)

        logger.info(
            f"Pixels after morphology: {np.sum(binary)} (before: {before_morphology})"
        )

        # If morphology removed all pixels, revert to the original binary mask
        if np.sum(binary) == 0:
            logger.warning(
                "Morphology removed all pixels, reverting to original binary mask"
            )
            binary = (image >= threshold).astype(np.uint8)

    # Apply watershed-based separation if requested
    if separate_axons and np.sum(binary) > 0:
        before_separation = np.sum(binary)
        # Get the object count before separation
        labels, num_before = ndi.label(binary)
        logger.info(f"Before separation: {num_before} axon objects detected")

        # Apply watershed-based separation of clumped axons
        binary = separate_clumped_axons(binary, min_distance=5, min_intensity=0.1)

        # Get the object count after separation
        labels, num_after = ndi.label(binary)
        logger.info(f"After separation: {num_after} axon objects detected")

        # If separation removed all pixels, revert to the original binary mask
        if np.sum(binary) == 0:
            logger.warning(
                "Separation removed all pixels, reverting to original binary mask"
            )
            binary = (image >= threshold).astype(np.uint8)

    return binary.astype(np.uint8)


def correct_edges(outside_points, binary, iterations=1):
    """Remove any detected axons outside the ROI or near the edges.

    Args:
        outside_points: Array of [y, x] coordinates outside the ROI
        binary: Binary axon mask
        iterations: Number of iterations for dilation

    Returns:
        Corrected binary mask
    """
    # Create a mask for outside points
    if len(outside_points) == 0 or np.sum(binary) == 0:
        return binary

    mask = np.zeros_like(binary, dtype=np.uint8)

    # Mark outside points
    for y, x in outside_points:
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            mask[y, x] = 1

    # Dilate the mask to include points near the edge
    dilated_mask = binary_dilation(
        mask, structure=np.ones((3, 3)), iterations=iterations
    )

    # Remove edge points from the binary image
    corrected = binary.copy()
    corrected[dilated_mask == 1] = 0

    # If correction removed all pixels, revert with a warning
    if np.sum(corrected) == 0 and np.sum(binary) > 0:
        logger.warning("Edge correction removed all pixels, using original binary mask")
        return binary

    logger.info(f"Pixels after edge correction: {np.sum(corrected)}")
    return corrected


def get_axon_contours(binary_mask):
    """Extract contours from binary mask for axon outlining.

    Args:
        binary_mask: Binary mask where 1 represents axons

    Returns:
        List of contours in the format expected by matplotlib
    """
    # Make a copy to ensure we have the correct dtype
    mask_copy = binary_mask.copy().astype(np.uint8)

    # If no positive pixels, return empty list
    if np.sum(mask_copy) == 0:
        logger.warning("No pixels in binary mask, no contours to extract")
        return []

    # Use OpenCV to find contours in the binary mask
    contours, _ = cv2.findContours(
        mask_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    logger.info(f"Found {len(contours)} contours in the axon mask")

    # Convert OpenCV contours to matplotlib format (list of point arrays)
    matplotlib_contours = []
    for contour in contours:
        # Reshape contour points and swap x,y coordinates for matplotlib
        contour_points = contour.reshape(-1, 2)[
            :, ::-1
        ]  # Swap columns to get (x,y) format
        matplotlib_contours.append(contour_points)

    return matplotlib_contours


def get_axon_centers(binary_mask):
    """Extract centers of axons from binary mask.

    Args:
        binary_mask: Binary mask where 1 represents axons

    Returns:
        List of (x, y) center coordinates
    """
    # Make a copy to ensure we have the correct dtype
    mask_copy = binary_mask.copy().astype(np.uint8)

    # If no positive pixels, return empty list
    if np.sum(mask_copy) == 0:
        logger.warning("No pixels in binary mask, no centers to extract")
        return []

    # Use OpenCV to find contours in the binary mask
    contours, _ = cv2.findContours(
        mask_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    logger.info(f"Found {len(contours)} contours for center extraction")

    # Calculate centroids of contours
    centers = []
    for contour in contours:
        # Calculate moments of contour
        M = cv2.moments(contour)

        # Calculate centroid if area is not zero
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))  # x, y format for matplotlib

    logger.info(f"Extracted {len(centers)} axon centers")
    return centers


def transform_to_square(data, coords, transform_size=(101, 101)):
    """Transform ROI data to a square grid.

    Args:
        data: Dictionary mapping (y, x) coordinates to values
        coords: List of (y, x) coordinates
        transform_size: Size of the output square grid

    Returns:
        Transformed image
    """
    # Calculate bounds of original data
    min_y = min(vert[0] for vert in coords)
    max_y = max(vert[0] for vert in coords)
    min_x = min(vert[1] for vert in coords)
    max_x = max(vert[1] for vert in coords)

    # Calculate ranges for normalization
    y_range = max_y - min_y
    x_range = max_x - min_x

    # Create target square grid
    height, width = transform_size
    transformed = np.zeros((height, width), dtype=np.float32)

    # Map each original point to the square grid
    point_count = 0
    for (y, x), value in data.items():
        # Skip zero values
        if value <= 0:
            continue

        # Calculate relative position in original space
        rel_y = y - min_y
        rel_x = x - min_x

        # Map to grid position using relative position (RSAT-style)
        norm_y = int(rel_y / y_range * 100)
        norm_x = int(rel_x / x_range * 100)

        # Ensure within bounds
        if 0 <= norm_y <= 100 and 0 <= norm_x <= 100:
            transformed[norm_y, norm_x] = value
            point_count += 1

    logger.info(f"Mapped {point_count} points to the transformed grid")
    return transformed


def transform_points(centers, coords, transform_size=(101, 101)):
    """Transform axon center points from original space to square grid.

    Args:
        centers: List of (x, y) center coordinates in original space
        coords: List of (y, x) coordinates defining the ROI bounds
        transform_size: Size of the output square grid

    Returns:
        List of (x, y) coordinates in the transformed space
    """
    if not centers:
        return []

    # Calculate bounds of original data
    min_y = min(vert[0] for vert in coords)
    max_y = max(vert[0] for vert in coords)
    min_x = min(vert[1] for vert in coords)
    max_x = max(vert[1] for vert in coords)

    # Calculate ranges for normalization
    y_range = max_y - min_y
    x_range = max_x - min_x

    # Transform each center point
    transformed_centers = []
    for x, y in centers:
        # Convert from image coordinates (x, y) to data coordinates (y, x)
        data_y = y + min_y
        data_x = x + min_x

        # Calculate relative position
        rel_y = data_y - min_y
        rel_x = data_x - min_x

        # Map to grid position (RSAT-style)
        norm_y = int(rel_y / y_range * 100)
        norm_x = int(rel_x / x_range * 100)

        # Ensure within bounds and add to transformed centers (as x, y for plotting)
        if 0 <= norm_y <= 100 and 0 <= norm_x <= 100:
            transformed_centers.append((norm_x, norm_y))

    logger.info(f"Transformed {len(transformed_centers)} centers to the square grid")
    return transformed_centers


def sliding_window_detection(
    image,
    mask=None,
    window_size=25,
    overlap=0.5,
    threshold=0.5,
    min_size=1,
    apply_morphology=True,
    min_intensity_percentile=0.2,
    separate_axons=False,
):
    """Detect axons using sliding window normalization and thresholding.

    This approach allows for better detection of axons in areas with varying intensity
    by normalizing and thresholding within local windows.

    Args:
        image: Input image
        mask: Optional mask of valid pixels
        window_size: Size of sliding window (square)
        overlap: Fraction of overlap between adjacent windows
        threshold: Threshold value for normalized windows (0.0-1.0)
        min_size: Minimum size of objects to keep
        apply_morphology: Whether to apply morphological operations
        min_intensity_percentile: Skip windows with max intensity below this percentile
            of the global image, to avoid false positives in low-intensity regions
        separate_axons: Whether to separate clumped axons using watershed

    Returns:
        Binary mask where 1 represents detected axons
    """
    if not np.any(image > 0):
        return np.zeros_like(image, dtype=np.uint8)

    # Compute global image statistics for filtering low-intensity windows
    if mask is not None:
        valid_image = image[mask > 0]
    else:
        valid_image = image[image > 0]

    if len(valid_image) == 0:
        return np.zeros_like(image, dtype=np.uint8)

    # Calculate intensity threshold based on percentile of non-zero pixels
    global_min = np.min(valid_image)
    global_max = np.max(valid_image)
    global_range = global_max - global_min

    # Calculate min intensity threshold as a percentage of the global range
    min_intensity_threshold = global_min + (global_range * min_intensity_percentile)

    logger.info(f"Global intensity range: min={global_min}, max={global_max}")
    logger.info(
        f"Using min intensity threshold of {min_intensity_threshold:.2f} ({min_intensity_percentile*100:.0f}% of range)"
    )

    # Create output binary mask
    binary_output = np.zeros_like(image, dtype=np.uint8)

    # Calculate step size based on overlap
    step_size = int(window_size * (1 - overlap))
    if step_size < 1:
        step_size = 1

    # Get image dimensions
    h, w = image.shape

    # Track window count for logging
    window_count = 0
    windows_with_axons = 0
    windows_skipped = 0

    # Process each window
    for y in range(0, h - window_size + 1, step_size):
        for x in range(0, w - window_size + 1, step_size):
            # Extract window
            window = image[y : y + window_size, x : x + window_size]

            # Skip empty windows
            if not np.any(window > 0):
                continue

            # Apply mask if provided
            window_mask = None  # Initialize window_mask
            if mask is not None:
                window_mask = mask[y : y + window_size, x : x + window_size]
                if not np.any(window_mask > 0):
                    continue
                valid_pixels = (window_mask > 0) & (window > 0)
            else:
                valid_pixels = window > 0

            # Calculate window statistics
            if not np.any(valid_pixels):
                continue

            window_max = np.max(window[valid_pixels])

            # Skip windows with low maximum intensity to prevent false positives
            if window_max < min_intensity_threshold:
                windows_skipped += 1
                continue

            # Normalize window
            norm_window = normalize_image(window, mask=window_mask)

            # Threshold normalized window
            binary_window = (norm_window >= threshold).astype(np.uint8)

            # Skip if no pixels above threshold
            if np.sum(binary_window) == 0:
                continue

            # Count windows with detected axons
            windows_with_axons += 1

            # Update output mask (using logical OR to combine results)
            binary_output[y : y + window_size, x : x + window_size] = np.logical_or(
                binary_output[y : y + window_size, x : x + window_size], binary_window
            ).astype(np.uint8)

            window_count += 1

    logger.info(
        f"Processed {window_count} windows, found axons in {windows_with_axons} windows, skipped {windows_skipped} low-intensity windows"
    )
    logger.info(f"Pixels above threshold before morphology: {np.sum(binary_output)}")

    # Return if no pixels detected
    if np.sum(binary_output) == 0:
        return binary_output

    # Apply morphological operations if requested
    if apply_morphology:
        before_morphology = np.sum(binary_output)

        # Clean up the binary mask
        binary_output = binary_dilation(binary_output, structure=disk(1))
        binary_output = binary_closing(
            binary_output, structure=disk(1)
        )  # Smaller closing to preserve separation

        # Only remove small objects if enough pixels detected
        if min_size > 1 and np.sum(binary_output) > min_size * 10:
            binary_output = remove_small_objects(
                binary_output.astype(bool), min_size=min_size
            )

        logger.info(
            f"Pixels after morphology: {np.sum(binary_output)} (before: {before_morphology})"
        )

        # If morphology removed all pixels, revert to original
        if np.sum(binary_output) == 0:
            logger.warning(
                "Morphology removed all pixels, reverting to pre-morphology mask"
            )
            return binary_output

    # Apply watershed-based separation if requested
    if separate_axons and np.sum(binary_output) > 0:
        before_separation = np.sum(binary_output)
        # Get the object count before separation
        labels, num_before = ndi.label(binary_output)
        logger.info(f"Before separation: {num_before} axon objects detected")

        # Apply watershed-based separation of clumped axons
        binary_output = separate_clumped_axons(
            binary_output, min_distance=5, min_intensity=0.1
        )

        # Get the object count after separation
        labels, num_after = ndi.label(binary_output)
        logger.info(f"After separation: {num_after} axon objects detected")

        # If separation removed all pixels, revert to the original binary mask
        if np.sum(binary_output) == 0:
            logger.warning(
                "Separation removed all pixels, reverting to pre-separation mask"
            )

    return binary_output.astype(np.uint8)


def plot_roi_with_axons(
    roi_path,
    output_path=None,
    fig_size=(15, 15),  # Increased figure size for 3x2 layout
    cmap="viridis",
    transform_size=(101, 101),
    threshold=0.5,
    min_size=1,
    apply_morphology=True,
    use_sliding_window=False,
    window_size=25,
    window_overlap=0.5,
    min_intensity_percentile=0.2,
    separate_axons=False,
):
    """Plot original ROI and its transformed version with axon detection overlay."""
    try:
        # Load the ROI pickle
        with open(roi_path, "rb") as f:
            package = pickle.load(f)
            intensity = package.get("roi", {})
            name = package.get("name", "Unknown")

        if not intensity:
            logger.error(f"No intensity data in {roi_path}")
            return None

        # Create the figure with 3x2 subplots (3 rows, 2 columns)
        # Use equal width columns to match histogram panel with transformed ROI panel
        fig, axes = plt.subplots(
            3, 2, figsize=fig_size, gridspec_kw={"width_ratios": [1, 1]}
        )  # Equal width ratio
        axes = axes.flatten()  # Flatten to 1D array for easier indexing

        # Get coordinates
        coords = list(intensity.keys())
        min_y = min(vert[0] for vert in coords)
        max_y = max(vert[0] for vert in coords)
        min_x = min(vert[1] for vert in coords)
        max_x = max(vert[1] for vert in coords)

        # Create original image
        original_shape = (max_y - min_y + 1, max_x - min_x + 1)
        original_img = np.zeros(original_shape, dtype=np.float32)
        mask = np.zeros(original_shape, dtype=np.uint8)

        # Fill in the intensity values and ROI mask
        for (y, x), value in intensity.items():
            y_idx, x_idx = y - min_y, x - min_x
            if 0 <= y_idx < original_shape[0] and 0 <= x_idx < original_shape[1]:
                original_img[y_idx, x_idx] = value
                mask[y_idx, x_idx] = 1

        # Step 1: Normalize the original image
        logger.info("Normalizing image...")
        normalized_img = normalize_image(original_img, mask=mask)

        # Step 2: Apply thresholding and detect axons
        if use_sliding_window:
            logger.info(
                f"Detecting axons using sliding window approach (window size: {window_size}, overlap: {window_overlap:.2f}, threshold: {threshold:.2f}, min intensity: {min_intensity_percentile:.2f}, separate axons: {separate_axons})..."
            )
            axon_mask = sliding_window_detection(
                original_img,
                mask=mask,
                window_size=window_size,
                overlap=window_overlap,
                threshold=threshold,
                min_size=min_size,
                apply_morphology=apply_morphology,
                min_intensity_percentile=min_intensity_percentile,
                separate_axons=separate_axons,
            )
        else:
            logger.info(f"Detecting axons using global threshold at {threshold:.2f}...")
            axon_mask = detect_axons(
                normalized_img,
                threshold=threshold,
                min_size=min_size,
                apply_morphology=apply_morphology,
                separate_axons=separate_axons,
            )

        # Step 3: Clean up edges (remove "axons" outside the ROI)
        outside_points = np.argwhere(mask == 0)
        axon_mask = correct_edges(outside_points, axon_mask)

        # Get axon centers and contours in original space
        axon_centers = get_axon_centers(axon_mask)
        axon_contours = get_axon_contours(axon_mask)
        logger.info(f"Detected {len(axon_centers)} axon centers in original space")

        # Step 4: Transform to square grid
        logger.info("Transforming to square grid...")

        # Create transformed data dictionary for original intensities
        transformed_data = {}
        for (y, x), value in intensity.items():
            transformed_data[(y, x)] = value

        # Transform original intensity values
        transformed_img = transform_to_square(
            transformed_data, coords, transform_size=transform_size
        )

        # Transform axon centers directly from original space to square grid
        transformed_centers = transform_points(
            axon_centers, coords, transform_size=transform_size
        )

        # Panel 1: Original ROI image
        im0 = axes[0].imshow(original_img, cmap=cmap)
        axes[0].set_title(f"Original ROI: {name}")
        axes[0].set_xlabel(f"X ({original_shape[1]} px)")
        axes[0].set_ylabel(f"Y ({original_shape[0]} px)")
        cbar = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        cbar.set_label("Intensity")

        # Panel 2: Normalized Image
        im1 = axes[1].imshow(normalized_img, cmap=cmap, vmin=0, vmax=1)
        axes[1].set_title(f"Normalized Image")
        axes[1].set_xlabel(f"X ({original_shape[1]} px)")
        axes[1].set_ylabel(f"Y ({original_shape[0]} px)")
        cbar = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("Normalized Intensity (0-1)")

        # Panel 3: Binary mask without contours, maintaining aspect ratio
        im2 = axes[2].imshow(axon_mask, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        axes[2].set_title(f"Binary Mask")
        axes[2].set_xlabel(f"X ({original_shape[1]} px)")
        axes[2].set_ylabel(f"Y ({original_shape[0]} px)")
        # Force aspect ratio to match data dimensions
        axes[2].set_aspect(original_shape[1] / original_shape[0])
        cbar = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.set_label("Binary Value (0 or 1)")

        # Panel 4: Original image with borders (contours) drawn around detected axons
        im3 = axes[3].imshow(original_img, cmap=cmap)
        if axon_contours:
            # Draw red borders around axons
            for contour in axon_contours:
                # Plot the contour on the original image
                axes[3].plot(contour[:, 0], contour[:, 1], "r-", linewidth=0.8)
        axes[3].set_title(f"Original Image with Axon Borders")
        axes[3].set_xlabel(f"X ({original_shape[1]} px)")
        axes[3].set_ylabel(f"Y ({original_shape[0]} px)")
        cbar = fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        cbar.set_label("Intensity")

        # Panel 5: Histogram of axon counts with KDE curve
        if axon_centers:
            # Extract y coordinates of transformed centers for the histogram
            # This ensures alignment with panel 6 (transformed ROI)
            transformed_y_coords = np.array(
                [center[1] for center in transformed_centers]
            )

            # Create histogram with same number of bins as transformed ROI y dimension
            num_bins = transform_size[0]  # Use y dimension of transform_size (101)

            # Create the histogram - bars should extend left from right edge
            counts, bins, patches = axes[4].hist(
                transformed_y_coords,
                bins=num_bins,
                alpha=0.7,
                orientation="horizontal",
                color="skyblue",
                range=(
                    0,
                    transform_size[0] - 1,
                ),  # Force range to match transformed plot
            )

            # Add KDE curve if we have at least 3 points
            if len(transformed_y_coords) >= 3:
                try:
                    from scipy.stats import gaussian_kde

                    # Use same y range as the transformed plot
                    kde_y_range = np.linspace(0, transform_size[0] - 1, 200)

                    # Compute KDE if we have enough unique points
                    if len(np.unique(transformed_y_coords)) >= 3:
                        kde = gaussian_kde(transformed_y_coords)
                        kde_values = kde(kde_y_range)

                        # Scale KDE to fit on the same plot
                        if np.max(kde_values) > 0:
                            kde_scale = max(counts) / max(kde_values) * 0.8
                            axes[4].plot(
                                kde_values * kde_scale, kde_y_range, "r-", linewidth=2
                            )
                except Exception as e:
                    logger.warning(f"Could not generate KDE curve: {str(e)}")

            axes[4].set_title(f"Axon Distribution")
            axes[4].set_xlabel("Count")
            axes[4].set_ylabel("")  # Remove y-axis label since it's redundant

            # Set y-axis limits to match the transformed plot
            axes[4].set_ylim(
                transform_size[0] - 1, 0
            )  # Invert to match image coordinates

            # Set x-axis to start at max value and go to 0 (reverse direction)
            max_count = max(counts) if len(counts) > 0 else 1
            axes[4].set_xlim(max_count * 1.2, 0)  # Reverse x-axis direction

            # Remove y-axis ticks since they're redundant with panel 6
            axes[4].set_yticks([])

            # Set aspect ratio to be the same as panel 6 (transformed ROI)
            # This ensures the histogram has exactly the same dimensions
            # Don't use set_aspect as it can distort the alignment
        else:
            axes[4].set_title("No Axons Detected")
            axes[4].set_xlabel("Count")
            axes[4].set_yticks([])
            axes[4].set_ylim(transform_size[0] - 1, 0)  # Match transformed ROI y-axis
            axes[4].set_xlim(1, 0)  # Reverse x-axis

        # Panel 6: Transformed unmodified image with red points at axon centers
        im5 = axes[5].imshow(transformed_img, cmap=cmap)
        # Plot small red dots at transformed axon centers
        if transformed_centers:
            x_coords = [center[0] for center in transformed_centers]
            y_coords = [center[1] for center in transformed_centers]
            axes[5].scatter(x_coords, y_coords, c="red", s=1, marker="o")  # Tiny points
        axes[5].set_title(f"Transformed ROI with Axon Centers")
        axes[5].set_xlabel(f"X (101 px)")
        axes[5].set_ylabel(f"Y (101 px)")
        cbar = fig.colorbar(im5, ax=axes[5], fraction=0.046, pad=0.04)
        cbar.set_label("Intensity")

        # After plotting all panels, ensure panels 5 and 6 have matching dimensions
        # Share y-axis between histogram and transformed ROI plot
        axes[4].set_position(axes[5].get_position())

        # Add overall title
        plt.suptitle(
            f"ROI Visualization with Axon Detection: {Path(roi_path).name}", fontsize=16
        )
        plt.tight_layout(rect=(0, 0, 1, 0.95))  # Adjust for suptitle

        # Save figure if output path is provided
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Figure saved to {output_path}")

        return fig

    except Exception as e:
        logger.error(f"Error visualizing ROI with axons: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize ROI with axon detection.")
    parser.add_argument("roi_path", help="Path to ROI pickle file.")
    parser.add_argument(
        "--output", "-o", help="Path to save the visualization (optional)."
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Colormap to use for visualization (default: viridis).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold value for normalized image (0.0-1.0, default: 0.5).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=1,  # Default to 1 to avoid filtering out all pixels
        help="Minimum size of axon objects to keep (default: 1).",
    )
    parser.add_argument(
        "--no-morphology",
        action="store_true",
        help="Disable morphological operations for axon detection.",
    )
    # Add sliding window parameters
    parser.add_argument(
        "--sliding-window",
        action="store_true",
        help="Use sliding window normalization and thresholding for better detection of individual axons.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=25,
        help="Size of sliding window in pixels (default: 25).",
    )
    parser.add_argument(
        "--window-overlap",
        type=float,
        default=0.5,
        help="Overlap between adjacent windows (0.0-1.0, default: 0.5).",
    )
    parser.add_argument(
        "--min-intensity",
        type=float,
        default=0.2,
        help="Minimum intensity percentile (0.0-1.0) for sliding window detection to avoid false positives (default: 0.2).",
    )
    parser.add_argument(
        "--separate-axons",
        action="store_true",
        help="Use watershed algorithm to separate clumped axons (default: False).",
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
    logger.info(f"Visualizing ROI with axon detection: {args.roi_path}")
    if args.output:
        logger.info(f"Output will be saved to: {args.output}")

    # Visualize ROI
    try:
        fig = plot_roi_with_axons(
            args.roi_path,
            output_path=args.output,
            cmap=args.cmap,
            threshold=args.threshold,
            min_size=args.min_size,
            apply_morphology=not args.no_morphology,
            use_sliding_window=args.sliding_window,
            window_size=args.window_size,
            window_overlap=args.window_overlap,
            min_intensity_percentile=args.min_intensity,
            separate_axons=args.separate_axons,
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
