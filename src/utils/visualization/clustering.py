"""Utilities for visualizing clustering results."""

# Standard Library Imports
from typing import Dict, List, Optional, Union, Tuple, Any
import os
from pathlib import Path

# Third Party Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import umap.umap_ as umap

# Internal Imports
from src.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)


def plot_umap_and_confusion_matrix(
    data: np.ndarray,
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
    cluster_labels: Optional[Dict[int, str]] = None,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    fig_size: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot UMAP projection and confusion matrix.

    Args:
        data: Data to visualize
        predicted_labels: Labels predicted by clustering algorithm
        true_labels: Ground truth labels
        cluster_labels: Mapping from cluster ID to label (optional)
        title: Plot title (optional)
        output_path: Path to save the figure (optional)
        fig_size: Figure size (width, height) in inches

    Returns:
        plt.Figure: The created figure
    """
    # Perform UMAP dimensionality reduction
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)

    # Create figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=fig_size)

    # Plot UMAP projection
    scatter = ax[0].scatter(
        embedding[:, 0], embedding[:, 1], c=predicted_labels, cmap="Spectral", s=5
    )

    ax[0].set_xlabel("UMAP1")
    ax[0].set_ylabel("UMAP2")
    ax[0].set_title("UMAP Projection")

    # Add legend if cluster labels are provided
    if cluster_labels is not None:
        legend_labels = [
            cluster_labels.get(label, f"Cluster {label}")
            for label in sorted(set(predicted_labels))
        ]
        legend = ax[0].legend(
            handles=scatter.legend_elements()[0],
            labels=legend_labels,
            title="Clusters",
            loc="upper right",
        )

    # Get label mapping
    if cluster_labels is not None:
        mapped_predicted_labels = np.array(
            [cluster_labels.get(label, str(label)) for label in predicted_labels]
        )
    else:
        mapped_predicted_labels = predicted_labels

    # Create confusion matrix
    conf_mat = confusion_matrix(true_labels, mapped_predicted_labels)

    # Plot confusion matrix
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax[1],
        xticklabels=sorted(set(mapped_predicted_labels)),
        yticklabels=sorted(set(true_labels)),
    )

    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("True")
    ax[1].set_title("Confusion Matrix")

    # Set overall title if provided
    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

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


def plot_vertical_distribution(
    roi_data: Dict[str, np.ndarray],
    output_path: Optional[str] = None,
    roi_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    fig_size: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """Plot vertical distribution of ROI data.

    Args:
        roi_data: Dictionary mapping ROI names to 2D arrays (101x101)
        output_path: Path to save the figure (optional)
        roi_names: List of ROI names to include (optional)
        title: Plot title (optional)
        fig_size: Figure size (width, height) in inches

    Returns:
        plt.Figure: The created figure
    """
    # Define default ROI layout
    roi_layout = [
        [None, "VISal", "VISrl", "VISa", "RSPagl"],
        ["VISli", "VISl", None, "VISam", "RSPd"],
        ["VISpor", None, None, "VISpm", "RSPv"],
    ]

    # Filter ROIs if names are provided
    if roi_names is not None:
        filtered_roi_data = {k: v for k, v in roi_data.items() if k in roi_names}
        roi_data = filtered_roi_data

    # Create figure
    fig, axes = plt.subplots(3, 5, figsize=fig_size)

    # Iterate over ROI layout
    for row_idx, row in enumerate(roi_layout):
        for col_idx, roi in enumerate(row):
            ax = axes[row_idx, col_idx]

            # Skip empty cells in layout
            if roi is None:
                ax.axis("off")
                continue

            # Skip if roi data is not available
            if roi not in roi_data:
                ax.text(
                    0.5,
                    0.5,
                    f"{roi}\n(No Data)",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")
                continue

            # Get data for this ROI
            data = roi_data[roi]

            # Calculate vertical distribution
            vertical_dist = np.sum(data, axis=1)

            # Plot vertical distribution
            ax.plot(vertical_dist, range(len(vertical_dist)))
            ax.set_title(roi)
            ax.invert_yaxis()  # Invert Y-axis to match anatomical convention
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Depth")

    # Set overall title if provided
    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

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
