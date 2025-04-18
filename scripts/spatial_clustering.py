# scripts/spatial_clustering.py
"""
Script for performing spatial clustering analysis.

This script loads ROI data and performs spatial clustering analysis to identify
patterns in the data based on spatial distribution and anatomical regions.

Example usage:
    python spatial_clustering.py path/to/data.pkl --age adult --output output/
"""

import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.core.services.clustering import SpatialClustering
from src.utils.visualization import (
    plot_umap_and_confusion_matrix,
    plot_vertical_distribution,
)
from src.utils.logging import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform spatial clustering analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_file", type=str, help="Path to the pickle file containing ROI data"
    )
    parser.add_argument(
        "--age",
        type=str,
        default=None,
        help="Age group to analyze (optional, if not provided all age groups will be analyzed)",
    )
    parser.add_argument(
        "--clusters", type=int, default=2, help="Number of clusters to create"
    )
    parser.add_argument(
        "--output", type=str, default="output", help="Directory to save output files"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["kmeans", "gmm"],
        default="kmeans",
        help="Clustering method to use (kmeans or gmm)",
    )
    parser.add_argument(
        "--analysis",
        type=str,
        choices=["roi", "animal"],
        default="roi",
        help="Analysis type: 'roi' for individual ROIs or 'animal' for whole animals",
    )
    return parser.parse_args()


def main():
    """Run the spatial clustering analysis."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Log arguments
    logger.info(f"Data file: {args.data_file}")
    logger.info(f"Age group: {args.age}")
    logger.info(f"Number of clusters: {args.clusters}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Clustering method: {args.method}")
    logger.info(f"Analysis type: {args.analysis}")

    # Load data and create clustering service
    try:
        clustering = SpatialClustering.from_pickle(args.data_file, age_group=args.age)
        clustering.n_clusters = args.clusters

        # Prepare data based on analysis type
        if args.analysis == "roi":
            logger.info("Preparing ROI data for clustering...")
            clustering.prepare_roi_data()
            analysis_name = "ROI-based"
        else:  # args.analysis == "animal"
            logger.info("Preparing animal data for clustering...")
            clustering.prepare_animal_data()
            analysis_name = "Animal-based"

        # Perform PCA
        logger.info("Performing PCA...")
        clustering.perform_pca()

        # Perform clustering
        logger.info(
            f"Performing {args.method} clustering with {args.clusters} clusters..."
        )
        if args.method == "kmeans":
            labels = clustering.perform_clustering()
        else:  # args.method == "gmm"
            labels = clustering.perform_gmm()

        # Determine true labels based on analysis type
        if args.analysis == "roi":
            # Create visualizations for medial/lateral classification
            logger.info("Creating medial/lateral classification visualization...")
            ml_cluster_labels = clustering.determine_cluster_labels(
                labels, clustering.roi_labels_ml
            )
            ml_fig = plot_umap_and_confusion_matrix(
                clustering.all_data,
                labels,
                clustering.roi_labels_ml,
                cluster_labels=ml_cluster_labels,
                title=f"{analysis_name} Clustering: Medial/Lateral",
                output_path=os.path.join(args.output, "medial_lateral_clustering.png"),
            )

            # Create visualizations for dorsal/ventral classification
            logger.info("Creating dorsal/ventral classification visualization...")
            dv_cluster_labels = clustering.determine_cluster_labels(
                labels, clustering.roi_labels_dv
            )
            dv_fig = plot_umap_and_confusion_matrix(
                clustering.all_data,
                labels,
                clustering.roi_labels_dv,
                cluster_labels=dv_cluster_labels,
                title=f"{analysis_name} Clustering: Dorsal/Ventral",
                output_path=os.path.join(args.output, "dorsal_ventral_clustering.png"),
            )

        # Create visualizations for age classification
        logger.info("Creating age classification visualization...")
        age_cluster_labels = clustering.determine_cluster_labels(
            labels, clustering.roi_labels_age
        )
        age_fig = plot_umap_and_confusion_matrix(
            clustering.all_data,
            labels,
            clustering.roi_labels_age,
            cluster_labels=age_cluster_labels,
            title=f"{analysis_name} Clustering: Age Groups",
            output_path=os.path.join(args.output, "age_clustering.png"),
        )

        # Success message
        logger.info(f"Analysis completed successfully. Results saved to {args.output}")

    except Exception as e:
        logger.error(f"Error performing clustering analysis: {str(e)}")
        logger.exception(e)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
