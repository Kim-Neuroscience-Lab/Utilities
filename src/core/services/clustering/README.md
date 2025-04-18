# Spatial Clustering Services

This module provides services for performing spatial clustering analysis on ROI (Region of Interest) data.

## Overview

The spatial clustering functionality allows analysis of brain regions based on their spatial patterns and anatomical organization. It can identify patterns across different age groups and brain regions.

## Key Components

### SpatialClustering

The `SpatialClustering` class provides methods for:

- Loading and preparing ROI data from pickle files
- Performing dimensionality reduction using PCA
- Clustering data using KMeans or Gaussian Mixture Models
- Determining cluster labels based on ground truth anatomical knowledge
- Supporting both ROI-level and animal-level analyses

## Usage Example

```python
from src.core.services.clustering import SpatialClustering
from src.utils.visualization import plot_umap_and_confusion_matrix

# Create a clustering service from a pickle file
clustering = SpatialClustering.from_pickle("path/to/data.pkl", age_group="adult")

# Set clustering parameters
clustering.n_clusters = 2

# Prepare ROI data
clustering.prepare_roi_data()

# Perform PCA
clustering.perform_pca()

# Perform clustering
labels = clustering.perform_clustering()  # or clustering.perform_gmm()

# Determine cluster labels
cluster_labels = clustering.determine_cluster_labels(
    labels, clustering.roi_labels_ml
)

# Visualize results
plot_umap_and_confusion_matrix(
    clustering.all_data,
    labels,
    clustering.roi_labels_ml,
    cluster_labels=cluster_labels,
    title="ROI Clustering: Medial/Lateral",
    output_path="output/medial_lateral_clustering.png"
)
```

## Anatomical Maps

The service includes predefined anatomical maps for:

- Medial/Lateral classification
- Dorsal/Ventral classification

These maps are based on typical mouse brain organization and help validate the clustering results against anatomical knowledge.

## Demo Script

A demonstration script is provided in `scripts/spatial_clustering.py` that shows how to use the spatial clustering functionality.

### Running the Demo

```bash
python scripts/spatial_clustering.py path/to/data.pkl --age adult --output output/ --clusters 2 --method kmeans --analysis roi
```

Options:

- `--age`: Age group to analyze (e.g., "p3", "p12", "p20", "adult")
- `--clusters`: Number of clusters to create (default: 2)
- `--output`: Directory to save output files (default: "output")
- `--method`: Clustering method to use ("kmeans" or "gmm", default: "kmeans")
- `--analysis`: Analysis type ("roi" for individual ROIs or "animal" for whole animals, default: "roi")
