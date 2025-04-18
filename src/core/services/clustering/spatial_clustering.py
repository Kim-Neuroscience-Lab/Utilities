"""Spatial clustering service."""

# Standard Library Imports
from typing import List, Dict, Any, Tuple, Optional, Union
import pickle

# Third Party Imports
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
import umap.umap_ as umap
from logging import Logger

# Internal Imports
from src.core.models.animal import Animal
from src.core.models.spatial.roi import ROI
from src.utils.logging import get_logger

# Initialize logger
logger: Logger = get_logger(__name__)


class SpatialClustering:
    """Service for clustering spatial data.

    This service provides methods for clustering ROI data based on
    spatial patterns and anatomical regions.

    Attributes:
        animals: List of animals to analyze
        n_clusters: Number of clusters to create
        all_data: Processed data for clustering
        roi_labels_ml: Medial/Lateral labels for ROIs
        roi_labels_dv: Dorsal/Ventral labels for ROIs
        roi_labels_age: Age labels for animals
    """

    def __init__(self, animals: List[Animal], n_clusters: int = 2):
        """Initialize the clustering service.

        Args:
            animals: List of animals to analyze
            n_clusters: Number of clusters to create (default: 2)
        """
        self.animals = animals
        self.n_clusters = n_clusters
        self.all_data = []
        self.roi_labels_ml = []  # Medial/Lateral labels
        self.roi_labels_dv = []  # Dorsal/Ventral labels
        self.roi_labels_age = []  # Age labels

        # Classification maps based on typical mouse brain organization
        self._medial_lateral_map = {
            "tea": "lateral",
            "visal": "lateral",
            "visrl": "medial",
            "visa": "medial",
            "rspagl": "medial",
            "visli": "lateral",
            "visl": "lateral",
            "visam": "medial",
            "rspd": "medial",
            "vispor": "lateral",
            "vispl": "lateral",
            "vispm": "medial",
            "rspv": "medial",
            "str": "medial",
        }

        self._dorsal_ventral_map = {
            "tea": "dorsal",
            "visal": "dorsal",
            "visrl": "dorsal",
            "visa": "dorsal",
            "rspagl": "dorsal",
            "visli": "ventral",
            "visl": "ventral",
            "visam": "dorsal",
            "rspd": "dorsal",
            "vispor": "ventral",
            "vispl": "ventral",
            "vispm": "dorsal",
            "rspv": "dorsal",
            "str": "ventral",
        }

    def prepare_roi_data(self) -> None:
        """Prepare ROI data for clustering analysis."""
        self.all_data = []
        self.roi_labels_ml = []
        self.roi_labels_dv = []
        self.roi_labels_age = []

        for animal in self.animals:
            for region_id, region in animal.regions.items():
                if not isinstance(region, ROI) or region.mask is None:
                    continue

                # Skip certain regions if needed
                roi_name = region.roi.lower() if region.roi else ""
                if roi_name in ["visp", "str"]:
                    continue

                # Append the ROI mask data
                self.all_data.append(region.mask)

                # Append classification labels
                ml_label = self._medial_lateral_map.get(roi_name, "unknown")
                dv_label = self._dorsal_ventral_map.get(roi_name, "unknown")
                self.roi_labels_ml.append(ml_label)
                self.roi_labels_dv.append(dv_label)
                self.roi_labels_age.append(animal.age or "unknown")

        # Convert to numpy arrays
        self.all_data = np.array(self.all_data)
        self.roi_labels_ml = np.array(self.roi_labels_ml)
        self.roi_labels_dv = np.array(self.roi_labels_dv)
        self.roi_labels_age = np.array(self.roi_labels_age)

        logger.info(f"Prepared {len(self.all_data)} ROIs for clustering analysis")

    def prepare_animal_data(self) -> None:
        """Prepare whole-animal data for clustering analysis."""
        max_length = 0
        whole_brains = []  # List to hold all concatenated data arrays

        # First pass: find the maximum length needed
        for animal in self.animals:
            whole_brain = []
            for region_id, region in animal.regions.items():
                if not isinstance(region, ROI) or region.mask is None:
                    continue

                # Skip certain regions if needed
                roi_name = region.roi.lower() if region.roi else ""
                if roi_name in ["visp", "str"]:
                    continue

                # Flatten the mask data
                whole_brain.append(region.mask.reshape(-1))

            if not whole_brain:
                continue
            elif len(whole_brain) == 1:
                concatenated = whole_brain[0]
            else:
                concatenated = np.concatenate(whole_brain)

            if concatenated.size > max_length:
                max_length = concatenated.size

            whole_brains.append(concatenated)

        # Second pass: pad arrays to ensure equal lengths
        padded_brains = []
        for brain in whole_brains:
            padded_brain = np.pad(
                brain, (0, max_length - brain.size), "constant", constant_values=0
            )
            padded_brains.append(padded_brain)

        self.all_data = np.array(padded_brains)

        # Set age labels
        self.roi_labels_age = [
            animal.age or "unknown" for animal in self.animals if animal.regions
        ]

        logger.info(f"Prepared {len(self.all_data)} animals for clustering analysis")

    def perform_pca(self, n_components: int = 2) -> None:
        """Perform PCA dimensionality reduction.

        Args:
            n_components: Number of PCA components (default: 2)
        """
        if not isinstance(self.all_data, np.ndarray) or len(self.all_data) == 0:
            logger.warning("No data available for PCA")
            return

        # Flatten data if needed
        if len(self.all_data.shape) > 2:
            data_reshaped = self.all_data.reshape(self.all_data.shape[0], -1)
        else:
            data_reshaped = self.all_data

        # Perform PCA
        pca = PCA(n_components=n_components)
        self.all_data = pca.fit_transform(data_reshaped)

        # Log explained variance
        explained_variance = pca.explained_variance_ratio_
        logger.info(f"PCA explained variance: {explained_variance}")

    def perform_clustering(self) -> np.ndarray:
        """Perform K-means clustering.

        Returns:
            np.ndarray: Cluster labels for each data point
        """
        if not isinstance(self.all_data, np.ndarray) or len(self.all_data) == 0:
            logger.warning("No data available for clustering")
            return np.array([])

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        kmeans.fit(self.all_data)

        return kmeans.labels_

    def perform_gmm(self) -> np.ndarray:
        """Perform Gaussian Mixture Model clustering.

        Returns:
            np.ndarray: Cluster labels for each data point
        """
        if not isinstance(self.all_data, np.ndarray) or len(self.all_data) == 0:
            logger.warning("No data available for GMM clustering")
            return np.array([])

        gmm = GaussianMixture(n_components=self.n_clusters, n_init=10)
        gmm.fit(self.all_data)

        return gmm.predict(self.all_data)

    def determine_cluster_labels(
        self, predicted_labels: np.ndarray, real_labels: np.ndarray
    ) -> Dict[int, str]:
        """Determine the most probable label for each cluster.

        Args:
            predicted_labels: Cluster labels from clustering algorithm
            real_labels: Ground truth labels

        Returns:
            Dict[int, str]: Mapping from cluster ID to most probable label
        """
        if len(predicted_labels) == 0 or len(real_labels) == 0:
            logger.warning("Empty labels provided for cluster label determination")
            return {}

        # Create a dictionary to hold the count of each label within each cluster
        cluster_counts = {}
        for cluster_id in set(predicted_labels):
            # For each label, check how many times it appears in each cluster
            current_cluster_indices = [
                i for i, x in enumerate(predicted_labels) if x == cluster_id
            ]
            labels_in_cluster = [real_labels[i] for i in current_cluster_indices]

            label_counts = {label: 0 for label in set(real_labels)}
            for label in labels_in_cluster:
                label_counts[label] += 1

            # Store the label counts dictionary in our cluster counts dictionary
            cluster_counts[cluster_id] = label_counts

        # Assign labels to clusters
        cluster_labels = {}
        for label in set(real_labels):
            # Sort clusters by label count
            sorted_clusters = sorted(
                cluster_counts.keys(),
                key=lambda x: cluster_counts[x][label],
                reverse=True,
            )

            # Assign the label to the cluster with the highest count
            for cid in sorted_clusters:
                if cid not in cluster_labels:
                    cluster_labels[cid] = label
                    break
                # Continue if this cluster already has a label

        return cluster_labels

    @classmethod
    def from_pickle(
        cls, pickle_path: str, age_group: str = None
    ) -> "SpatialClustering":
        """Create a SpatialClustering instance from a pickle file.

        Args:
            pickle_path: Path to the pickle file
            age_group: Age group to load (optional)

        Returns:
            SpatialClustering: A new SpatialClustering instance
        """
        try:
            animals = []
            with open(pickle_path, "rb") as f:
                raw = pickle.load(f)

                # If age_group is provided, only load that age group
                if age_group is not None and age_group in raw:
                    age_data = raw[age_group]
                    for animal_name, animal_regions in age_data.items():
                        # Create an Animal instance
                        animal = Animal(animal_id=animal_name, age=age_group)

                        # Add ROIs to the animal
                        for roi_name, roi_data in animal_regions.items():
                            # Create ROI
                            roi = ROI(
                                region_id=roi_name.lower(),
                                roi=roi_name,
                                intensity=None,  # We don't have intensity data here
                                mask=roi_data,  # The data is already processed as a mask
                            )
                            animal.add_region(roi)

                        animals.append(animal)
                else:
                    # Load all age groups
                    for age, age_data in raw.items():
                        for animal_name, animal_regions in age_data.items():
                            # Create an Animal instance
                            animal = Animal(animal_id=animal_name, age=age)

                            # Add ROIs to the animal
                            for roi_name, roi_data in animal_regions.items():
                                # Create ROI
                                roi = ROI(
                                    region_id=roi_name.lower(),
                                    roi=roi_name,
                                    intensity=None,  # We don't have intensity data here
                                    mask=roi_data,  # The data is already processed as a mask
                                )
                                animal.add_region(roi)

                            animals.append(animal)

            return cls(animals=animals)
        except Exception as e:
            logger.error(f"Error loading pickle file: {str(e)}")
            return cls(animals=[])
