"""
Depth Analyzer module for computing depth distributions from pickle files.

This module provides functionality to analyze depth distributions across multiple brain segments
and regions, following a standardized binning approach.
"""

# Standard Library Imports
import os
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Any,
    Union,
    TypedDict,
    cast,
    Sequence,
    Set,
)
import pickle
import logging
import sys

# Third Party Imports
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Internal Imports
from src.utils.logging import get_logger
from src.core.models.spatial.roi import ROI

# Initialize logger
logger = get_logger(__name__)

# Define standard region order
STANDARD_REGION_ORDER = [
    "RSPv",
    "RSPd",
    "RSPagl",
    "VISpm",
    "VISam",
    "VISa",
    "VISrl",
    "VISal",
    "VISl",
    "VISli",
    "VISpl",
    "VISpor",
]


class RegionData(TypedDict):
    """Type definition for region-specific data."""

    masks: Sequence[NDArray[np.float64]]  # List of 2D masks, one per section
    mask_sums: float
    normalized_sum: float


class AnimalData(TypedDict):
    """Type definition for animal-specific data."""

    regions: Dict[str, RegionData]


class AgeGroupData(TypedDict):
    """Type definition for age group data."""

    animals: Dict[str, AnimalData]


class DistributionResult(TypedDict):
    """Type definition for distribution result."""

    regions: Dict[str, pd.DataFrame]
    normalized_sums: pd.DataFrame


class SafeUnpickler(pickle.Unpickler):
    """Custom unpickler that handles missing modules gracefully and converts paths."""

    def find_class(self, module: str, name: str) -> Any:
        """Override find_class to handle missing modules and path conversions.

        Args:
            module: The module name
            name: The class name

        Returns:
            A placeholder class or the actual class if found
        """
        # Handle pathlib objects specially
        if module == "pathlib" and name in ("WindowsPath", "PosixPath", "Path"):
            return Path

        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            # Create a simple placeholder for the missing class
            placeholder_dict = {
                "__repr__": lambda self: f"<Unpickleable object from module {module}, class {name}>",
                "__str__": lambda self: f"<Unpickleable object from module {module}, class {name}>",
                "_module": module,
                "_class_name": name,
                "_attributes": {},
            }
            return type(f"MissingClass_{name}", (), placeholder_dict)

    def persistent_load(self, pid: Any) -> Any:
        """Handle persistent IDs during unpickling."""
        return pid


class DepthAnalyzer:
    """Analyzer for computing depth distributions from pickle files.

    This class handles loading pickle files containing ROI data and computing
    depth distributions across brain regions.
    """

    def __init__(self, num_bins: int = 101) -> None:
        """Initialize the DepthAnalyzer.

        Args:
            num_bins: Number of bins for depth analysis (default: 101)
        """
        self.num_bins = num_bins
        self._region_data: Dict[str, Dict[str, Dict[str, RegionData]]] = {}
        self._region_order: List[str] = STANDARD_REGION_ORDER.copy()
        self._additional_regions: Set[str] = set()  # Track non-standard regions

    def load_pickle_file(self, file_path: str) -> None:
        """Load and process a pickle file.

        The pickle file should contain data structured as:
        data[age_group][animal][region] = List[ROI]
        where each ROI object has a mask attribute.

        Args:
            file_path: Path to the pickle file

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")

        try:
            # Increase recursion limit temporarily for complex pickle files
            original_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(3000)

            try:
                with open(file_path, "rb") as f:
                    data = SafeUnpickler(f).load()
            finally:
                # Restore original recursion limit
                sys.setrecursionlimit(original_limit)

            # Process each age group
            for age_group, animals in data.items():
                if not isinstance(animals, dict):
                    logger.warning(
                        f"Skipping invalid age group {age_group}: not a dictionary"
                    )
                    continue

                if age_group not in self._region_data:
                    self._region_data[age_group] = {}

                # Process each animal
                for animal_id, regions in animals.items():
                    if not isinstance(regions, dict):
                        logger.warning(
                            f"Skipping invalid animal {animal_id}: not a dictionary"
                        )
                        continue

                    if animal_id not in self._region_data[age_group]:
                        self._region_data[age_group][animal_id] = {}

                    # Process each region
                    for region_name, roi_list in regions.items():
                        if not isinstance(roi_list, list):
                            logger.warning(
                                f"Skipping invalid region {region_name}: not a list"
                            )
                            continue

                        # Standardize region name and handle non-standard regions
                        std_region_name = self._standardize_region_name(region_name)
                        if std_region_name not in STANDARD_REGION_ORDER:
                            # Add to additional regions if not already there
                            if std_region_name not in self._additional_regions:
                                self._additional_regions.add(std_region_name)
                                logger.info(
                                    f"Adding non-standard region to analysis: {std_region_name}"
                                )

                        # Collect all masks from ROIs
                        masks: List[NDArray[np.float64]] = []
                        for roi in roi_list:
                            # Try to get mask from ROI object
                            mask = None
                            if (
                                hasattr(roi, "_attributes")
                                and "mask" in roi._attributes
                            ):
                                mask = roi._attributes["mask"]
                            elif hasattr(roi, "mask"):
                                mask = roi.mask

                            if mask is not None and isinstance(mask, np.ndarray):
                                # Ensure mask is float64
                                mask = mask.astype(np.float64)
                                masks.append(mask)
                            else:
                                logger.warning(f"No valid mask found in ROI object")

                        # Process the masks for this region
                        if masks:
                            mask_sums = float(sum(np.sum(mask) for mask in masks))
                            normalized_sum = float(sum(np.mean(mask) for mask in masks))

                            region_data: RegionData = {
                                "masks": masks,
                                "mask_sums": mask_sums,
                                "normalized_sum": normalized_sum,
                            }
                            self._region_data[age_group][animal_id][
                                std_region_name
                            ] = region_data
                            logger.info(
                                f"Processed {len(masks)} masks for {age_group}/{animal_id}/{std_region_name}"
                            )

            logger.info(f"Successfully loaded data from {file_path}")

        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise

    @staticmethod
    def _standardize_region_name(name: str) -> str:
        """Standardize region names to match the expected format.

        Args:
            name: Original region name

        Returns:
            str: Standardized region name
        """
        # Convert to uppercase
        name = name.upper()

        # Handle common variations
        name_map = {
            "RSPV": "RSPv",
            "RSPD": "RSPd",
            "RSPAGL": "RSPagl",
            "VISPM": "VISpm",
            "VISAM": "VISam",
            "VISA": "VISa",
            "VISRL": "VISrl",
            "VISAL": "VISal",
            "VISL": "VISl",
            "VISLI": "VISli",
            "VISPL": "VISpl",
            "VISPOR": "VISpor",
        }

        return name_map.get(name, name)

    def _get_region_order(self) -> List[str]:
        """Get the complete region order including additional regions.

        Returns:
            List[str]: List of regions in order (standard regions followed by additional regions)
        """
        return self._region_order + sorted(list(self._additional_regions))

    def generate_depth_distribution(
        self, age_group: str, animal_id: str
    ) -> DistributionResult:
        """Generate depth distribution CSVs for a specific animal.

        Args:
            age_group: Age group identifier
            animal_id: Animal identifier

        Returns:
            DistributionResult: Dictionary containing region-specific depth DataFrames
            and normalized mask sum DataFrame. The normalized_sums DataFrame has columns:
            region, row1, row2, ..., rowN where each row value is the sum of that row
            across all sections divided by (100 * number of sections).

        Raises:
            ValueError: If no data found for the specified age group and animal ID
        """
        if (
            age_group not in self._region_data
            or animal_id not in self._region_data[age_group]
        ):
            raise ValueError(f"No data found for {age_group}/{animal_id}")

        logger.info(f"Generating depth distribution for {age_group}/{animal_id}")
        animal_data = self._region_data[age_group][animal_id]

        # Process each region separately
        region_dfs: Dict[str, pd.DataFrame] = {}
        normalized_sums_data: List[Dict[str, Union[str, float]]] = []

        # Find maximum number of rows across all regions and masks
        max_rows = 0
        for region_data in animal_data.values():
            masks = region_data["masks"]
            max_rows = max(max_rows, max(mask.shape[0] for mask in masks))

        # Use complete region order (standard + additional)
        for region in self._get_region_order():
            if region in animal_data:
                region_data = animal_data[region]
                masks = region_data["masks"]
                num_sections = len(masks)

                # Create a DataFrame with section_index and rows for depth distribution
                section_indices = list(range(num_sections))
                data_dict = {"section_index": section_indices}

                # Calculate row sums for both depth distribution and normalized sums
                row_sums = np.zeros(max_rows)
                for row_idx in range(max_rows):
                    row_values = []
                    for mask in masks:
                        # If this mask has enough rows, add the row, otherwise add zeros
                        if row_idx < mask.shape[0]:
                            row_value = float(mask[row_idx].sum())
                            row_values.append(row_value)
                            row_sums[row_idx] += row_value
                        else:
                            row_values.append(0.0)
                    data_dict[f"row{row_idx + 1}"] = row_values

                # Create depth distribution DataFrame
                df = pd.DataFrame(data_dict)
                region_dfs[region] = df

                # Calculate normalized row sums (divide by 100 * number of sections)
                normalized_row_sums: Dict[str, Union[str, float]] = {
                    "region": str(region),  # Ensure region is a string
                }
                for row_idx in range(max_rows):
                    normalized_row_sums[f"row{row_idx + 1}"] = float(
                        row_sums[row_idx] / (100 * num_sections)
                    )
                normalized_sums_data.append(normalized_row_sums)

        return {
            "regions": region_dfs,
            "normalized_sums": pd.DataFrame(normalized_sums_data),
        }

    def save_distributions(
        self, age_group: str, animal_id: str, output_dir: Path
    ) -> None:
        """Save depth distributions to CSV files.

        Args:
            age_group: Age group identifier
            animal_id: Animal identifier
            output_dir: Directory to save CSV files

        Raises:
            ValueError: If output directory creation fails
        """
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        distributions = self.generate_depth_distribution(age_group, animal_id)

        # Save individual region CSVs
        for region, df in distributions["regions"].items():
            filename = f"{age_group}_{animal_id}_{region}_depths.csv"
            output_path = output_dir / filename
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {output_path}")

        # Save normalized mask sums CSV
        filename = f"{age_group}_{animal_id}_normalized_mask_sums.csv"
        output_path = output_dir / filename
        distributions["normalized_sums"].to_csv(output_path, index=False)
        logger.info(f"Saved {output_path}")
