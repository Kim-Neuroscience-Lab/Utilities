# src/analysis/roi_area_analyzer.py
"""
ROI Area Analyzer module for computing areas of regions of interest from pickle files.

This module provides functionality to analyze the area of ROIs across multiple brain segments
and regions, following the structure of the Allen Brain Atlas.
"""

# Standard Library Imports
import os
import sys
import platform
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import time
from datetime import datetime

# Third Party Imports
from logging import Logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import sparse
from tqdm import tqdm
from pydantic import BaseModel, Field, ConfigDict

# Internal Imports
from src.utils.hardware import detect_gpu_backend
from src.utils.logging import get_logger
from src.utils.performance import batch_process, timed_execution
from src.utils.constants import (
    MB_IN_BYTES,
    ROI_SIZE_THRESHOLD,
    BATCH_SIZE,
    BUFFER_SIZE,
)
from src.core.models.animal import Animal
from src.analysis.area_analyzer import AreaAnalyzer

# Configure logger
logger = logging.getLogger("ROIAreaAnalyzer")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)


@dataclass
class ROIAreaResult:
    """Data class for storing ROI area analysis results.

    Attributes:
        segment_id: ID of the brain segment
        region_name: Name of the brain region
        area_pixels: Area of the ROI in pixels
    """

    segment_id: str
    region_name: str
    area_pixels: int


class ROIAreaAnalyzer(AreaAnalyzer):
    """Analyzer for computing areas of ROIs from pickle files.

    This class provides methods to:
    1. Process ROI data from pickle files
    2. Compute areas using different methods (fast, GPU, sparse) based on data size
    3. Generate summaries and statistics
    4. Track computation method usage
    5. Handle nested directory structures and create analysis outputs

    Attributes:
        input_dir: Directory containing ROI pickle files
        max_workers: Maximum number of worker threads for parallel processing
        use_gpu: Whether to use GPU acceleration if available
        gpu_backend: GPU backend to use (cupy, mps, or None)
        gpu_module: GPU module to use (cupy or torch)
        method_counts: Dictionary tracking computation method usage
        cache: Dictionary caching processed ROI data
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Additional fields
    gpu_backend: Optional[str] = Field(None, description="GPU backend type")
    gpu_module: Optional[Any] = Field(None, description="GPU module")
    method_counts: Dict[str, int] = Field(
        default_factory=lambda: {"fast": 0, "gpu": 0, "sparse": 0},
        description="Method usage counters",
    )
    cache: Dict[str, Any] = Field(
        default_factory=dict,
        description="Cache for processed results",
    )

    def __init__(
        self,
        input_dir: Union[str, Path],
        max_workers: Optional[int] = None,
        use_gpu: bool = False,
    ) -> None:
        """Initialize the ROI area analyzer.

        Args:
            input_dir: Directory containing ROI pickle files
            max_workers: Maximum number of worker threads for parallel processing
            use_gpu: Whether to use GPU acceleration if available
        """
        # Initialize base class first
        super().__init__(
            input_dir=str(input_dir),
            max_workers=max_workers if max_workers is not None else os.cpu_count(),  # type: ignore
            use_gpu=use_gpu,
        )

        # GPU initialization
        if use_gpu:
            gpu_backend, gpu_module = detect_gpu_backend()
            self.gpu_backend = gpu_backend
            self.gpu_module = gpu_module
            if self.gpu_backend:
                logger.info(f"Using GPU acceleration with {self.gpu_backend}")
                self.use_gpu = True
            else:
                logger.warning("No GPU backend available, falling back to CPU")
                self.use_gpu = False
        else:
            logger.info("Using CPU computation (GPU acceleration disabled)")
            self.gpu_backend = None
            self.gpu_module = None
            self.use_gpu = False

    def __hash__(self) -> int:
        """Make ROIAreaAnalyzer hashable by using input_dir as the hash key."""
        return hash(str(self.input_dir))

    def __eq__(self, other: object) -> bool:
        """Define equality for ROIAreaAnalyzer based on input_dir."""
        if not isinstance(other, ROIAreaAnalyzer):
            return NotImplemented
        return str(self.input_dir) == str(other.input_dir)

    def _compute_roi_area_fast(
        self,
        roi_data: Dict[str, Any],
    ) -> int:
        """Compute area by counting non-zero values (fastest method).

        This method is most efficient for small ROIs. It counts only the non-zero
        intensity values in the ROI.

        Args:
            roi_data: Dictionary containing ROI data

        Returns:
            int: Area of the ROI in pixels (count of non-zero values)
        """
        self.method_counts["fast"] += 1
        return sum(1 for val in roi_data["roi"].values() if val > 0)

    def _compute_roi_area_gpu(
        self,
        roi_data: Dict[str, Any],
        threshold: int = 0,
    ) -> int:
        """Compute area using available GPU acceleration.
        Optimized for batch processing and GPU utilization.

        Args:
            roi_data: Dictionary containing ROI data
            threshold: Threshold value for pixel intensity

        Returns:
            int: Area of the ROI in pixels
        """
        self.method_counts["gpu"] += 1
        values: np.ndarray = np.array(list(roi_data["roi"].values()), dtype=np.uint8)

        if not self.use_gpu:
            return int(np.count_nonzero(values > threshold))

        if self.gpu_backend == "cupy":
            # Process in larger chunks for better GPU utilization
            try:
                # Transfer to GPU
                logger.debug(f"Transferring {values.shape[0]} values to GPU")
                values_gpu: Any = self.gpu_module.asarray(values)  # type: ignore

                # Create mask on GPU
                mask_gpu: Any = values_gpu > threshold

                # Compute result on GPU
                area: int = int(self.gpu_module.sum(mask_gpu))  # type: ignore

                # Clean up GPU memory
                del values_gpu
                del mask_gpu
                self.gpu_module.get_default_memory_pool().free_all_blocks()  # type: ignore

                return area

            except Exception as e:
                logger.warning(f"GPU processing failed: {str(e)}. Falling back to CPU.")
                return int(np.count_nonzero(values > threshold))

        elif self.gpu_backend == "mps":
            try:
                # Convert to torch tensor and move to MPS
                logger.debug(f"Processing {values.shape[0]} values on MPS")
                values_gpu: Any = self.gpu_module.from_numpy(values).to("mps")  # type: ignore

                # Process on GPU
                mask_gpu: Any = values_gpu > threshold
                area: int = int(mask_gpu.sum().item())  # type: ignore

                # Clean up
                del values_gpu
                del mask_gpu
                self.gpu_module.mps.empty_cache()  # type: ignore

                return area

            except Exception as e:
                logger.warning(f"MPS processing failed: {str(e)}. Falling back to CPU.")
                return int(np.count_nonzero(values > threshold))

        return int(np.count_nonzero(values > threshold))

    def _compute_roi_area_sparse(self, roi_data: Dict[str, Any]) -> int:
        """Compute area using memory-efficient sparse matrix representation.

        This method is optimal for large ROIs when GPU is not available, as it
        minimizes memory usage through sparse matrix representation.

        Args:
            roi_data: Dictionary containing ROI data

        Returns:
            int: Area of the ROI in pixels
        """
        self.method_counts["sparse"] += 1
        coords: np.ndarray = np.array(list(roi_data["roi"].keys()))
        values: np.ndarray = np.array(list(roi_data["roi"].values()), dtype=np.uint8)

        y_max: int = int(coords.max(axis=0)[0]) + 1
        x_max: int = int(coords.max(axis=0)[1]) + 1
        sparse_mat: sparse.coo_matrix = sparse.coo_matrix(
            (values, (coords[:, 0], coords[:, 1])),
            shape=(y_max, x_max),
            dtype=np.uint8,
        )
        return int((sparse_mat.data > 0).sum())

    def _compute_roi_area(self, roi_data: Dict[str, Any]) -> int:
        """Compute area of ROI in pixels using the most appropriate method.

        The method selection is based on:
        1. ROI size (small ROIs use fast method)
        2. GPU availability (large ROIs use GPU if available)
        3. Memory efficiency (falls back to sparse method for large ROIs without GPU)

        Args:
            roi_data: Dictionary containing ROI data

        Returns:
            int: Area of the ROI in pixels
        """
        if not isinstance(roi_data, dict) or "roi" not in roi_data:
            raise ValueError("ROI data must be a dictionary containing 'roi' key")

        roi_coords: Dict[str, Any] = roi_data["roi"]
        if not isinstance(roi_coords, dict):
            raise ValueError("ROI coordinates must be a dictionary")

        roi_size: int = len(roi_coords)

        # For small ROIs, use fast method
        if roi_size < ROI_SIZE_THRESHOLD:
            logger.debug(f"Using fast method for ROI with {roi_size} pixels")
            return self._compute_roi_area_fast(roi_data)

        # For large ROIs, use GPU if available
        if self.use_gpu:
            logger.debug(f"Using GPU method for ROI with {roi_size} pixels")
            return self._compute_roi_area_gpu(roi_data)

        # Otherwise, use sparse matrix for memory efficiency
        logger.debug(f"Using sparse matrix method for ROI with {roi_size} pixels")
        return self._compute_roi_area_sparse(roi_data)

    @lru_cache(maxsize=1024)
    def _parse_filename(
        self,
        filename: str,
    ) -> Tuple[str, str, str]:
        """Parse ROI filename into components.

        Args:
            filename: Name of the ROI file

        Returns:
            Tuple of (animal_id, segment_id, region_name)
        """
        base: str = os.path.splitext(filename)[0]
        parts: List[str] = base.split("_")
        return parts[0], parts[1], "_".join(parts[2:])

    def _read_pickle_file(
        self,
        file_path: Path,
    ) -> Any:
        """Read pickle file with optimized buffering.

        Args:
            file_path: Path to the pickle file

        Returns:
            Dictionary containing ROI data
        """
        with open(file_path, "rb", buffering=BUFFER_SIZE) as f:
            return pickle.load(f)

    def _process_single_file(
        self,
        file: Path,
    ) -> Optional[Dict[str, Any]]:
        """Process a single ROI file.

        Args:
            file: Path to the ROI file

        Returns:
            Dictionary containing processed ROI data
        """
        try:
            # Use cached result if available
            cache_key = str(file)
            if cache_key in self.cache:
                return self.cache[cache_key]

            # Parse filename first to avoid loading file if filename is invalid
            animal_id: str
            segment_id: str
            region_name: str
            animal_id, segment_id, region_name = self._parse_filename(file.name)

            roi_data: Dict[str, Any] = self._read_pickle_file(file)
            area: int = self._compute_roi_area(roi_data)

            result: Dict[str, Any] = {
                "animal_id": animal_id,
                "segment_id": segment_id,
                "region_name": region_name,
                "area_pixels": area,
                "file_path": str(file),
            }

            # Cache the result
            self.cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Error processing {file.name}: {str(e)}")
            return None

    def _process_batch_gpu(
        self,
        roi_data_list: List[Dict[str, Any]],
        batch_size: int = 32,  # Optimal batch size for GPU processing
    ) -> List[int]:
        """Process multiple ROIs on GPU in a single batch for better efficiency.

        Args:
            roi_data_list: List of dictionaries containing ROI data
            batch_size: Size of mini-batches for GPU processing

        Returns:
            List of areas of the ROIs in pixels
        """
        if not roi_data_list:
            return []

        try:
            # Pre-allocate memory for results
            results: List[int] = []

            # Process in mini-batches for better memory management
            for i in range(0, len(roi_data_list), batch_size):
                batch = roi_data_list[i : i + batch_size]

                # Extract values from batch
                batch_values: List[np.ndarray] = [
                    np.array(list(data["roi"].values()), dtype=np.uint8)
                    for data in batch
                ]

                if self.gpu_backend == "mps":
                    try:
                        # Convert all arrays to tensors at once
                        tensors = [
                            self.gpu_module.from_numpy(values).to("mps")  # type: ignore
                            for values in batch_values
                        ]

                        # Process each tensor
                        for tensor in tensors:
                            mask = tensor > 0
                            area = int(mask.sum().item())
                            results.append(area)
                            # Clean up immediately
                            del mask
                            del tensor
                            # Update method count for each ROI processed
                            self.method_counts["gpu"] += 1

                        # Clear GPU cache after batch
                        self.gpu_module.mps.empty_cache()  # type: ignore

                    except Exception as e:
                        logger.warning(
                            f"MPS batch processing failed: {str(e)}. Processing individually."
                        )
                        for data in batch:
                            area = self._compute_roi_area_gpu(data)
                            results.append(area)
                            # Update method count for fallback processing
                            self.method_counts["gpu"] += 1

                elif self.gpu_backend == "cupy":
                    try:
                        # Process batch on CUDA
                        for values in batch_values:
                            # Transfer and process in one go
                            values_gpu = self.gpu_module.asarray(values)  # type: ignore
                            area = int(self.gpu_module.count_nonzero(values_gpu))  # type: ignore
                            results.append(area)
                            del values_gpu
                            # Update method count for each ROI processed
                            self.method_counts["gpu"] += 1

                        # Clear GPU memory after batch
                        self.gpu_module.get_default_memory_pool().free_all_blocks()  # type: ignore

                    except Exception as e:
                        logger.warning(
                            f"CUDA batch processing failed: {str(e)}. Processing individually."
                        )
                        for data in batch:
                            area = self._compute_roi_area_gpu(data)
                            results.append(area)
                            # Update method count for fallback processing
                            self.method_counts["gpu"] += 1

            return results

        except Exception as e:
            logger.error(
                f"Batch GPU processing failed: {str(e)}. Falling back to individual processing."
            )
            results = []
            for data in roi_data_list:
                area = self._compute_roi_area_gpu(data)
                results.append(area)
                # Update method count for fallback processing
                self.method_counts["gpu"] += 1
            return results

    def _process_batch(self, batch: List[Path]) -> List[Dict[str, Any]]:
        """Process a batch of ROI files.

        Args:
            batch: List of paths to ROI files

        Returns:
            List of dictionaries containing analysis results
        """
        results = []
        roi_data_list = []
        file_info_list = []

        # First, load all ROI data and collect file info
        for file_path in batch:
            start_time = time.time()
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            logger.info(f"Processing {file_path.name} (Size: {file_size:.2f} MB)")

            try:
                # Check cache first
                cache_key = str(file_path)
                if cache_key in self.cache:
                    logger.info(f"Using cached results for {file_path.name}")
                    results.append(self.cache[cache_key])
                    continue

                # Load ROI data
                with open(file_path, "rb") as f:
                    roi_data = pickle.load(f)

                # Get file info
                animal_id, segment_id, region_name = self._parse_filename(
                    file_path.name
                )

                # Store data and info
                roi_data_list.append(roi_data)
                file_info_list.append(
                    (file_path, animal_id, segment_id, region_name, start_time)
                )

            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {str(e)}")
                continue

        if not roi_data_list:
            return results

        # Determine processing method based on ROI sizes
        roi_sizes = [len(data["roi"]) for data in roi_data_list]
        use_gpu_batch = (
            self.use_gpu
            and any(size >= ROI_SIZE_THRESHOLD for size in roi_sizes)
            and len(roi_sizes) > 1  # Only use batch processing for multiple ROIs
        )

        if use_gpu_batch:
            # Process batch on GPU
            try:
                areas = self._process_batch_gpu(roi_data_list)

                # Create results
                for (
                    file_path,
                    animal_id,
                    segment_id,
                    region_name,
                    start_time,
                ), area in zip(file_info_list, areas):
                    result = {
                        "animal_id": animal_id,
                        "segment_id": segment_id,
                        "region_name": region_name,
                        "area_pixels": area,
                        "file_path": str(file_path),
                    }
                    self.cache[str(file_path)] = result
                    results.append(result)

                    end_time = time.time()
                    processing_time = end_time - start_time
                    logger.info(
                        f"Completed processing {file_path.name} in {processing_time:.2f} seconds"
                    )

            except Exception as e:
                logger.error(
                    f"GPU batch processing failed: {str(e)}. Processing individually."
                )
                # Fall back to individual processing
                for file_path, animal_id, segment_id, region_name, _ in file_info_list:
                    result = self._process_single_file(file_path)
                    if result:
                        results.append(result)

        else:
            # Process individually
            for file_path, animal_id, segment_id, region_name, _ in file_info_list:
                result = self._process_single_file(file_path)
                if result:
                    results.append(result)

        return results

    def _get_file_batches(self) -> Iterator[List[Path]]:
        """Get batches of files to process.

        Returns:
            Iterator of batches of file paths
        """
        pkl_files = list(self.input_dir.glob("*.pkl"))
        for i in range(0, len(pkl_files), BATCH_SIZE):
            yield pkl_files[i : i + BATCH_SIZE]

    def analyze_directory(self) -> pd.DataFrame:
        """
        Analyze all ROI files in the input directory.

        Returns:
            DataFrame containing analysis results with columns:
            - animal_id: ID of the animal
            - segment_id: ID of the brain segment
            - region_name: Name of the brain region
            - area_pixels: Area of the ROI in pixels
            - file_path: Path to the source file
        """
        # Reset method counts
        self.method_counts = {"fast": 0, "gpu": 0, "sparse": 0}

        if not self.input_dir.exists():
            logger.error(f"Directory not found: {self.input_dir}")
            return pd.DataFrame()

        # Find all pickle files and group them by animal_id and segment_id
        pkl_files = list(self.input_dir.glob("*.pkl"))
        if not pkl_files:
            logger.warning(f"No .pkl files found in {self.input_dir}")
            return pd.DataFrame()

        # Group files by segment to ensure we process all files for each segment
        segment_files: Dict[str, List[Path]] = {}
        for file in pkl_files:
            try:
                animal_id, segment_id, _ = self._parse_filename(file.name)
                key = f"{animal_id}_{segment_id}"
                if key not in segment_files:
                    segment_files[key] = []
                segment_files[key].append(file)
            except Exception as e:
                logger.error(f"Error parsing filename {file.name}: {str(e)}")
                continue

        all_results = []
        total_files = len(pkl_files)

        with tqdm(total=total_files, desc="Processing ROI files") as pbar:
            # Process files segment by segment
            for segment_key, files in segment_files.items():
                for batch in [
                    files[i : i + BATCH_SIZE] for i in range(0, len(files), BATCH_SIZE)
                ]:
                    batch_results = self._process_batch(batch)
                    all_results.extend(batch_results)
                    pbar.update(len(batch))

        # Create DataFrame and optimize types
        df = pd.DataFrame(all_results)
        if not df.empty:
            self._optimize_dataframe(df)

        # Log computation method statistics
        logger.info("\nComputation method usage statistics:")
        logger.info(f"Fast method used: {self.method_counts['fast']} times")
        logger.info(f"GPU method used: {self.method_counts['gpu']} times")
        logger.info(f"Sparse method used: {self.method_counts['sparse']} times")

        return df

    @staticmethod
    def _optimize_dataframe(df: DataFrame) -> None:
        """Optimize DataFrame memory usage.

        Args:
            df: DataFrame to optimize
        """
        # Convert string columns to categorical
        for col in ["animal_id", "segment_id", "region_name"]:
            df[col] = df[col].astype("category")

        # Convert numeric columns to appropriate types
        df["area_pixels"] = df["area_pixels"].astype(np.int32)

    def get_summary_by_region(self, df: Optional[DataFrame] = None) -> DataFrame:
        """Get summary statistics grouped by region.

        Args:
            df: DataFrame containing analysis results

        Returns:
            DataFrame containing summary statistics grouped by region, aggregating across all segments
        """
        if df is None:
            df = self.analyze_directory()

        with tqdm(total=1, desc="Generating region summary") as pbar:
            # First group by region_name and segment_id to get segment-level stats
            segment_stats = (
                df.groupby(["region_name", "segment_id"], observed=True)["area_pixels"]
                .agg(["sum"])
                .reset_index()
            )

            # Then group by region_name to get region-level stats
            summary: DataFrame = (
                segment_stats.groupby("region_name", observed=True)
                .agg(
                    {
                        "segment_id": "count",  # Count of segments
                        "sum": [
                            "mean",
                            "std",
                            "min",
                            "max",
                            lambda x: x.quantile(0.25),
                            lambda x: x.quantile(0.75),
                            "sum",
                        ],  # Total area across all segments
                    }
                )
                .round(2)
            )

            # Flatten column names and rename
            summary.columns = [
                "segment_count",
                "mean_area",
                "std_area",
                "min_area",
                "max_area",
                "q25_area",
                "q75_area",
                "total_area",
            ]

            # Reset index to make region_name a column
            summary = summary.reset_index()
            pbar.update(1)

        return summary

    def get_summary_by_segment(self, df: Optional[DataFrame] = None) -> DataFrame:
        """Get summary statistics grouped by segment.

        Args:
            df: DataFrame containing analysis results

        Returns:
            DataFrame containing summary statistics grouped by segment, aggregating across all regions
        """
        if df is None:
            df = self.analyze_directory()

        with tqdm(total=1, desc="Generating segment summary") as pbar:
            # First group by segment_id and region_name to get region-level stats
            region_stats = (
                df.groupby(["segment_id", "region_name"], observed=True)["area_pixels"]
                .agg(["sum"])
                .reset_index()
            )

            # Then group by segment_id to get segment-level stats
            summary: DataFrame = (
                region_stats.groupby("segment_id", observed=True)
                .agg(
                    {
                        "region_name": "count",  # Count of regions
                        "sum": [
                            "mean",
                            "std",
                            "min",
                            "max",
                            lambda x: x.quantile(0.25),
                            lambda x: x.quantile(0.75),
                            "sum",
                        ],  # Total area across all regions
                    }
                )
                .round(2)
            )

            # Flatten column names and rename
            summary.columns = [
                "region_count",
                "mean_area",
                "std_area",
                "min_area",
                "max_area",
                "q25_area",
                "q75_area",
                "total_area",
            ]

            # Reset index to make segment_id a column
            summary = summary.reset_index()
            pbar.update(1)

        return summary

    def clear_cache(self) -> None:
        """Clear the internal cache of processed results."""
        self.cache.clear()
        self._parse_filename.cache_clear()

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the system and available GPU backends.

        Returns:
            Dictionary containing:
            - System information (OS, architecture)
            - GPU backend and device details
            - Computation method usage statistics
        """
        info: Dict[str, Any] = {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "gpu_backend": self.gpu_backend,
            "using_gpu": self.use_gpu,
            "computation_methods": self.method_counts,
        }

        # Add GPU device info if available
        if self.use_gpu and self.gpu_backend == "cupy":
            device: Any = self.gpu_module.cuda.runtime.getDeviceProperties(0)  # type: ignore
            info["gpu_device"] = device["name"].decode()
        elif self.use_gpu and self.gpu_backend == "mps":
            info["gpu_device"] = "Apple Silicon"

        return info

    def _is_valid_roi_file(self, file_path: Union[str, Path]) -> bool:
        """Check if a file matches the expected ROI file pattern.

        Args:
            file_path: Path to the file to check

        Returns:
            bool: True if the file matches the pattern, False otherwise
        """
        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        if not file_path.suffix == ".pkl":
            return False

        # Pattern: M###_s###_*.pkl (e.g., M762_s038_RSPagl.pkl)
        parts: List[str] = file_path.stem.split("_")
        if len(parts) < 3:
            return False

        return (
            parts[0].startswith("M")
            and parts[1].startswith("s")
            and parts[1][1:].isdigit()
            and parts[0][1:].isdigit()
        )

    def _find_roi_directories(self) -> List[Tuple[Path, str]]:
        """Recursively find directories containing valid ROI files.

        Returns:
            List of tuples containing (directory_path, animal_id)
        """
        roi_dirs: List[Tuple[Path, str]] = []

        for root, _, files in os.walk(self.input_dir):
            root_path = Path(root)
            pkl_files: List[Union[str, Path]] = [f for f in files if f.endswith(".pkl")]

            # Check if any files match our pattern
            valid_files: List[Union[str, Path]] = [
                f for f in pkl_files if self._is_valid_roi_file(root_path / f)
            ]

            if valid_files:
                # Get animal ID from the first valid file
                animal_id: str = (
                    valid_files[0].split("_")[0]
                    if isinstance(valid_files[0], str)
                    else valid_files[0].stem.split("_")[0]
                )
                roi_dirs.append((root_path, animal_id))

        return roi_dirs

    def analyze_all_directories(self) -> Dict[str, DataFrame]:
        """Analyze all directories containing ROI files.

        Returns:
            Dictionary mapping animal IDs to their analysis results
        """
        # Create log file in the root directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = Path(self.input_dir)
        # Find the root directory (where p3, p6, etc. are located)
        root_dir = input_path
        while root_dir.name.startswith(("p ", "p_", "p-", "m_", "m-")):
            root_dir = root_dir.parent
        log_file = root_dir / f"area_analysis_log_{timestamp}.txt"

        # Add file handler for the log file
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

        try:
            # Initialize total method counts
            total_method_counts = {"fast": 0, "gpu": 0, "sparse": 0}

            # Log system configuration
            logger.info("=== Analysis Configuration ===")
            sys_info = self.get_system_info()
            for key, value in sys_info.items():
                logger.info(f"{key}: {value}")
            logger.info(f"Input directory: {self.input_dir}")
            logger.info(f"Max workers: {self.max_workers}")
            logger.info(f"Using GPU: {self.use_gpu}")
            logger.info("=" * 50)

            results: Dict[str, DataFrame] = {}
            roi_dirs: List[Tuple[Path, str]] = self._find_roi_directories()

            if not roi_dirs:
                logger.warning(
                    f"No directories containing valid ROI files found in {self.input_dir}"
                )
                return results

            total_start_time = time.time()

            for dir_path, animal_id in tqdm(roi_dirs, desc="Processing directories"):
                logger.info(f"\nProcessing directory: {dir_path}")
                dir_start_time = time.time()

                # Create analysis directory in the parent directory of the ROI files
                analysis_dir: Path = (
                    dir_path.parent / f"{animal_id.lower()}_area_analysis"
                )
                analysis_dir.mkdir(exist_ok=True)

                # Temporarily set input_dir to current directory
                original_input_dir: Path = self.input_dir
                self.input_dir = dir_path

                try:
                    # Reset method counts for this directory
                    self.method_counts = {"fast": 0, "gpu": 0, "sparse": 0}

                    # Run analysis
                    df: pd.DataFrame = self.analyze_directory()
                    if not df.empty:
                        results[animal_id] = df

                        # Generate and save summaries
                        region_summary: DataFrame = self.get_summary_by_region(df)
                        segment_summary: DataFrame = self.get_summary_by_segment(df)

                        # Save results
                        df.to_csv(analysis_dir / "detailed_results.csv", index=False)
                        region_summary.to_csv(
                            analysis_dir / "region_summary.csv", index=False
                        )
                        segment_summary.to_csv(
                            analysis_dir / "segment_summary.csv", index=False
                        )

                        # Add this directory's method counts to the total
                        for method in total_method_counts:
                            total_method_counts[method] += self.method_counts[method]

                        # Log directory processing summary
                        dir_time = time.time() - dir_start_time
                        logger.info(
                            f"Directory {dir_path} processed in {dir_time:.2f}s"
                        )
                        logger.info(f"Files processed: {len(df)}")
                        logger.info(f"Results saved to: {analysis_dir}")

                except Exception as e:
                    logger.error(f"Error processing {dir_path}: {str(e)}")
                    continue
                finally:
                    # Restore original input_dir
                    self.input_dir = original_input_dir

            # Log final summary with total method counts
            total_time = time.time() - total_start_time
            logger.info("\n=== Analysis Summary ===")
            logger.info(f"Total processing time: {total_time:.2f}s")
            logger.info(f"Total directories processed: {len(roi_dirs)}")
            logger.info(f"Method usage counts: {total_method_counts}")
            logger.info("=" * 50)

            # Update the instance method counts with the totals
            self.method_counts = total_method_counts

            return results

        finally:
            # Remove the file handler
            logger.removeHandler(file_handler)
            file_handler.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze ROI areas from pickle files")
    parser.add_argument("input_dir", help="Directory containing ROI pickle files")
    parser.add_argument(
        "--use-gpu", action="store_true", help="Use GPU acceleration if available"
    )
    parser.add_argument("--workers", type=int, help="Number of worker threads")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level",
    )
    args: argparse.Namespace = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        # Initialize analyzer
        analyzer = ROIAreaAnalyzer(
            input_dir=args.input_dir, max_workers=args.workers, use_gpu=args.use_gpu
        )

        # Print system info
        print("\nSystem Information:")
        print("-" * 20)
        sys_info: Dict[str, Any] = analyzer.get_system_info()
        for key, value in sys_info.items():
            if key != "computation_methods":
                print(f"{key}: {value}")

        # Run analysis on all directories
        print("\nAnalyzing ROIs...")
        results: Dict[str, pd.DataFrame] = analyzer.analyze_all_directories()

        if not results:
            print("No ROIs found to analyze!")
            sys.exit(1)

        # Print computation statistics
        print("\nComputation Method Usage:")
        print("-" * 20)
        for method, count in analyzer.method_counts.items():
            print(f"{method.capitalize()} method: {count} ROIs")

        # Print basic statistics for each animal
        print("\nAnalysis Results:")
        print("-" * 20)
        for animal_id, df in results.items():
            print(f"\nAnimal: {animal_id}")
            print(f"Total ROIs analyzed: {len(df)}")
            print(f"Number of unique regions: {df['region_name'].nunique()}")
            print(f"Number of segments: {df['segment_id'].nunique()}")

        sys.exit(0)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
