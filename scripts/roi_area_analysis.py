# scripts/roi_area_analysis.py
"""
Script for analyzing ROI areas from pickle files in nested directory structures.
"""

import argparse
import logging
import multiprocessing as mp
import os
import sys
import psutil
import platform
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set, Iterator, Union
from dataclasses import dataclass

import pandas as pd
from tqdm.auto import tqdm

from src.analysis.roi_area_analyzer import ROIAreaAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ProgressState:
    """Class to track progress across different operations."""

    total_directories: int = 0
    scanned_directories: int = 0
    total_files: int = 0
    processed_files: int = 0
    current_animal: str = ""
    current_operation: str = ""
    error_count: int = 0

    def get_progress_stats(self) -> Dict[str, Any]:
        """Get current progress statistics."""
        return {
            "Scanning Progress": f"{self.scanned_directories}/{self.total_directories} directories",
            "Processing Progress": f"{self.processed_files}/{self.total_files} files",
            "Current Animal": self.current_animal,
            "Current Operation": self.current_operation,
            "Errors": self.error_count,
        }


class ProgressManager:
    """Manages progress bars and status display."""

    def __init__(self):
        self.state = ProgressState()
        self.main_pbar = None
        self.scan_pbar = None
        self.process_pbar = None

    def init_progress_bars(self):
        """Initialize all progress bars."""
        # Main progress bar for overall progress
        self.main_pbar = tqdm(
            total=100, desc="Overall Progress", position=0, leave=True
        )

        # Progress bar for directory scanning
        self.scan_pbar = tqdm(
            total=1,  # Will be updated when we know total
            desc="Scanning Directories",
            position=1,
            leave=True,
        )

        # Progress bar for file processing
        self.process_pbar = tqdm(
            total=1,  # Will be updated when we know total
            desc="Processing Files",
            position=2,
            leave=True,
        )

    def update_scan_progress(self, n: int = 1):
        """Update directory scanning progress."""
        self.state.scanned_directories += n
        if self.scan_pbar:
            self.scan_pbar.update(n)
            self._update_main_progress()

    def update_process_progress(self, n: int = 1):
        """Update file processing progress."""
        self.state.processed_files += n
        if self.process_pbar:
            self.process_pbar.update(n)
            self._update_main_progress()

    def set_total_directories(self, total: int):
        """Set total number of directories to scan."""
        self.state.total_directories = total
        if self.scan_pbar:
            self.scan_pbar.total = total
            self.scan_pbar.refresh()

    def set_total_files(self, total: int):
        """Set total number of files to process."""
        self.state.total_files = total
        if self.process_pbar:
            self.process_pbar.total = total
            self.process_pbar.refresh()

    def set_current_animal(self, animal_id: str):
        """Set current animal being processed."""
        self.state.current_animal = animal_id
        if self.process_pbar:
            self.process_pbar.set_description(f"Processing {animal_id}")

    def _update_main_progress(self):
        """Update overall progress."""
        if not self.main_pbar:
            return

        # Calculate overall progress (50% scanning, 50% processing)
        scan_progress = (
            self.state.scanned_directories / max(1, self.state.total_directories)
        ) * 50
        process_progress = (
            self.state.processed_files / max(1, self.state.total_files)
        ) * 50
        total_progress = int(scan_progress + process_progress)

        self.main_pbar.n = total_progress
        self.main_pbar.refresh()

    def close(self):
        """Close all progress bars."""
        if self.main_pbar:
            self.main_pbar.close()
        if self.scan_pbar:
            self.scan_pbar.close()
        if self.process_pbar:
            self.process_pbar.close()


# Global progress manager
progress = ProgressManager()


def check_gpu_availability() -> Tuple[bool, str]:
    """Check for GPU availability, with special handling for Apple Silicon.

    Returns:
        Tuple of (is_available, device_name)
    """
    try:
        if platform.system() == "Darwin" and platform.processor() == "arm":
            # Check for Apple Silicon MPS
            try:
                import torch

                if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                    # Test MPS availability with a small tensor operation
                    device = torch.device("mps")
                    test_tensor = torch.ones(1, device=device)
                    del test_tensor
                    return True, "Apple Silicon (MPS)"
            except Exception as e:
                logger.debug(f"MPS not available: {str(e)}")
                pass

        # Check for CUDA
        try:
            import torch

            if torch.cuda.is_available():
                return True, f"CUDA ({torch.cuda.get_device_name(0)})"
        except Exception as e:
            logger.debug(f"CUDA not available: {str(e)}")
            pass

    except ImportError:
        logger.warning("PyTorch not installed. GPU acceleration not available.")
    except Exception as e:
        logger.warning(f"Error checking GPU availability: {str(e)}")

    return False, "CPU"


def get_optimal_workers(total_cpus: int | None = None) -> Dict[str, int]:
    """Calculate optimal worker distribution based on available CPUs.

    This function determines the optimal number of workers for different tasks:
    - Directory scanning: 25% of CPUs (min 1)
    - Parallel directories: 25% of CPUs (min 1)
    - Workers per directory: 50% of CPUs (min 1)

    Args:
        total_cpus: Total number of CPUs to use. If None, uses all available CPUs.

    Returns:
        Dictionary containing optimal worker counts for each task
    """
    if total_cpus is None:
        total_cpus = mp.cpu_count()

    # Ensure we have at least 1 worker for each task
    scan_workers = max(1, total_cpus // 4)  # 25% for scanning
    parallel_dirs = max(1, total_cpus // 4)  # 25% for parallel directories
    workers_per_dir = max(1, total_cpus // 2)  # 50% for processing within directories

    return {
        "scan_workers": scan_workers,
        "parallel_dirs": parallel_dirs,
        "workers_per_dir": workers_per_dir,
    }


def parse_worker_arg(arg_value: Union[str, int], default_value: int) -> int:
    """Parse worker argument that can be either a number or 'max'/'auto'.

    Args:
        arg_value: The argument value (either int or string)
        default_value: Default value to use if arg_value is not 'max' or 'auto'

    Returns:
        Number of workers to use
    """
    if isinstance(arg_value, str):
        if arg_value.lower() in ("max", "auto"):
            return mp.cpu_count()
    return default_value if arg_value is None else int(arg_value)


def scan_directory(directory: Path) -> List[Tuple[Path, str]]:
    """Scan a single directory for ROI files.

    Args:
        directory: Directory to scan

    Returns:
        List of tuples containing (directory_path, animal_id) for valid ROI directories
    """
    roi_dirs: List[Tuple[Path, str]] = []

    try:
        # Create temporary analyzer just for file validation
        analyzer = ROIAreaAnalyzer(str(directory))

        # Check all files in this directory
        pkl_files = [f for f in directory.glob("*.pkl")]
        valid_files = [f for f in pkl_files if analyzer._is_valid_roi_file(f)]

        if valid_files:
            # Get animal ID from the first valid file
            animal_id = valid_files[0].stem.split("_")[0]
            roi_dirs.append((directory, animal_id))

    except Exception as e:
        logger.error(f"Error scanning directory {directory}: {e}")
        progress.state.error_count += 1

    progress.update_scan_progress()
    return roi_dirs


def find_roi_directories_parallel(
    root_dir: Path,
    max_workers: int,
) -> List[Tuple[Path, str]]:
    """Find all directories containing ROI files in parallel.

    Args:
        root_dir: Root directory to start search from
        max_workers: Number of worker threads

    Returns:
        List of tuples containing (directory_path, animal_id)
    """
    all_dirs = []

    # Get all subdirectories first
    for root, dirs, _ in os.walk(root_dir):
        root_path = Path(root)
        all_dirs.extend(root_path / d for d in dirs)
        all_dirs.append(root_path)  # Include root directory itself

    # Process directories in parallel
    roi_dirs: List[Tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_dir = {executor.submit(scan_directory, d): d for d in all_dirs}

        with tqdm(total=len(all_dirs), desc="Scanning directories") as pbar:
            for future in as_completed(future_to_dir):
                try:
                    results = future.result()
                    if results:
                        roi_dirs.extend(results)
                except Exception as e:
                    directory = future_to_dir[future]
                    logger.error(f"Error processing directory {directory}: {e}")
                pbar.update(1)

    return roi_dirs


def get_analyzed_files(analysis_dir: Path) -> Set[str]:
    """Get set of already analyzed files from detailed_results.csv.

    Args:
        analysis_dir: Path to analysis directory

    Returns:
        Set of file paths that have already been analyzed
    """
    results_file = analysis_dir / "detailed_results.csv"
    if not results_file.exists():
        return set()

    try:
        df = pd.read_csv(results_file)
        return set(df["file_path"].values)
    except Exception as e:
        logger.warning(f"Error reading existing results: {e}")
        return set()


def get_optimal_batch_size(sample_file: Path | None = None) -> int:
    """Calculate optimal batch size based on available system memory and file sizes.

    This function determines the batch size by:
    1. Checking available system memory
    2. Estimating ROI data size (from sample or default)
    3. Leaving buffer for other processes

    Args:
        sample_file: Optional path to a sample ROI file to estimate size

    Returns:
        Optimal batch size for processing
    """
    try:
        # Get available system memory (in bytes)
        mem = psutil.virtual_memory()
        available_memory = mem.available

        # Reserve 25% of available memory for other processes
        usable_memory = available_memory * 0.75

        # Estimate size per ROI
        if sample_file and sample_file.exists():
            # Use actual file size if sample provided
            roi_size = sample_file.stat().st_size
        else:
            # Default estimate: 1MB per ROI (conservative)
            roi_size = 1 * 1024 * 1024

        # Add overhead for DataFrame and processing (2x file size)
        roi_size_with_overhead = roi_size * 2

        # Calculate batch size
        optimal_batch_size = max(1, int(usable_memory / roi_size_with_overhead))

        # Cap at reasonable limits
        return min(max(optimal_batch_size, 10), 1000)

    except Exception as e:
        logger.warning(f"Error calculating optimal batch size: {e}")
        return 100  # Default fallback


def parse_batch_size_arg(
    arg_value: Union[str, int, None], sample_file: Path | None = None
) -> int:
    """Parse batch size argument that can be either a number or 'auto'.

    Args:
        arg_value: The argument value (either int or string)
        sample_file: Optional path to a sample ROI file

    Returns:
        Batch size to use
    """
    if isinstance(arg_value, str) and arg_value.lower() == "auto":
        return get_optimal_batch_size(sample_file)
    return (
        100 if arg_value is None else int(arg_value)
    )  # Default to 100 if not specified


def process_animal_directory(
    animal_dir: Path,
    animal_id: str,
    force_reanalyze: bool = False,
    use_gpu: bool = False,
    max_workers: int = mp.cpu_count() - 1,
    batch_size: Union[str, int] = "auto",
) -> Tuple[str, pd.DataFrame]:
    """Process a single animal's directory.

    Args:
        animal_dir: Path to animal's directory
        animal_id: Animal ID
        force_reanalyze: Whether to reanalyze already processed files
        use_gpu: Whether to use GPU acceleration
        max_workers: Number of worker threads
        batch_size: Number of files to process in each batch, or 'auto'

    Returns:
        Tuple of (animal_id, results DataFrame)
    """
    progress.set_current_animal(animal_id)

    # Create or get analysis directory
    analysis_dir = animal_dir.parent / f"{animal_id.lower()}_area_analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Get already analyzed files
    analyzed_files = set() if force_reanalyze else get_analyzed_files(analysis_dir)

    # Initialize analyzer for this directory
    analyzer = ROIAreaAnalyzer(
        str(animal_dir), max_workers=max_workers, use_gpu=use_gpu
    )

    # Get list of files to process
    pkl_files = [f for f in animal_dir.glob("*.pkl") if analyzer._is_valid_roi_file(f)]
    files_to_process = [f for f in pkl_files if str(f) not in analyzed_files]

    if not files_to_process:
        logger.info(f"No new files to process for {animal_id}")
        if analyzed_files:
            # Return existing results
            return animal_id, pd.read_csv(analysis_dir / "detailed_results.csv")
        return animal_id, pd.DataFrame()

    # Get sample file for batch size calculation if needed
    sample_file = files_to_process[0] if files_to_process else None
    actual_batch_size = parse_batch_size_arg(batch_size, sample_file)
    logger.info(f"Using batch size of {actual_batch_size} for {animal_id}")

    # Process files in batches to manage memory
    results_list = []

    with tqdm(total=len(files_to_process), desc=f"Processing {animal_id}") as pbar:
        for i in range(0, len(files_to_process), actual_batch_size):
            batch = files_to_process[i : i + actual_batch_size]

            # Process batch
            analyzer.input_dir = animal_dir  # Ensure correct directory
            batch_results = []
            for file in batch:
                try:
                    result = analyzer._process_single_file(file)
                    if result:
                        batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {file}: {e}")
                    progress.state.error_count += 1
                progress.update_process_progress()
                pbar.update(1)

            # Convert batch results to DataFrame
            if batch_results:
                batch_df = pd.DataFrame(batch_results)

                # Append to existing results file if it exists
                results_file = analysis_dir / "detailed_results.csv"
                if results_file.exists() and not force_reanalyze:
                    batch_df.to_csv(results_file, mode="a", header=False, index=False)
                else:
                    batch_df.to_csv(results_file, mode="w", header=True, index=False)

                results_list.append(batch_df)

                # Update summaries
                all_results = (
                    pd.concat(results_list) if len(results_list) > 1 else batch_df
                )
                region_summary = analyzer.get_summary_by_region(all_results)
                segment_summary = analyzer.get_summary_by_segment(all_results)

                region_summary.to_csv(analysis_dir / "region_summary.csv", index=False)
                segment_summary.to_csv(
                    analysis_dir / "segment_summary.csv", index=False
                )

            # Clear memory
            batch_results.clear()

    # Combine all results
    if results_list:
        final_results = pd.concat(results_list)
    elif analyzed_files:
        # Return existing results if no new files were processed
        final_results = pd.read_csv(analysis_dir / "detailed_results.csv")
    else:
        final_results = pd.DataFrame()

    return animal_id, final_results


def main() -> int:
    """Main function for ROI area analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze ROI areas from pickle files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Worker Configuration:
  For --workers, --parallel-dirs, and --scan-workers:
  - Use a number to specify exact worker count
  - Use 'max' or 'auto' to use all available CPUs
  - Default is an optimized distribution of available CPUs
  
  For --batch-size:
  - Use a number to specify exact batch size
  - Use 'auto' to automatically determine based on available memory
  - Default is 'auto'
  
Examples:
  # Use all available CPUs and automatic batch size
  %(prog)s /path/to/data --workers auto --parallel-dirs auto --scan-workers auto --batch-size auto
  
  # Use maximum parallelization with specific batch size
  %(prog)s /path/to/data --workers max --parallel-dirs max --scan-workers max --batch-size 200
  
  # Custom configuration
  %(prog)s /path/to/data --workers 4 --parallel-dirs 2 --scan-workers 2 --batch-size 100
""",
    )

    parser.add_argument(
        "input_dir",
        help="Root directory containing nested ROI pickle files (e.g., '01_first pass')",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", help="Use GPU acceleration if available"
    )
    parser.add_argument(
        "--batch-size",
        type=str,
        default="auto",
        help="Number of files to process in each batch. Use 'auto' for automatic sizing based on memory. (default: auto)",
    )

    # Get optimal worker distribution
    optimal_workers = get_optimal_workers()

    parser.add_argument(
        "--workers",
        type=str,
        default=optimal_workers["workers_per_dir"],
        help="Number of worker threads per directory. Use 'max' or 'auto' for all CPUs. (default: optimized)",
    )
    parser.add_argument(
        "--parallel-dirs",
        type=str,
        default=optimal_workers["parallel_dirs"],
        help="Number of directories to process in parallel. Use 'max' or 'auto' for all CPUs. (default: optimized)",
    )
    parser.add_argument(
        "--scan-workers",
        type=str,
        default=optimal_workers["scan_workers"],
        help="Number of worker threads for directory scanning. Use 'max' or 'auto' for all CPUs. (default: optimized)",
    )
    parser.add_argument(
        "--force-reanalyze",
        action="store_true",
        help="Force reanalysis of already processed files",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level",
    )
    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Parse worker arguments
    scan_workers = parse_worker_arg(args.scan_workers, optimal_workers["scan_workers"])
    parallel_dirs = parse_worker_arg(
        args.parallel_dirs, optimal_workers["parallel_dirs"]
    )
    workers_per_dir = parse_worker_arg(args.workers, optimal_workers["workers_per_dir"])

    # Log worker configuration
    logger.info("Worker configuration:")
    logger.info(f"- Directory scanning workers: {scan_workers}")
    logger.info(f"- Parallel directories: {parallel_dirs}")
    logger.info(f"- Workers per directory: {workers_per_dir}")
    logger.info(f"- Batch size: {args.batch_size}")

    try:
        # Initialize progress tracking
        progress.init_progress_bars()

        # Check GPU availability
        gpu_available, device_name = check_gpu_availability()
        if args.use_gpu and not gpu_available:
            logger.warning("GPU acceleration requested but no GPU found. Using CPU.")
        elif args.use_gpu:
            logger.info(f"Using GPU acceleration with {device_name}")

        # Find all directories containing ROI files in parallel
        print("\nScanning for ROI directories...")
        roi_dirs = find_roi_directories_parallel(
            Path(args.input_dir), max_workers=scan_workers
        )

        if not roi_dirs:
            print(
                f"No directories containing valid ROI files found in {args.input_dir}"
            )
            return 1

        print(f"\nFound {len(roi_dirs)} directories containing ROI files:")
        for dir_path, animal_id in roi_dirs:
            print(f"- {dir_path} (Animal: {animal_id})")

        # Process directories in parallel if requested
        results: Dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=parallel_dirs) as executor:
            future_to_dir = {
                executor.submit(
                    process_animal_directory,
                    dir_path,
                    animal_id,
                    args.force_reanalyze,
                    args.use_gpu,
                    workers_per_dir,
                    args.batch_size,
                ): (dir_path, animal_id)
                for dir_path, animal_id in roi_dirs
            }

            for future in tqdm(
                as_completed(future_to_dir),
                total=len(roi_dirs),
                desc="Processing directories",
            ):
                dir_path, animal_id = future_to_dir[future]
                try:
                    animal_id, df = future.result()
                    if not df.empty:
                        results[animal_id] = df
                except Exception as e:
                    logger.error(f"Error processing directory {dir_path}: {e}")

        if not results:
            print("No ROIs were successfully analyzed!")
            return 1

        # Print summary statistics for each animal
        print("\nAnalysis Results:")
        print("-" * 20)
        total_rois = 0
        for animal_id, df in results.items():
            print(f"\nAnimal: {animal_id}")
            print(f"Total ROIs analyzed: {len(df)}")
            print(f"Number of unique regions: {df['region_name'].nunique()}")
            print(f"Number of segments: {df['segment_id'].nunique()}")
            total_rois += len(df)

        print(f"\nTotal ROIs analyzed across all animals: {total_rois}")
        print(
            "\nResults have been saved in '*_area_analysis' directories next to each animal's directory."
        )

        return 0

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

    finally:
        progress.close()


if __name__ == "__main__":
    sys.exit(main())
