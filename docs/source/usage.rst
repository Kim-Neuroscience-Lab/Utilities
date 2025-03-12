Usage
=====

This section provides detailed usage instructions for the utilities in this package.

ROI Area Analysis
---------------

The ROI Area Analysis tool is designed to compute areas of regions of interest (ROIs) from pickle files across multiple directories.

Basic Usage
~~~~~~~~~~

To analyze ROIs in a directory::

    python roi_area_analysis.py /path/to/data

Command Line Arguments
~~~~~~~~~~~~~~~~~~

Required Arguments:
    * ``input_dir``: Directory containing ROI pickle files

Optional Arguments:
    * ``--use-gpu``: Enable GPU acceleration if available
    * ``--workers N``: Number of worker threads (default: number of CPU cores)
    * ``--log-level {DEBUG,INFO,WARNING,ERROR}``: Set logging level (default: INFO)
    * ``--output-dir PATH``: Custom output directory for analysis results
    * ``--batch-size N``: Batch size for processing ROIs (default: 32)
    * ``--skip-existing``: Skip analysis if output directory already exists

Examples
~~~~~~~~

1. Basic analysis with default settings::

    python roi_area_analysis.py /data/roi_files

2. Using GPU acceleration with 8 worker threads::

    python roi_area_analysis.py /data/roi_files --use-gpu --workers 8

3. Custom output directory and batch size::

    python roi_area_analysis.py /data/roi_files --output-dir /results --batch-size 64

4. Debug level logging and skip existing results::

    python roi_area_analysis.py /data/roi_files --log-level DEBUG --skip-existing

Output Structure
~~~~~~~~~~~~~

For each animal, the script creates a directory named ``{animal_id}_area_analysis`` containing:

* ``detailed_results.csv``: Complete analysis results for all ROIs
* ``region_summary.csv``: Summary statistics grouped by brain region
* ``segment_summary.csv``: Summary statistics grouped by segment

Example output directory structure::

    m729_area_analysis/
    ├── detailed_results.csv
    ├── region_summary.csv
    └── segment_summary.csv

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~

1. GPU Acceleration
    * Use ``--use-gpu`` when processing large ROIs
    * Automatically falls back to CPU if GPU processing fails
    * Supports both CUDA and MPS (Apple Silicon) backends

2. Batch Processing
    * Default batch size of 32 is optimal for most cases
    * Increase batch size for more GPU utilization
    * Decrease batch size if running into memory issues

3. Worker Threads
    * Defaults to number of CPU cores
    * Adjust based on system resources and other workloads

Error Handling
~~~~~~~~~~~~

The script includes comprehensive error handling:

* Validates input directory existence
* Handles invalid ROI files gracefully
* Provides detailed error messages in logs
* Supports interruption with Ctrl+C

Log Files
~~~~~~~~

Logs are saved with timestamps and include:

* System information
* Processing statistics
* Method usage counts
* Error messages and warnings
* Performance metrics

Example log output::

    === Analysis Configuration ===
    system: Darwin
    machine: arm64
    processor: arm
    python_version: 3.12.8
    numpy_version: 2.2.3
    gpu_backend: mps
    using_gpu: True

    Computation method usage:
    Fast method: 0 ROIs
    GPU method: 19 ROIs
    Sparse method: 0 ROIs 