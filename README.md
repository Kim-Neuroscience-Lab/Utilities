# Kim Lab Utilities

A collection of computational tools and utilities used in the Kim Neuroscience Lab for data analysis and processing.

## Features

- **ROI Area Analysis**: Compute and analyze areas of regions of interest from pickle files
  - GPU acceleration support (CUDA/MPS)
  - Batch processing capabilities
  - Comprehensive logging and error handling
- **Core Utilities**: Common functions and tools used across different analysis pipelines
- **Data Management**: Tools for handling and organizing experimental data
- **I/O Operations**: Utilities for file handling and data format conversions

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Kim-Neuroscience-Lab/Utilities.git
cd Utilities
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### ROI Area Analysis

1. Basic usage:

```bash
python roi_area_analysis.py /path/to/data
```

2. With GPU acceleration:

```bash
python roi_area_analysis.py /path/to/data --use-gpu
```

3. Custom settings:

```bash
python roi_area_analysis.py /path/to/data --use-gpu --workers 8 --batch-size 64
```

For more examples and detailed usage instructions, see the [documentation](https://kim-neuroscience-lab.github.io/Utilities/).

## Documentation

Full documentation is available at: https://kim-neuroscience-lab.github.io/Utilities/

### Building Documentation Locally

1. Install documentation dependencies:

```bash
pip install sphinx sphinx-rtd-theme myst-parser
```

2. Build the HTML documentation:

```bash
cd docs
make html
```

3. View the documentation:

Option 1 - Using Python's built-in server:

```bash
cd build/html
python -m http.server 8000
```

Then open http://127.0.0.1:8000 in your web browser

Option 2 - Direct file access:

```bash
# macOS
open build/html/index.html

# Linux
xdg-open build/html/index.html

# Windows
start build/html/index.html
```

### Updating Documentation

The documentation source files are in `docs/source/`:

- `index.rst`: Main documentation page
- `installation.rst`: Installation guide
- `usage.rst`: Usage instructions
- `modules/*.rst`: Module-specific documentation

After making changes, rebuild the documentation using `make html` in the `docs` directory.

## Project Structure

```
Utilities/
├── src/
│   ├── analysis/       # Analysis tools and algorithms
│   ├── core/          # Core functionality and shared utilities
│   ├── data/          # Data management and processing
│   ├── io/            # Input/output operations
│   └── utils/         # General utility functions
├── docs/              # Documentation
│   ├── source/        # Documentation source files
│   └── build/html/    # Built documentation
├── tests/             # Test suite
└── scripts/           # Command-line tools and scripts
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Lab Website**: [Kim Neuroscience Lab](https://www.ejkimlab.com/)
- **GitHub**: [Kim-Neuroscience-Lab](https://github.com/Kim-Neuroscience-Lab)
