# Kim Lab Utilities

A collection of computational tools and utilities used in the Kim Neuroscience Lab for data analysis and processing.

[![Documentation Status](https://github.com/Kim-Neuroscience-Lab/Utilities/workflows/Documentation/badge.svg)](https://kim-neuroscience-lab.github.io/Utilities/)

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

To build the documentation locally:

```bash
cd docs
make html
```

Then open `docs/build/html/index.html` in your web browser.

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
├── tests/             # Test suite
└── scripts/           # Command-line tools and scripts
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Lab Website**: [Kim Lab - UCSC](https://www.ejkimlab.com/)
- **GitHub**: [Kim-Neuroscience-Lab](https://github.com/Kim-Neuroscience-Lab)

## Acknowledgments

- UCSC Neuroscience Department
- Contributors and lab members
- Open source community
