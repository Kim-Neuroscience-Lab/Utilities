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

# Neuroscience Data Utilities

A collection of utilities for processing neuroscience data, focusing on data imputation and analysis.

## Imputation Utilities

The package provides several imputation methods for handling missing data in neuroscience datasets.

### KNN Imputation

The KNN (K-Nearest Neighbors) imputation method replaces missing values with the weighted average of the K nearest neighbors in the feature space.

#### Command-line Usage

```bash
# Basic usage with default parameters
python -m src.scripts.impute_by_knn --input data/raw/mice_data_vsv_h2b.csv --output data/processed/mice_data_imputed.csv

# Advanced usage with custom parameters
python -m src.scripts.impute_by_knn \
  --input data/raw/mice_data_vsv_h2b.csv \
  --output data/processed/mice_data_imputed.csv \
  --neighbors 3 \
  --weights distance \
  --metric nan_euclidean \
  --class-column age_categorical \
  --separate-imputation \
  --verbose

# With visualization (auto-saves in the same directory as output)
python -m src.scripts.impute_by_knn \
  --input data/raw/mice_data_vsv_h2b.csv \
  --plot

# Save visualization to a specific file
python -m src.scripts.impute_by_knn \
  --input data/raw/mice_data_vsv_h2b.csv \
  --save-plot figures/imputed_data_visualization.png
```

#### Available Options

- `--input`, `-i`: Path to input CSV file (required)
- `--output`, `-o`: Path to output CSV file (default: input_imputed.csv)
- `--neighbors`, `-n`: Number of neighbors to use (default: 5)
- `--weights`, `-w`: Weight function used in prediction (choices: uniform, distance; default: uniform)
- `--metric`, `-m`: Distance metric for the tree (default: nan_euclidean)
- `--exclude`, `-e`: Columns to exclude from imputation (default: Age age_categorical Animal)
- `--include`: Columns to include in imputation (if specified, only these columns will be imputed)
- `--class-column`, `-c`: Column to use for class-based imputation (e.g., age_categorical)
- `--separate-imputation`, `-s`: Perform separate imputation for each class
- `--no-preprocess`: Skip preprocessing numeric columns
- `--random-state`: Random state for reproducibility (default: 137)
- `--verbose`, `-v`: Enable verbose output
- `--plot`, `-p`: Plot the imputed data as a heatmap with red outlines around imputed values
- `--save-plot`: Save the plot to the specified path instead of displaying it

### MICE Imputation

Multiple Imputation by Chained Equations (MICE) is an advanced imputation method that imputes missing values by modeling each feature with missing values as a function of other features.

```bash
# Basic usage
python -m scripts.impute_by_mice data/raw/mice_data_vsv_h2b.csv

# With visualization
python -m scripts.impute_by_mice data/raw/mice_data_vsv_h2b.csv --plot

# With custom settings
python -m scripts.impute_by_mice data/raw/mice_data_vsv_h2b.csv \
  --max-iterations 20 \
  --strategy median \
  --exclude-columns Age Animal \
  --verbose
```

### Example: Impute by Age Group

To impute missing values separately for each age group:

```bash
python -m src.scripts.impute_by_knn \
  --input data/raw/mice_data_vsv_h2b.csv \
  --output data/processed/mice_data_age_imputed.csv \
  --class-column age_categorical \
  --separate-imputation
```

### Example: Impute Only vsvQUANT Columns

To impute only columns containing "vsvQUANT" in their name:

```bash
python -m src.scripts.impute_by_knn \
  --input data/raw/mice_data_vsv_h2b.csv \
  --output data/processed/mice_data_vsv_imputed.csv \
  --include $(grep -o "vsvQUANT[^,]*" data/raw/mice_data_vsv_h2b.csv | head -1)
```

### Example: Visualize Imputed Data

Both imputation methods now support advanced visualization with red outlines highlighting imputed values:

```bash
# KNN imputation with visualization
python -m src.scripts.impute_by_knn \
  --input data/raw/mice_data_vsv_h2b.csv \
  --class-column age_categorical \
  --plot

# MICE imputation with visualization
python -m scripts.impute_by_mice data/raw/mice_data_vsv_h2b.csv --plot
```

## Python API Usage

You can also use the imputation classes directly in your Python code:

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.data.imputation import KNNImputerService, KNNImputerConfig

# Load data
data = pd.read_csv("data/raw/mice_data_vsv_h2b.csv")

# Store original missing value mask
missing_mask = data.isna()

# Configure KNN imputation
config = KNNImputerConfig(
    n_neighbors=3,
    weights="distance",
    exclude_columns=["Age", "age_categorical", "Animal"],
    class_column="age_categorical",
    separate_imputation=True
)

# Create imputer and process data
imputer = KNNImputerService(config)
imputed_data = imputer.impute(data)

# Get statistics about the imputation
stats = imputer.get_imputation_statistics()
print(f"Total missing values: {stats['total_missing_values']}")

# Save results
imputed_data.to_csv("data/processed/imputed_data.csv", index=False)

# Visualize the results with red outlines for imputed values
# Get numeric columns
numeric_cols = imputed_data.select_dtypes(include=["number"]).columns.tolist()
numeric_df = imputed_data[numeric_cols]
numeric_missing_mask = missing_mask[numeric_cols]

# Scale data for better visualization
scaled_df = (numeric_df - numeric_df.min(axis=0)) / (numeric_df.max(axis=0) - numeric_df.min(axis=0))

# Create heatmap
plt.figure(figsize=(12, 8))
plt.imshow(scaled_df.to_numpy(), aspect='auto', cmap='viridis')
plt.colorbar(label='Normalized Value')

# Highlight imputed values with red outlines
for i in range(len(numeric_df)):
    for j, col in enumerate(numeric_cols):
        if numeric_missing_mask.iloc[i, numeric_missing_mask.columns.get_loc(col)]:
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5),
                1, 1,
                fill=False,
                edgecolor='red',
                linewidth=1.5
            )
            plt.gca().add_patch(rect)

plt.xlabel('Features')
plt.ylabel('Samples')
plt.title('Imputed Data Visualization (Red outline = imputed value)')
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
plt.tight_layout()
plt.savefig("figures/imputed_data_with_highlights.png")
plt.show()
```
