# scripts/view_pickle.py
"""
Script for viewing and plotting the contents of a pickle file.

This script loads and displays the contents of a specified pickle file.
It provides functionality to:
1. Inspect serialized Python objects (showing head/tail for large data)
2. Plot numeric data in various formats (arrays, dataframes, lists, sparse coordinates)
3. Handle cases where original modules are not available
4. Convert paths between platforms

Example usage:
    python view_pickle.py path/to/file.pkl  # View contents
    python view_pickle.py path/to/file.pkl --plot  # Plot if data is plottable
    python view_pickle.py path/to/file.pkl --head 3 --tail 3  # Show first/last 3 items
"""

import argparse
from pathlib import Path, PureWindowsPath, PurePosixPath
import pickle
import sys
import numpy as np
import matplotlib

# Set the backend to TkAgg which works well across platforms
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from typing import Any, Union, Dict, Tuple
from collections.abc import Mapping


class SafeUnpickler(pickle.Unpickler):
    """Custom unpickler that handles missing modules gracefully and converts paths."""

    def find_class(self, module, name):
        """Override find_class to handle missing modules and path conversions.

        Args:
            module: The module name
            name: The class name

        Returns:
            A placeholder class, converted path, or the actual class if found
        """
        # Handle pathlib objects specially
        if module == "pathlib" and name in ("WindowsPath", "PosixPath", "Path"):
            # Convert all paths to the current platform's path type
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
            }

            return type(f"MissingClass_{name}", (), placeholder_dict)

    def persistent_load(self, pid):
        """Handle persistent IDs during unpickling."""
        return pid


def is_numeric(x: Any) -> bool:
    """Check if a value is numeric (including numpy numeric types).

    Args:
        x: Value to check

    Returns:
        bool: True if the value is numeric
    """
    return isinstance(x, (int, float, np.number, np.integer, np.floating))


def is_tuple_coord(x: Any) -> bool:
    """Check if a value is a valid coordinate tuple.

    Args:
        x: Value to check

    Returns:
        bool: True if the value is a tuple of two numeric values
    """
    if not isinstance(x, tuple):
        return False
    if len(x) != 2:
        return False
    return all(is_numeric(i) for i in x)


def is_plottable(data: Any) -> bool:
    """Check if the data can be plotted.

    Args:
        data: Data to check for plotting capability

    Returns:
        bool: True if data can be plotted, False otherwise
    """
    # Check for numpy arrays (including those stored as attributes)
    if isinstance(data, (np.ndarray, pd.DataFrame, pd.Series)):
        return True
    if isinstance(data, (list, tuple)) and all(is_numeric(x) for x in data):
        return True

    # Handle special case of unpickleable ROI objects with mask attribute
    if (
        not isinstance(data, (list, tuple, dict))
        and hasattr(data, "_module")
        and hasattr(data, "_class_name")
        and hasattr(data, "_attributes")
    ):

        if data._class_name == "ROI":
            # Check for common plottable attributes
            for attr_name in ["mask", "h2b_distribution", "cell_mask", "coordinates"]:
                if attr_name in data._attributes:
                    attr_value = data._attributes[attr_name]
                    if isinstance(attr_value, np.ndarray):
                        return True

    if isinstance(data, dict):
        # Handle nested ROI dictionary structure
        if "roi" in data and isinstance(data["roi"], dict):
            return all(
                is_tuple_coord(k) and is_numeric(v) for k, v in data["roi"].items()
            )
        # Handle regular dictionaries with numeric values
        if all(is_numeric(v) for v in data.values()):
            return True
        # Handle coordinate-based dictionaries
        if all(
            is_tuple_coord(k) and is_numeric(v) for k, v in data.items() if k != "name"
        ):
            return True
    return False


def plot_coordinate_dict(
    data: Dict[Tuple, Any], title: str = "Coordinate Data Visualization"
) -> None:
    """Plot a dictionary with coordinate tuples as keys.

    Args:
        data: Dictionary with (x,y) tuple keys and numeric values
        title: Title for the plot, defaults to "Coordinate Data Visualization"
    """
    # Extract coordinates and values
    coords = np.array(list(data.keys()))
    values = np.array([float(v) for v in data.values()])

    print(f"Creating plot for {len(coords)} coordinates with title: {title}")
    print(
        f"Coordinate range: x ({coords[:, 0].min()}-{coords[:, 0].max()}), y ({coords[:, 1].min()}-{coords[:, 1].max()})"
    )
    print(f"Value range: {values.min()}-{values.max()}")

    # Determine the grid size
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    grid_shape = (int(x_max - x_min + 1), int(y_max - y_min + 1))
    print(f"Grid shape: {grid_shape}")

    # Create a grid and fill it with the values
    grid = np.zeros(grid_shape)
    for (x, y), val in data.items():
        grid[int(float(x) - x_min), int(float(y) - y_min)] = float(val)

    plt.figure(figsize=(12, 8))
    im = plt.imshow(grid, cmap="viridis", aspect="auto")
    plt.colorbar(im, label="Value")
    plt.title(title)
    plt.xlabel("Y coordinate")
    plt.ylabel("X coordinate")
    plt.tight_layout()

    print(f"Plot created. Calling plt.show()...")
    plt.show(block=True)  # Ensure the plot blocks execution until closed
    print(
        "plt.show() completed. If you don't see a plot, check your matplotlib configuration."
    )


def plot_data(data: Any) -> None:
    """Plot the data using an appropriate visualization.

    Args:
        data: Data to plot (numpy array, pandas DataFrame/Series, list, dict, or coordinate dict)
    """
    print(f"Attempting to plot data of type: {type(data).__name__}")

    # Special case for ROI objects
    if (
        not isinstance(data, (list, tuple, dict))
        and hasattr(data, "_module")
        and hasattr(data, "_class_name")
        and hasattr(data, "_attributes")
        and data._class_name == "ROI"
    ):

        # Try to plot mask or other 2D arrays
        for attr_name in ["mask", "h2b_distribution", "cell_mask"]:
            if attr_name in data._attributes:
                attr_value = data._attributes[attr_name]
                if isinstance(attr_value, np.ndarray) and attr_value.ndim == 2:
                    plot_2d_array_with_histograms(
                        attr_value,
                        f"ROI {data._attributes.get('name', 'Unknown')} - {attr_name}",
                    )
                    return
                elif isinstance(attr_value, np.ndarray):
                    plt.figure(figsize=(10, 6))
                    if attr_value.ndim == 1:
                        plt.plot(attr_value)
                    else:
                        print(
                            f"Cannot plot {attr_value.ndim}-dimensional array directly"
                        )
                        return
                    roi_name = data._attributes.get("name", "Unknown")
                    plt.title(f"ROI {roi_name} - {attr_name}")
                    plt.tight_layout()
                    print("Displaying plot. Close the plot window to continue.")
                    plt.show(block=True)  # Ensure plot blocks execution until closed
                    return

    # Handle various data types for plotting
    if isinstance(data, dict):
        # Handle nested ROI dictionary structure
        if "roi" in data and isinstance(data["roi"], dict):
            print(f"Found nested ROI structure with {len(data['roi'])} coordinates")
            plot_coordinate_dict(
                data["roi"], title=f"ROI: {data.get('name', 'Unnamed')}"
            )
            return

        # Check if it's a coordinate-based dictionary
        if all(isinstance(k, tuple) and len(k) == 2 for k in data.keys()):
            print(f"Found coordinate dictionary with {len(data)} points")
            plot_title = str(data.get("name", "Coordinate Data"))
            plot_coordinate_dict(
                {k: v for k, v in data.items() if isinstance(k, tuple)},
                title=plot_title,
            )
            return

        # Regular dictionary plotting
        print(f"Plotting regular dictionary with {len(data)} items")
        plt.figure(figsize=(10, 6))
        plt.bar([str(k) for k in data.keys()], [float(v) for v in data.values()])
        plt.xticks(rotation=45)
        plt.title("Dictionary Values")
        plt.tight_layout()
        print("Displaying plot. Close the plot window to continue.")
        plt.show(block=True)  # Ensure plot blocks execution until closed
        return

    elif isinstance(data, pd.DataFrame):
        plt.figure(figsize=(10, 6))
        if data.select_dtypes(include=[np.number]).columns.size > 0:
            data.plot()
        else:
            print("DataFrame contains no numeric columns to plot")
            return

    elif isinstance(data, pd.Series):
        plt.figure(figsize=(10, 6))
        data.plot()

    elif isinstance(data, np.ndarray):
        if data.ndim == 2:
            plot_2d_array_with_histograms(data, f"2D Array of shape {data.shape}")
            return
        elif data.ndim == 1:
            plt.figure(figsize=(10, 6))
            plt.plot(data)
            plt.title(f"1D Array of shape {data.shape}")
            plt.ylabel("Value")
            plt.xlabel("Index")
            plt.grid(True, alpha=0.3)
        else:
            print(f"Cannot plot {data.ndim}-dimensional array")
            return

    elif isinstance(data, (list, tuple)):
        plt.figure(figsize=(10, 6))
        plt.plot(data)
        plt.title(f"{type(data).__name__} of length {len(data)}")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    print("Displaying plot. Close the plot window to continue.")
    plt.show(block=True)  # Ensure plot blocks execution until closed


def plot_2d_array_with_histograms(data: np.ndarray, title: str) -> None:
    """Plot a 2D array with optional histograms of nonzero pixels.

    Args:
        data: 2D numpy array to plot
        title: Title for the plot
    """
    args = parse_args()

    if args.hist and args.hist > 0:
        # Create a figure with a grid for the main image and histograms
        fig = plt.figure(figsize=(12, 10))
        # Create a grid layout
        gs = GridSpec(4, 4, figure=fig)

        # Main array plot in the center
        ax_main = fig.add_subplot(gs[1:4, 0:3])
        im = ax_main.imshow(data, cmap="viridis", aspect="auto")
        ax_main.set_title(title)

        # Get the extent of the main image for alignment
        extent = ax_main.get_xlim() + ax_main.get_ylim()

        # Get nonzero pixel positions
        nonzero_positions = np.where(data != 0)

        # X-axis histogram (row distribution) - aligned with y-axis of the main plot
        ax_x = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
        if len(nonzero_positions[0]) > 0:  # Check if there are any nonzero pixels
            # Create histogram bins that align with pixel positions
            bins = np.linspace(0, data.shape[0], min(args.hist, data.shape[0]))
            ax_x.hist(nonzero_positions[0], bins=bins, color="blue", alpha=0.7)
            ax_x.set_title("Row Distribution of Nonzero Pixels")
            ax_x.set_ylabel("Count")
        else:
            ax_x.text(0.5, 0.5, "No nonzero pixels", ha="center", va="center")
        ax_x.set_xticks([])  # Hide x ticks for the top histogram

        # Add colorbar
        cbar_ax = fig.add_subplot(gs[1:4, 3])
        plt.colorbar(im, cax=cbar_ax)

        # Y-axis histogram (column distribution) - aligned with x-axis of the main plot
        ax_y = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
        if len(nonzero_positions[1]) > 0:  # Check if there are any nonzero pixels
            # Create histogram bins that align with pixel positions
            bins = np.linspace(0, data.shape[1], min(args.hist, data.shape[1]))
            counts, edges = np.histogram(nonzero_positions[1], bins=bins)
            ax_y.barh(
                y=(edges[:-1] + edges[1:]) / 2,  # Center of each bin
                width=counts,
                height=edges[1] - edges[0],  # Width of each bin
                color="green",
                alpha=0.7,
            )
            ax_y.set_title("Column Distribution", rotation=-90, x=1.1, y=0.5)
            ax_y.set_xlabel("Count")
        ax_y.yaxis.set_ticks_position("right")
        ax_y.yaxis.set_label_position("right")

        # Remove axis from colorbar to prevent confusion
        cbar_ax.axis("off")

        # Adjust spacing for better alignment
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
    else:
        # Standard plot without histograms
        plt.figure(figsize=(10, 6))
        plt.imshow(data)
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()

    print("Displaying plot. Close the plot window to continue.")
    plt.show(block=True)  # Ensure plot blocks execution until closed


def format_dict_item(k: Any, v: Any, indent: int = 2) -> str:
    """Format a single dictionary item with proper indentation.

    Args:
        k: Dictionary key
        v: Dictionary value
        indent: Number of spaces for indentation

    Returns:
        str: Formatted key-value pair
    """
    spaces = " " * indent
    if isinstance(v, (dict, list, tuple, np.ndarray)):
        return f"{spaces}{k!r}: {type(v).__name__}({len(v)} items)"
    return f"{spaces}{k!r}: {v!r}"


def format_data_preview(data: Any, head: int = 5, tail: int = 5) -> str:
    """Format data for preview, showing head and tail for large sequences/mappings.

    Args:
        data: Data to format
        head: Number of items to show from start
        tail: Number of items to show from end

    Returns:
        str: Formatted string representation of the data
    """
    # Extract object attributes if available
    if (
        not isinstance(data, (list, tuple, dict))
        and hasattr(data, "_module")
        and hasattr(data, "_class_name")
    ):
        # Try to extract and display object attributes
        attrs = {}
        for attr in dir(data):
            # Skip private attributes, methods, and class attributes
            if attr.startswith("_") and attr not in ("_module", "_class_name"):
                continue
            if attr in ("_module", "_class_name"):
                continue
            try:
                value = getattr(data, attr)
                # Skip methods and other callables
                if callable(value):
                    continue
                attrs[attr] = value
            except (RecursionError, Exception):
                # Skip attributes that can't be accessed
                continue

        if attrs:
            attr_lines = [
                f"  {k}: {type(v).__name__}" for k, v in sorted(attrs.items())
            ]
            return (
                f"<{data._class_name} object from module {data._module} with {len(attrs)} attributes>\n"
                + "\n".join(attr_lines)
            )
        return f"<{data._class_name} object from module {data._module} with no accessible attributes>"

    if isinstance(data, (pd.DataFrame, pd.Series)):
        total_rows = len(data)
        if total_rows <= (head + tail):
            return str(data)
        return f"{data.head(head)}\n...\n[{total_rows - head - tail} rows omitted]\n...\n{data.tail(tail)}"

    elif isinstance(data, np.ndarray):
        if data.size <= (head + tail):
            return str(data)
        if data.ndim == 1:
            return f"Array of shape {data.shape}:\n{data[:head]}\n...\n[{data.size - head - tail} items omitted]\n...\n{data[-tail:]}"
        return f"Array of shape {data.shape} and dtype {data.dtype}"

    elif isinstance(data, (list, tuple)):
        if len(data) <= (head + tail):
            return str(data)
        return f"{type(data).__name__} of length {len(data)}:\n{data[:head]}\n...\n[{len(data) - head - tail} items omitted]\n...\n{data[-tail:]}"

    elif isinstance(data, dict):
        # Handle nested ROI dictionary structure
        if "roi" in data and isinstance(data["roi"], dict):
            name = data.get("name", "Unnamed ROI")
            roi_data = data["roi"]
            items = list(roi_data.items())
            total_items = len(items)

            if total_items <= (head + tail):
                formatted_items = [format_dict_item(k, v) for k, v in items]
                return f"ROI: {name!r} with {total_items} points:\n" + "\n".join(
                    formatted_items
                )

            head_items = [format_dict_item(k, v) for k, v in items[:head]]
            tail_items = [format_dict_item(k, v) for k, v in items[-tail:]]
            return (
                f"ROI: {name!r} with {total_items} points:\n"
                + "\n".join(head_items)
                + f"\n{' ' * 2}...\n{' ' * 2}[{total_items - head - tail} items omitted]\n{' ' * 2}...\n"
                + "\n".join(tail_items)
            )

        # Regular dictionary formatting
        items = list(data.items())
        total_items = len(items)

        if total_items <= (head + tail):
            formatted_items = [format_dict_item(k, v) for k, v in items]
            return f"Dictionary with {total_items} items:\n" + "\n".join(
                formatted_items
            )

        head_items = [format_dict_item(k, v) for k, v in items[:head]]
        tail_items = [format_dict_item(k, v) for k, v in items[-tail:]]
        return (
            f"Dictionary with {total_items} items:\n"
            + "\n".join(head_items)
            + f"\n{' ' * 2}...\n{' ' * 2}[{total_items - head - tail} items omitted]\n{' ' * 2}...\n"
            + "\n".join(tail_items)
        )

    return str(data)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Namespace containing the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="View and plot contents of a pickle file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pickle_file", type=str, help="Path to the pickle file to view")
    parser.add_argument(
        "--debug", action="store_true", help="Show detailed error information"
    )
    parser.add_argument("--plot", action="store_true", help="Plot the data if possible")
    parser.add_argument(
        "--head", type=int, default=5, help="Number of items to show from start"
    )
    parser.add_argument(
        "--tail", type=int, default=5, help="Number of items to show from end"
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save the plot to the specified file instead of displaying",
    )
    parser.add_argument(
        "--index",
        type=str,
        help="Access a specific dictionary item by key/index (use dot notation for nested access, e.g., 'key1.key2')",
    )
    parser.add_argument(
        "--hist",
        type=int,
        metavar="BINS",
        help="For 2D arrays, also plot histograms of nonzero pixel distributions along each axis",
    )
    return parser.parse_args()


def get_item_by_path(data: Any, path: str) -> Any:
    """Access a nested item in the data structure using a dot-separated path.

    Args:
        data: The data structure to navigate
        path: Dot-separated string indicating the path to the item (e.g., 'key1.key2')
            For dictionaries, numeric indices (e.g., 'dict.0') access the nth key-value pair
            For objects, attributes can be accessed directly (e.g., 'object.attribute')

    Returns:
        The item at the specified path

    Raises:
        KeyError: If the path cannot be resolved
    """
    if not path:
        return data

    # Use a maximum recursion limit to prevent stack overflow
    MAX_DEPTH = 20

    def get_nested_item(current, path_parts, depth=0):
        """Recursively access nested items with depth checking."""
        if depth > MAX_DEPTH:
            raise RecursionError(
                f"Maximum recursion depth ({MAX_DEPTH}) exceeded while accessing path"
            )

        if not path_parts:
            return current

        part = path_parts[0]
        remaining = path_parts[1:]

        # Handle attribute access
        if not isinstance(current, (dict, list, tuple)):
            try:
                if hasattr(current, part):
                    return get_nested_item(getattr(current, part), remaining, depth + 1)
            except Exception as e:
                raise KeyError(f"Error accessing attribute '{part}': {str(e)}")

        # Handle dictionary access
        if isinstance(current, dict):
            # Direct key access
            if part in current:
                return get_nested_item(current[part], remaining, depth + 1)

            # Numeric index access
            if part.isdigit():
                idx = int(part)
                keys = list(current.keys())
                if 0 <= idx < len(keys):
                    return get_nested_item(current[keys[idx]], remaining, depth + 1)
                raise KeyError(f"Index {idx} out of range (0-{len(current)-1})")

            raise KeyError(f"Key '{part}' not found in dictionary")

        # Handle list/tuple access
        if isinstance(current, (list, tuple)) and part.isdigit():
            idx = int(part)
            if 0 <= idx < len(current):
                return get_nested_item(current[idx], remaining, depth + 1)
            raise IndexError(f"Index {idx} out of range (0-{len(current)-1})")

        raise TypeError(
            f"Cannot access '{part}' in object of type {type(current).__name__}"
        )

    # Split path and process
    parts = path.split(".")
    return get_nested_item(data, parts)


def main():
    """Main function to load, display, and optionally plot pickle file contents."""
    args = parse_args()
    pickle_path = Path(args.pickle_file)

    if not pickle_path.exists():
        print(f"Error: File {pickle_path} does not exist")
        return

    try:
        # Increase recursion limit for complex pickle files
        import sys

        original_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(3000)  # Increase limit for loading complex pickles

        try:
            with open(pickle_path, "rb") as f:
                unpickler = SafeUnpickler(f)
                data = unpickler.load()
        finally:
            # Restore original recursion limit
            sys.setrecursionlimit(original_limit)

        # If index is provided, try to get the specified item
        if args.index:
            try:
                data = get_item_by_path(data, args.index)
                print(f"Viewing item at path: {args.index}")
            except (KeyError, IndexError, TypeError, RecursionError) as e:
                print(f"Error accessing item: {e}")
                return

        print(format_data_preview(data, args.head, args.tail))

        if args.plot:
            print(f"Plotting enabled. Checking if data is plottable...")
            if is_plottable(data):
                print(f"Data is plottable. Creating visualization...")

                # If save option is provided, save to file instead of displaying
                if args.save:
                    print(f"Saving plot to {args.save}...")
                    plt.ioff()  # Turn off interactive mode for saving
                    plot_data(data)
                    plt.savefig(args.save)
                    plt.close()
                    print(f"Plot saved to {args.save}")
                else:
                    # Display the plot interactively
                    plt.ion()  # Turn on interactive mode for displaying
                    plot_data(data)
            else:
                print("Data is not in a format that can be plotted")
                print(
                    "Supported formats: numpy arrays, pandas DataFrames/Series, numeric lists/dictionaries"
                )

    except Exception as e:
        if args.debug:
            import traceback

            print(f"Error loading pickle file:\n{traceback.format_exc()}")
        else:
            print(f"Error loading pickle file: {e}")
            print("Run with --debug flag for more detailed error information")


if __name__ == "__main__":
    main()
