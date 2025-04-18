"""ROI model for spatial analysis."""

# Standard Library Imports
from typing import Dict, Tuple, List, Optional, Any, Union, Callable
from pathlib import Path
import pickle

# Third Party Imports
import numpy as np
from pydantic import BaseModel, Field, validator, ConfigDict
import cv2
from scipy.ndimage import binary_dilation
from skimage.exposure import equalize_adapthist

# Internal Imports
from src.core.models.region import Region
from src.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)


class ROI(Region):
    """Region of Interest (ROI) model.

    This model extends the Region model to include functionality for
    handling intensity data and creating axon masks.

    Attributes:
        intensity: Dictionary mapping (y, x) coordinates to intensity values
        filename: Path to the file where this ROI was loaded from
        coverage: Coverage value for adjustment
        area: Total area of the ROI
        h2b_distribution: Distribution of H2B values
        mask: Normalized mask of the ROI
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    intensity: Optional[Dict[Tuple[int, int], int]] = Field(
        default=None,
        description="Dictionary mapping (y, x) coordinates to intensity values.",
    )
    filename: Optional[str] = Field(
        default=None, description="Path to the file where this ROI was loaded from."
    )
    coverage: Optional[float] = Field(
        default=None, description="Coverage value for adjustment."
    )
    h2b_distribution: Optional[np.ndarray] = Field(
        default=None, description="Distribution of H2B values."
    )
    mask: Optional[np.ndarray] = Field(
        default=None, description="Normalized mask of the ROI."
    )

    @property
    def verts(self) -> List[Tuple[int, int]]:
        """Get the vertices (coordinates) of the ROI.

        Returns:
            List[Tuple[int, int]]: List of (y, x) coordinates.
        """
        if self.intensity is None:
            return []
        return list(self.intensity.keys())

    @property
    def area(self) -> int:
        """Get the total area of the ROI.

        Returns:
            int: The total area of the ROI.
        """
        if self.intensity is None:
            return 0
        return len(self.intensity)

    def mean(self) -> float:
        """Calculate the mean intensity value.

        Returns:
            float: The mean intensity value.
        """
        if self.intensity is None:
            return 0.0

        values = []
        for i, j in self.intensity.keys():
            values.append(self.intensity[(i, j)])

        return float(np.mean(values))

    def adjust_to_coverage(self) -> None:
        """Multiplies all intensity values by the coverage value."""
        if self.coverage is None:
            logger.warning(
                f"Coverage value not set. Cannot adjust {self.filename} to coverage."
            )
            return

        if self.intensity is None:
            logger.warning(
                f"Intensity values not set. Cannot adjust {self.filename} to coverage."
            )
            return

        for i, j in self.intensity.keys():
            self.intensity[(i, j)] = int(
                np.floor(float(self.intensity[(i, j)]) * self.coverage)
            )

    def bounds(self) -> Tuple[Tuple[int, int, int, int], int, int]:
        """Get the bounds of the ROI.

        Returns:
            Tuple[Tuple[int, int, int, int], int, int]:
                A tuple containing (min_y, max_y, min_x, max_x), width, height.
        """
        if not self.verts:
            return (0, 0, 0, 0), 0, 0

        min_y = min(vert[0] for vert in self.verts)
        max_y = max(vert[0] for vert in self.verts)
        min_x = min(vert[1] for vert in self.verts)
        max_x = max(vert[1] for vert in self.verts)

        bounds = (min_y, max_y, min_x, max_x)
        width = max_y - min_y
        height = max_x - min_x

        return bounds, width, height

    def calculate_h2b_distribution(
        self, h2b_centers: Dict[Tuple[int, int], Any]
    ) -> None:
        """Calculate the distribution of H2B values in the ROI.

        Args:
            h2b_centers: Dictionary mapping coordinates to H2B values.
        """
        if self.intensity is None:
            logger.warning(
                "Cannot calculate H2B distribution: intensity values not set."
            )
            return

        points = set(self.intensity.keys())
        min_y = min(vert[0] for vert in self.verts)
        max_y = max(vert[0] for vert in self.verts)
        h2b_distribution = np.zeros(101)

        intersection = points.intersection(h2b_centers.keys())
        for j, i in intersection:
            # Calculate point in distribution
            relative_y = (j - min_y) / (max_y - min_y)
            h2b_distribution[int(np.floor(relative_y * 100))] += 1

        self.h2b_distribution = h2b_distribution

    def create_axon_mask(
        self,
        get_axon_mask_func: Callable[[np.ndarray], np.ndarray],
        output_dir: Optional[str] = None,
    ) -> None:
        """Identify axons in the ROI and create a mask.

        Args:
            get_axon_mask_func: Function that takes a numpy array and returns a binary mask
            output_dir: Directory to save output images (optional)
        """
        if self.intensity is None:
            logger.warning("Cannot create axon mask: intensity values not set.")
            return

        verts = self.verts
        # Find the bounding box for the ROI
        min_x = min(vert[1] for vert in verts)
        max_x = max(vert[1] for vert in verts)
        min_y = min(vert[0] for vert in verts)
        max_y = max(vert[0] for vert in verts)

        # Create an image of the ROI
        image = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
        mask = np.zeros_like(image)
        for vert in verts:
            y, x = vert[0] - min_y, vert[1] - min_x
            image[y, x] = self.intensity[vert]
            mask[y, x] = 1

        # Get the binary mask of the axons
        binary = get_axon_mask_func(image)

        # Clean up edges
        outside_points = np.argwhere(mask == 0)
        binary = self._correct_edges(outside_points, binary)

        # Get dimensions
        x_range = max_x - min_x
        y_range = max_y - min_y

        # Optional: Save visualization
        if output_dir is not None and self.filename is not None:
            # Create enhanced image for visualization
            image = (equalize_adapthist(image, clip_limit=0.0005) * 255).astype(
                np.uint8
            )
            colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            colored_image[binary == 1] = (0, 0, 255)

            # Ensure output directory exists
            stem = Path(self.filename).stem
            output_folder = Path(f"{output_dir}/{stem[:4]}/{self.roi}").resolve()
            output_folder.mkdir(parents=True, exist_ok=True)

            # Save image
            cv2.imwrite(f"{str(output_folder)}/{stem}.png", colored_image)

        # Normalize the coordinates to fit into a 101x101 grid
        normalized_mask = np.zeros((101, 101), dtype=np.uint8)
        for vert in verts:
            y, x = vert[0] - min_y, vert[1] - min_x
            norm_y, norm_x = int(y / y_range * 100), int(x / x_range * 100)
            if 0 < norm_y < 100 and 0 < norm_x < 100:
                normalized_mask[norm_y, norm_x] += 1 if (binary[y, x] > 0) else 0

        # Set the mask
        self.mask = normalized_mask

        # Optional: Clear intensity data to save memory
        # self.intensity = None

    @staticmethod
    def _correct_edges(outside_points, binary_image, iterations=5):
        """Clean up the binary image by removing edge artifacts.

        Args:
            outside_points: Points outside the ROI
            binary_image: Binary image to clean
            iterations: Number of dilation iterations

        Returns:
            np.ndarray: Cleaned binary image
        """
        # Create a mask from outside points
        mask = np.zeros_like(binary_image, dtype=np.uint8)
        mask[tuple(zip(*outside_points))] = 1

        # Dilate the mask
        dilated_mask = binary_dilation(
            mask, structure=np.ones((3, 3)), iterations=iterations
        )

        # Remove edge points from the binary image
        binary_image[dilated_mask == 1] = 0

        return binary_image

    @classmethod
    def from_file(cls, filename: str) -> "ROI":
        """Load an ROI from a file.

        Args:
            filename: Path to the file to load.

        Returns:
            ROI: The loaded ROI object.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be loaded as an ROI.
        """
        if not Path(filename).exists():
            raise FileNotFoundError(f"File {filename} does not exist.")

        try:
            with open(filename, "rb") as f:
                roi = pickle.load(f)
            if not isinstance(roi, ROI):
                raise ValueError(
                    f"File {filename} does not contain a valid ROI object."
                )
            return roi
        except Exception as e:
            raise ValueError(f"Failed to load ROI from {filename}: {str(e)}")
