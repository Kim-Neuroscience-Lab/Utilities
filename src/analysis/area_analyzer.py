# src/analysis/area_analyzer.py
"""

1. Takes in a either a single pickle file or a nested directory of pickle files containing
  segmented region data for an animal or multiple animals
2. Recursively searches for pickle files in the input directory and builds a hierarchical
  mapping of the files in the following format:
    - Age group(s)
      - Animal(s)
       - Region(s)
        - Segment(s)
    - *Works for a single pickle file as well*
3. Instantiates a population dictionary of the format:
    {
        "age_group_id_0": {
            animal_object_0,
            animal_object_1,
            ...
            animal_object_n
        },
        "age_group_id_1": {
            animal_object_0,
            animal_object_1,
            ...
            animal_object_m
        },
        ...
        "age_group_id_q": {
            animal_object_0,
            animal_object_1,
            ...
            animal_object_p
        }
    }
    where each `Animal` holds a dictionary of `Region` objects which each hold a dictionary
      of `Segment` objects
4. The pickle files are asynchronously loaded into memory and processed in parallel either by
  CPU or GPU depending on the device specified

The areas of the segments are calculated and stored in its respective `Segment` object.
As files are processed, the results are written to a CSV file associated with each animal.

The population dictionary is then saved to a pickle file for later use.
"""

# Standard Library Imports
from typing import Dict, Any

# External Imports
from pydantic import BaseModel, Field

# Internal Imports
from src.utils.constants import MB_IN_BYTES
from src.core.models.animal import Animal
from src.utils.logging import get_logger

# Initialize Logger
logger = get_logger(__name__)


class AreaAnalyzer(BaseModel):
    class Config:
        batch_size: int = Field(
            default=1, ge=1, description="Number of files to process in a single batch"
        )
        buffer_size: int = Field(
            default=MB_IN_BYTES, ge=1, description="Buffer size for reading files"
        )
        device: str = Field(
            default="cpu",
            description="Device to use for processing.",
            examples=["cpu", "gpu", "mps"],
        )

        def __post_init__(self):
            logger.info(
                f"Initialized AreaAnalyzer with batch size {self.batch_size},"
                + f"buffer size {self.buffer_size},"
                + f"with device {self.device}"
            )

    config: Config = Config()
    animals: Dict[str, Animal] = Field(
        default_factory=dict, description="Dictionary of animals"
    )
