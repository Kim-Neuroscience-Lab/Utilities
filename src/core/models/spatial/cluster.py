"""Cluster model for spatial analysis."""

# Standard Library Imports
from typing import List, Any, Optional
from uuid import uuid4, UUID

# Third Party Imports
import numpy as np
from pydantic import BaseModel, Field


class Cluster(BaseModel):
    """Represents a cluster of points in spatial analysis.

    This model is used to group points for spatial data analysis.

    Attributes:
        id: The unique identifier for the cluster
        points: The points belonging to this cluster
        label: The label for this cluster
        color: The color used to visualize this cluster
        name: The name of this cluster
    """

    id: UUID = Field(
        default_factory=uuid4, description="The unique identifier for the cluster."
    )
    points: List[Any] = Field(
        default_factory=list, description="The points belonging to this cluster."
    )
    label: Optional[int] = Field(
        default=None, description="The numeric label for this cluster."
    )
    color: Optional[str] = Field(
        default=None, description="The color used for visualizing this cluster."
    )
    name: Optional[str] = Field(default=None, description="The name of this cluster.")

    def __len__(self) -> int:
        """Returns the number of points in the cluster.

        Returns:
            int: The number of points in the cluster.
        """
        return len(self.points)

    def __str__(self) -> str:
        """Returns the string representation of the cluster.

        Returns:
            str: The string representation of the cluster.
        """
        return self.name or f"Cluster-{self.id}"

    def __repr__(self) -> str:
        """Returns the string representation of the cluster.

        Returns:
            str: The string representation of the cluster.
        """
        return self.__str__()
