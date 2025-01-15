"""Data structures for the dataset."""

from dataclasses import dataclass

# pylint: disable=no-member
from typing import List, Optional, Tuple

import datumaro
import numpy as np
import open3d as o3d

from libs.detection.detector_2d import Detection2D


@dataclass
class Frame:  # pylint: disable=missing-class-docstring
    image: np.ndarray
    id: str
    pointcloud: Optional[o3d.geometry.PointCloud] = None
    annotations: Optional[datumaro.components.annotation.Annotations] = None
    detections: Optional[List[Detection2D]] = None

    def file_name_from_id(self) -> str:
        """Return the file name from the frame ID."""
        return self.id.replace("/", "_") if self.id else ""

    def annotations_count(self) -> int:  # pragma: no cover
        """Return the number of annotations."""
        return len(self.annotations) if self.annotations else 0

    def detections_count(self) -> int:  # pragma: no cover
        """Return the number of detections."""
        return len(self.detections) if self.detections else 0


@dataclass
class BoundingBox:  # pylint: disable=missing-class-docstring, missing-function-docstring
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return int(self.x), int(self.y), int(self.w), int(self.h)

    def resize(self, factor: float) -> "BoundingBox":
        return BoundingBox(
            int(self.x * factor),
            int(self.y * factor),
            int(self.w * factor),
            int(self.h * factor),
        )
