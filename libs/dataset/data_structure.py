"""Data structures for the dataset."""

from dataclasses import dataclass

# pylint: disable=no-member
from typing import List, Optional, Tuple

import datumaro
import numpy as np
import open3d as o3d

from libs.detection.detector import Detection


@dataclass
class Frame:  # pylint: disable=missing-class-docstring
    image: np.ndarray
    id: Optional[str] = None
    pointcloud: Optional[o3d.geometry.PointCloud] = None
    annotations: Optional[datumaro.components.annotation.Annotations] = None
    detections: Optional[List[Detection]] = None

    def battery_pack(self) -> Optional[str]:
        """Return the battery pack number if ID is available."""
        return self.id.split("/")[0] if self.id else None

    def frame_name(self) -> Optional[str]:
        """Return the frame name if ID is available."""
        return self.id.split("/")[-1] if self.id else None


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
