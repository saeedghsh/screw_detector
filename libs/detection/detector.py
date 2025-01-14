"""This module provides classes for detecting screws in images."""

# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments

from typing import Any, List, Optional

import numpy as np


class Detection:
    """A simple container for detection results."""

    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        confidence: float = 1.0,
        label: Optional[int] = None,
    ):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence
        self.label = label

    def __repr__(self):
        return (
            f"Detection(label={self.label}, x={self.x}, y={self.y}, "
            f"width={self.width}, height={self.height}, confidence={self.confidence})"
        )

    def get_bbox(self) -> Any:
        """Return the bounding box as a tuple (x, y, w, h)."""
        return self.x, self.y, self.width, self.height

    @staticmethod
    def label_name_mapper(label: Optional[int]) -> str:
        """Map a label (index) to a human-readable name."""
        label_names = ["screw_head", "screw_hole"]
        return label_names[label] if label is not None else "NO_LABEL"


class Detector:
    """Abstract base class for any screw detection approach."""

    def detect(self, image: np.ndarray) -> List[Detection]:
        """return a list of detections (bounding boxes, etc.)."""
        raise NotImplementedError("detect() must be implemented by subclass")
