"""This module provides classes for detecting screws in images."""

# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments

from typing import Any, List

import numpy as np


class Detection:
    """A simple container for detection results."""

    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        label: int,
        confidence: float = 1.0,
    ):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.confidence = confidence

    def __repr__(self):
        return (
            f"Detection(label={self.label}, x={self.x}, y={self.y}, "
            f"width={self.width}, height={self.height}, confidence={self.confidence})"
        )

    def get_bbox(self) -> Any:
        """Return the bounding box as a tuple (x, y, w, h)."""
        return self.x, self.y, self.width, self.height

    @staticmethod
    def label_name_mapper(label: int) -> str:
        """Map a label (index) to a human-readable name."""
        label_names = ["screw_head", "screw_hole"]
        if label >= len(label_names) or label < 0:
            return "UNKNOWN"
        return label_names[label]


class Detector:
    """Abstract base class for any screw detection approach."""

    def __init__(self, configuration: dict):
        self._configuration = configuration

    @property
    def configuration(self) -> dict:
        """Return the configuration dictionary."""
        return self._configuration

    def detect(self, image: np.ndarray) -> List[Detection]:
        """return a list of detections (bounding boxes, etc.)."""
        raise NotImplementedError("detect() must be implemented by subclass")
