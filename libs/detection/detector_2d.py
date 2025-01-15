"""This module provides classes for detecting screws in images."""

# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments

from typing import List

import numpy as np

from libs.dataset.data_structure import Detection2D


class Detector2D:
    """Abstract base class for any screw detection approach."""

    def __init__(self, configuration: dict):
        self._configuration = configuration

    @property
    def configuration(self) -> dict:
        """Return the configuration dictionary."""
        return self._configuration

    def detect(self, image: np.ndarray) -> List[Detection2D]:
        """return a list of detections (bounding boxes, etc.)."""
        raise NotImplementedError("detect() must be implemented by subclass")
