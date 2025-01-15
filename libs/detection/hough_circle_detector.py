"""This module contains the HoughCircleDetector class, which detects screws by
finding circles of a certain radius range using OpenCV's HoughCircles method."""

from typing import List

import cv2
import numpy as np

from libs.detection.detector_2d import Detection2D, Detector2D

# pylint: disable=no-member


class HoughCircleDetector(Detector2D):
    """Detect circles of a certain radius range."""

    def _hough_circles_args(self) -> dict:
        return {
            k: v
            for k, v in self._configuration.items()
            if k in {"dp", "minDist", "param1", "param2", "minRadius", "maxRadius"}
        }

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        kernel_size = self._configuration["gaussian_blur_kernel_size"]
        return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    def detect(self, image: np.ndarray) -> List[Detection2D]:
        """Detect circles in the image using HoughCircles."""
        circles = cv2.HoughCircles(
            self._preprocess_image(image), cv2.HOUGH_GRADIENT, **self._hough_circles_args()
        )
        detections = []
        object_labels = self.configuration["object_labels"]
        if object_labels != ["screw_head"]:  # pragma: no cover
            raise ValueError("HoughCircleDetector can only assume one class: screw_head")

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for x, y, r in circles:
                x_min = x - r
                y_min = y - r
                width = 2 * r
                height = 2 * r
                detections.append(
                    Detection2D(
                        x=x_min,
                        y=y_min,
                        width=width,
                        height=height,
                        confidence=1.0,
                        label=0,
                    )
                )
        return detections
