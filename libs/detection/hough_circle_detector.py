"""
This module contains the HoughCircleDetector class, which detects screws by
finding circles of a certain radius range using OpenCV's HoughCircles method.
"""

# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=no-member

from typing import List

import cv2
import numpy as np

from libs.detection.detector import Detection, Detector


def hough_circle_detector_config() -> dict:
    """Return the configuration for the HoughCircleDetector."""
    # config numbers coming from some statistical analysis of the dataset
    expected_circle_radius = int(85 / 2)
    expected_circle_radius_min = expected_circle_radius - 15
    expected_circle_radius_max = expected_circle_radius + 20
    smallest_distance_bboxes_pairwise = 136.0588108135596
    smallest_distance_bboxes_pairwise = int(smallest_distance_bboxes_pairwise * 0.95)
    return {
        "dp": 1.2,  # default
        "min_dist": smallest_distance_bboxes_pairwise,
        "param1": 200,  # default
        "param2": 40,  # default
        "min_radius": expected_circle_radius_min,
        "max_radius": expected_circle_radius_max,
    }


class HoughCircleDetector(Detector):
    """
    Detect screws by finding circles of a certain radius range.
    Uses OpenCV's HoughCircles as an example.
    """

    def __init__(self, dp=1.2, min_dist=20, param1=100, param2=30, min_radius=10, max_radius=30):
        """
        :param dp: Inverse ratio of accumulator resolution to image resolution
        :param min_dist: Minimum distance between detected centers
        :param param1: Upper threshold for Canny edge detector
        :param param2: Threshold for center detection
        :param min_radius: Minimum circle radius
        :param max_radius: Maximum circle radius
        """
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius

    def detect(self, image: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = cv2.medianBlur(gray, 3)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        detections = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for x, y, r in circles:
                x_min = x - r
                y_min = y - r
                width = 2 * r
                height = 2 * r
                detections.append(
                    Detection(
                        x=x_min,
                        y=y_min,
                        width=width,
                        height=height,
                        confidence=1.0,
                        label=None,  # since its only detection, and not classification
                    )
                )
        return detections
