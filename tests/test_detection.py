# pylint: disable=missing-module-docstring, missing-function-docstring

from unittest.mock import patch

import numpy as np
import pytest

from libs.detection.detector_2d import Detection2D, Detector2D
from libs.detection.hough_circle_detector import HoughCircleDetector


def test_detection_initialization():
    detection = Detection2D(10, 20, 30, 40, confidence=0.95, label=1)
    assert detection.x == 10
    assert detection.y == 20
    assert detection.width == 30
    assert detection.height == 40
    assert detection.confidence == 0.95
    assert detection.label == 1


def test_detection_repr():
    detection = Detection2D(10, 20, 30, 40, confidence=0.95, label=1)
    repr_str = repr(detection)
    assert "Detection2D" in repr_str
    assert "label=1" in repr_str
    assert "x=10" in repr_str
    assert "y=20" in repr_str
    assert "width=30" in repr_str
    assert "height=40" in repr_str
    assert "confidence=0.95" in repr_str


def test_detection_get_bbox():
    detection = Detection2D(10, 20, 30, 40, confidence=0.95, label=1)
    bbox = detection.get_bbox()
    assert bbox == (10, 20, 30, 40)


@pytest.mark.parametrize(
    "label, expected_name",
    [(0, "screw_head"), (1, "screw_hole"), (99, "UNKNOWN")],
)
def test_detection_label_name_mapper(label, expected_name):
    assert Detection2D.label_name_mapper(label) == expected_name


def test_detector_abstract_method():
    class TestDetector(Detector2D):
        # pylint: disable=abstract-method, missing-class-docstring, too-few-public-methods
        pass

    test_detector = TestDetector({})

    image = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(NotImplementedError, match="detect\\(\\) must be implemented by subclass"):
        test_detector.detect(image)


def test_hough_circle_detector_initialization():
    config = {
        "dp": 1.2,
        "minDist": 20,
        "param1": 100,
        "param2": 30,
        "minRadius": 10,
        "maxRadius": 30,
        "gaussian_blur_kernel_size": 5,
        "object_labels": ["screw_head"],
    }
    detector = HoughCircleDetector(config)
    assert detector.configuration == config


@patch("cv2.HoughCircles", return_value=np.array([[[50, 50, 20]]]))
def test_hough_circle_detector_detect(mock_hough_circles):
    config = {
        "dp": 1.2,
        "minDist": 20,
        "param1": 100,
        "param2": 30,
        "minRadius": 10,
        "maxRadius": 30,
        "gaussian_blur_kernel_size": 5,
        "object_labels": ["screw_head"],
    }
    detector = HoughCircleDetector(config)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = detector.detect(image)
    # Ensure HoughCircles was called
    mock_hough_circles.assert_called_once()
    # Check the output detections
    assert len(detections) == 1
    detection = detections[0]
    assert isinstance(detection, Detection2D)
    assert detection.x == 30  # 50 - 20
    assert detection.y == 30  # 50 - 20
    assert detection.width == 40  # 2 * 20
    assert detection.height == 40  # 2 * 20
    assert detection.confidence == 1.0
    assert detection.label == 0
