# pylint: disable=missing-module-docstring, missing-function-docstring
from unittest import mock

import numpy as np
import open3d as o3d
import pytest
from datumaro.components.annotation import AnnotationType

from libs.dataset.data_structure import BoundingBox, Frame
from libs.visualization import Visualizer, _draw_bbox


@pytest.fixture
def visualizer_fixture():
    config = {
        "image_resize_factor": 0.5,
        "visualize_2d": True,
        "visualize_3d": False,
        "show_output": True,
        "save_output": True,
        "output_dir": "output",
    }
    label_name_mapper = mock.MagicMock(return_value="mock_label")
    return Visualizer(config, label_name_mapper)


def test_visualizer_resizes_image(request: pytest.FixtureRequest):
    visualizer = request.getfixturevalue("visualizer_fixture")
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    pointcloud = o3d.geometry.PointCloud()
    annotations = [mock.MagicMock(label=0, get_bbox=mock.MagicMock(return_value=(20, 20, 40, 40)))]
    annotations[0].type = AnnotationType.bbox
    frame = Frame(image, "x/y/z", pointcloud, annotations)
    with mock.patch("cv2.imshow"), mock.patch("cv2.waitKey"), mock.patch("cv2.destroyAllWindows"):
        visualizer.visualize_frame(frame)


def test_visualizer_visualize_frame_no_visualization():
    config = {
        "image_resize_factor": 1.0,
        "visualize_2d": False,
        "visualize_3d": False,
    }
    label_name_mapper = mock.MagicMock(return_value="mock_label")
    visualizer = Visualizer(config, label_name_mapper)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    pointcloud = o3d.geometry.PointCloud()
    annotations = []
    frame = Frame(image, "x/y/z", pointcloud, annotations)
    visualizer.visualize_frame(frame)


def test_visualizer_visualize_frame():
    config = {
        "image_resize_factor": 1.0,
        "visualize_2d": True,
        "visualize_3d": False,
        "show_output": True,
        "save_output": True,
        "output_dir": "output",
    }
    label_name_mapper = mock.MagicMock(return_value="mock_label")
    visualizer = Visualizer(
        config, annotation_label_mapper=label_name_mapper, detection_label_mapper=label_name_mapper
    )

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    pointcloud = o3d.geometry.PointCloud()

    # Mock annotations and detections
    annotation = mock.MagicMock(label=0)
    annotation.type = AnnotationType.bbox
    annotation.get_bbox.return_value = (50, 50, 100, 100)
    annotations = [annotation]

    detection = mock.MagicMock(label=1)
    detection.get_bbox.return_value = (30, 30, 40, 40)
    detections = [detection]

    frame = Frame(image, "x/y/z", pointcloud, annotations)

    with (
        mock.patch("libs.visualization._draw_bbox") as mock_draw_bbox,
        mock.patch("libs.visualization._write_label_with_prefix") as mock_write_label,
        mock.patch("cv2.imshow"),
        mock.patch("cv2.waitKey"),
        mock.patch("cv2.destroyAllWindows"),
    ):
        frame.detections = detections
        visualizer.visualize_frame(frame)
        # Ensure _draw_bbox was called for both annotations and detections
        assert mock_draw_bbox.call_count == 2
        # Ensure _write_label_with_prefix was called with correct prefixes
        mock_write_label.assert_any_call(mock.ANY, "A", "mock_label", mock.ANY, mock.ANY)
        mock_write_label.assert_any_call(mock.ANY, "D", "mock_label", mock.ANY, mock.ANY)


def test_visualizer_visualize_3d():
    config = {
        "image_resize_factor": 1.0,
        "visualize_2d": False,
        "visualize_3d": True,
        "show_output": True,
        "save_output": True,
        "output_dir": "output",
    }
    label_name_mapper = mock.MagicMock(return_value="mock_label")
    visualizer = Visualizer(config, label_name_mapper)
    pointcloud = o3d.geometry.PointCloud()
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    annotations = []
    frame = Frame(image, "x/y/z", pointcloud, annotations)
    with mock.patch("open3d.visualization.draw_geometries"):
        visualizer.visualize_frame(frame)


def test_bounding_box_as_tuple():
    bbox = BoundingBox(10, 20, 30, 40)
    assert bbox.as_tuple() == (10, 20, 30, 40)


@pytest.mark.parametrize(
    "factor, expected",
    [
        (0.5, (5, 10, 15, 20)),
        (2.0, (20, 40, 60, 80)),
        (1.0, (10, 20, 30, 40)),
    ],
)
def test_bounding_box_resize(factor, expected):
    bbox = BoundingBox(10, 20, 30, 40)
    resized_bbox = bbox.resize(factor)
    assert resized_bbox.as_tuple() == expected


@pytest.mark.parametrize("filled", [True, False])
def test_draw_bbox(filled):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = BoundingBox(10, 20, 30, 40)
    color = (255, 0, 0)
    _draw_bbox(image, bbox, color, filled)
    assert image.shape == (100, 100, 3)  # Ensure the image shape is unchanged
