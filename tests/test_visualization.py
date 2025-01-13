# pylint: disable=missing-module-docstring, missing-function-docstring
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import open3d as o3d
import pytest
from datumaro.components.annotation import AnnotationType

from libs.dataset.manager import Frame
from libs.visualization.dataset import Visualizer


@pytest.fixture
def visualizer_fixture():
    config = SimpleNamespace(
        image_resize_factor=0.5,
        visualize_2d=True,
        visualize_3d=False,
        show_output=True,
        save_output=True,
        output_dir="output",
    )
    label_name_mapper = MagicMock(return_value="mock_label")
    return Visualizer(config, label_name_mapper)


def test_visualizer_resizes_image(request: pytest.FixtureRequest):
    visualizer = request.getfixturevalue("visualizer_fixture")
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    pointcloud = o3d.geometry.PointCloud()
    annotations = [MagicMock(label=0, get_bbox=MagicMock(return_value=(20, 20, 40, 40)))]
    annotations[0].type = AnnotationType.bbox
    frame = Frame("x/y/z", image, pointcloud, annotations)
    with patch("cv2.imshow"), patch("cv2.waitKey"), patch("cv2.destroyAllWindows"):
        visualizer.visualize_frame(frame)


def test_visualize_frame_no_visualization():
    config = SimpleNamespace(
        image_resize_factor=1.0,
        visualize_2d=False,
        visualize_3d=False,
    )
    label_name_mapper = MagicMock(return_value="mock_label")
    visualizer = Visualizer(config, label_name_mapper)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    pointcloud = o3d.geometry.PointCloud()
    annotations = []
    frame = Frame("x/y/z", image, pointcloud, annotations)
    visualizer.visualize_frame(frame)  # Should not raise any errors


def test_visualize_frame_with_bbox():
    config = SimpleNamespace(
        image_resize_factor=1.0,
        visualize_2d=True,
        visualize_3d=False,
        show_output=True,
        save_output=True,
        output_dir="output",
    )
    label_name_mapper = MagicMock(return_value="mock_label")
    visualizer = Visualizer(config, label_name_mapper)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    pointcloud = o3d.geometry.PointCloud()
    annotation = MagicMock(label=0)
    annotation.type = AnnotationType.bbox
    annotation.get_bbox.return_value = (50, 50, 100, 100)
    annotations = [annotation]
    frame = Frame("x/y/z", image, pointcloud, annotations)
    with patch("cv2.imshow"), patch("cv2.waitKey"), patch("cv2.destroyAllWindows"):
        visualizer.visualize_frame(frame)


def test_visualize_3d():
    config = SimpleNamespace(
        image_resize_factor=1.0,
        visualize_2d=False,
        visualize_3d=True,
        show_output=True,
        save_output=True,
        output_dir="output",
    )
    label_name_mapper = MagicMock(return_value="mock_label")
    visualizer = Visualizer(config, label_name_mapper)
    pointcloud = o3d.geometry.PointCloud()
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    annotations = []
    frame = Frame("x/y/z", image, pointcloud, annotations)
    with patch("open3d.visualization.draw_geometries"):
        visualizer.visualize_frame(frame)
