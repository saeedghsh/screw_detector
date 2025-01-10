# pylint: disable=missing-module-docstring, missing-function-docstring
from unittest.mock import MagicMock, patch

import numpy as np
import open3d as o3d
import pytest
from datumaro.components.annotation import AnnotationType

from libs.visualization import Visualizer, VisualizerConfig


@pytest.fixture
def visualizer_fixture():
    config = VisualizerConfig(
        image_resize_factor=0.5,
        draw_annotation_as="bbox",
        visualize_2d=True,
        visualize_3d=False,
    )
    label_name_mapper = MagicMock(return_value="mock_label")
    return Visualizer(config, label_name_mapper)


def test_visualize_frame_with_mask_contour():
    config = VisualizerConfig(
        image_resize_factor=0.5,
        draw_annotation_as="mask_contour",
        visualize_2d=True,
        visualize_3d=False,
    )
    label_name_mapper = MagicMock(return_value="mock_label")
    visualizer = Visualizer(config, label_name_mapper)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    pointcloud = o3d.geometry.PointCloud()
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:150, 50:150] = 255
    annotation = MagicMock(label=0)
    annotation.type = AnnotationType.mask
    annotation.image = mask
    annotation.get_bbox.return_value = (50, 50, 100, 100)
    annotations = [annotation]
    with patch("cv2.imshow"), patch("cv2.waitKey"), patch("cv2.destroyAllWindows"):
        visualizer.visualize_frame(image, pointcloud, annotations)


def test_visualizer_resizes_image(request: pytest.FixtureRequest):
    visualizer = request.getfixturevalue("visualizer_fixture")
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    pointcloud = o3d.geometry.PointCloud()
    annotations = [MagicMock(label=0, get_bbox=MagicMock(return_value=(20, 20, 40, 40)))]
    annotations[0].type = AnnotationType.bbox
    with patch("cv2.imshow"), patch("cv2.waitKey"), patch("cv2.destroyAllWindows"):
        visualizer.visualize_frame(image, pointcloud, annotations)


def test_visualize_frame_no_visualization():
    config = VisualizerConfig(
        image_resize_factor=1.0,
        draw_annotation_as="bbox",
        visualize_2d=False,
        visualize_3d=False,
    )
    label_name_mapper = MagicMock(return_value="mock_label")
    visualizer = Visualizer(config, label_name_mapper)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    pointcloud = o3d.geometry.PointCloud()
    annotations = []
    visualizer.visualize_frame(image, pointcloud, annotations)  # Should not raise any errors


def test_visualizer_config():
    config = VisualizerConfig(
        image_resize_factor=0.75,
        draw_annotation_as="mask_contour",
        visualize_2d=True,
        visualize_3d=True,
    )
    assert config.image_resize_factor == 0.75
    assert config.draw_annotation_as == "mask_contour"
    assert config.visualize_2d is True
    assert config.visualize_3d is True


def test_visualize_frame_with_bbox():
    config = VisualizerConfig(
        image_resize_factor=1.0,
        draw_annotation_as="bbox",
        visualize_2d=True,
        visualize_3d=False,
    )
    label_name_mapper = MagicMock(return_value="mock_label")
    visualizer = Visualizer(config, label_name_mapper)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    pointcloud = o3d.geometry.PointCloud()
    annotation = MagicMock(label=0)
    annotation.type = AnnotationType.bbox
    annotation.get_bbox.return_value = (50, 50, 100, 100)
    annotations = [annotation]
    with patch("cv2.imshow"), patch("cv2.waitKey"), patch("cv2.destroyAllWindows"):
        visualizer.visualize_frame(image, pointcloud, annotations)


def test_visualize_3d():
    config = VisualizerConfig(
        image_resize_factor=1.0,
        draw_annotation_as="bbox",
        visualize_2d=False,
        visualize_3d=True,
    )
    label_name_mapper = MagicMock(return_value="mock_label")
    visualizer = Visualizer(config, label_name_mapper)
    pointcloud = o3d.geometry.PointCloud()
    with patch("open3d.visualization.draw_geometries"):
        visualizer.visualize_frame(np.zeros((200, 200, 3), dtype=np.uint8), pointcloud, [])
