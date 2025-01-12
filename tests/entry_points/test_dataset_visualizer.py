# pylint: disable=missing-module-docstring, missing-function-docstring
import os
import sys
from unittest import mock

import pytest

from entry_points.dataset_visualizer import main


@pytest.fixture
def dataset_manager_fixture():
    with mock.patch("entry_points.dataset_visualizer.DatasetManager") as mock_manager:
        yield mock_manager


@pytest.fixture
def visualizer_fixture():
    with mock.patch("entry_points.dataset_visualizer.Visualizer") as mock_visualizer:
        yield mock_visualizer


def test_main_executes_visualization(request: pytest.FixtureRequest):
    dataset_manager = request.getfixturevalue("dataset_manager_fixture")
    visualizer = request.getfixturevalue("visualizer_fixture")

    dataset_manager_instance = dataset_manager.return_value
    dataset_manager_instance.frame_ids.keys.return_value = [1]
    dataset_manager_instance.image.return_value = "image"
    dataset_manager_instance.pointcloud.return_value = "pointcloud"
    dataset_manager_instance.frame_annotations.return_value = "annotations"
    dataset_manager_instance.frame.return_value = "frame"

    with mock.patch.object(sys, "argv", ["dataset_visualizer"]):
        result = main(sys.argv[1:])
        assert result == os.EX_OK

    dataset_manager.assert_called_once()
    visualizer.assert_called_once()
    visualizer_instance = visualizer.return_value
    visualizer_instance.visualize_frame.assert_called_once_with("frame")


def test_main_handles_empty_frame_ids(request: pytest.FixtureRequest):
    """Test the edge case where frame_ids is empty to ensure 100% coverage."""
    dataset_manager = request.getfixturevalue("dataset_manager_fixture")
    visualizer = request.getfixturevalue("visualizer_fixture")

    dataset_manager_instance = dataset_manager.return_value
    dataset_manager_instance.frame_ids.keys.return_value = []  # Simulate empty frame_ids

    with mock.patch.object(sys, "argv", ["dataset_visualizer"]):
        result = main(sys.argv[1:])
        assert result == os.EX_OK

    dataset_manager.assert_called_once()
    visualizer_instance = visualizer.return_value
    visualizer_instance.visualize_frame.assert_not_called()  # Ensure no frames were visualized
