# pylint: disable=missing-module-docstring, missing-function-docstring
import os
import sys
from unittest import mock

import pytest
from datumaro.components.annotation import Label

from entry_points.dataset_visualizer import main
from libs.dataset_management import Frame


@pytest.fixture
def dataset_manager_fixture():
    with mock.patch("entry_points.dataset_visualizer.DatasetManager") as mock_manager:
        dataset_manager_instance = mock_manager.return_value

        # Mock frame IDs and frame method
        dataset_manager_instance.frame_ids.keys.return_value = ["mock/frame/1"]

        mock_frame = mock.create_autospec(Frame, instance=True)
        mock_frame.id = "mock/frame/1"
        mock_frame.image = "image"
        mock_frame.pointcloud = "pointcloud"
        mock_frame.annotations = [Label(label=0), Label(label=1)]  # Mock annotations for labels
        dataset_manager_instance.frame.return_value = mock_frame

        # Mock label name mapper to return the correct label names
        dataset_manager_instance.label_name_mapper.side_effect = lambda idx: [
            "screw_head",
            "screw_hole",
        ][idx]

        yield mock_manager


@pytest.fixture
def visualizer_fixture():
    with mock.patch("entry_points.dataset_visualizer.Visualizer") as mock_visualizer:
        yield mock_visualizer


def test_main_executes_visualization(request: pytest.FixtureRequest):
    dataset_manager = request.getfixturevalue("dataset_manager_fixture")
    visualizer = request.getfixturevalue("visualizer_fixture")

    with mock.patch.object(sys, "argv", ["dataset_visualizer"]):
        result = main(sys.argv[1:])
        assert result == os.EX_OK

    dataset_manager.assert_called_once()
    visualizer.assert_called_once()
    visualizer_instance = visualizer.return_value
    visualizer_instance.visualize_frame.assert_called_once_with(
        dataset_manager.return_value.frame.return_value
    )


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
