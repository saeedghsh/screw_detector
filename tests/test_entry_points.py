# pylint: disable=missing-module-docstring, missing-function-docstring
import os
import sys
from unittest import mock

import pytest

from entry_points.entry_detector import main as entry_detector_main
from entry_points.entry_visualizer import main as entry_visualizer_main
from libs.dataset.data_structure import Frame


@pytest.fixture
def dataset_manager_fixture():
    with mock.patch("entry_points.entry_visualizer.DatasetManager") as mock_manager:
        dataset_manager_instance = mock_manager.return_value

        # Mock frame IDs and frame method
        dataset_manager_instance.frame_ids.keys.return_value = ["mock/frame/1"]
        dataset_manager_instance.frame_count.return_value = 1  # Mock frame count

        mock_frame = mock.create_autospec(Frame, instance=True)
        mock_frame.id = "mock/frame/1"
        mock_frame.image = "image"
        mock_frame.pointcloud = "pointcloud"

        # Create mock Annotation objects with get_bbox method
        mock_annotation_1 = mock.MagicMock()
        mock_annotation_1.get_bbox.return_value = (0, 0, 10, 10)
        mock_annotation_2 = mock.MagicMock()
        mock_annotation_2.get_bbox.return_value = (10, 10, 20, 20)

        mock_frame.annotations = [mock_annotation_1, mock_annotation_2]
        dataset_manager_instance.frame.return_value = mock_frame

        # Mock label name mapper to return the correct label names
        dataset_manager_instance.label_name_mapper.side_effect = lambda idx: [
            "screw_head",
            "screw_hole",
        ][idx]

        yield mock_manager


@pytest.fixture
def visualizer_fixture():
    with mock.patch("entry_points.entry_visualizer.Visualizer") as mock_visualizer:
        yield mock_visualizer


def test_main_executes_visualization(request: pytest.FixtureRequest):
    dataset_manager = request.getfixturevalue("dataset_manager_fixture")
    visualizer = request.getfixturevalue("visualizer_fixture")

    with mock.patch.object(sys, "argv", ["dataset_visualizer"]):
        result = entry_visualizer_main(sys.argv[1:])
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
    dataset_manager_instance.frame_count.return_value = 0  # Mock zero frame count

    with mock.patch.object(sys, "argv", ["dataset_visualizer"]):
        result = entry_visualizer_main(sys.argv[1:])
        assert result == os.EX_OK

    dataset_manager.assert_called_once()
    visualizer_instance = visualizer.return_value
    visualizer_instance.visualize_frame.assert_not_called()  # Ensure no frames were visualized


@mock.patch("entry_points.entry_detector.load_cached_split")
@mock.patch("entry_points.entry_detector.DatasetManager")
@mock.patch("entry_points.entry_detector.Visualizer")
@mock.patch("entry_points.entry_detector.HoughCircleDetector")
def test_entry_detector_main(
    mock_detector, mock_visualizer, mock_dataset_manager, mock_load_cached_split
):
    # pylint: disable=unused-argument
    mock_cached_split = mock.MagicMock()
    mock_cached_split.test_frame_ids = ["mock_frame_1"]  # Ensure non-empty test_frame_ids
    mock_load_cached_split.return_value = mock_cached_split

    mock_frame = mock.MagicMock()
    mock_dataset_manager.return_value.frame.return_value = mock_frame

    with mock.patch.object(
        sys, "argv", ["entry_detector", "dataset", "--cached-split-path", "/path/to/split.json"]
    ):
        result = entry_detector_main(sys.argv[1:])
        assert result == os.EX_OK

    mock_load_cached_split.assert_called_once_with("/path/to/split.json")
    mock_dataset_manager.assert_called_once()
    mock_visualizer.return_value.visualize_frame.assert_called_once_with(mock_frame)


def test_entry_detector_main_direct_mode():
    with (
        mock.patch.object(
            sys, "argv", ["entry_detector", "direct", "--input-path", "/path/to/images"]
        ),
        mock.patch("entry_points.entry_detector.load_images", return_value=[mock.MagicMock()]),
        mock.patch("entry_points.entry_detector.HoughCircleDetector"),
        mock.patch("entry_points.entry_detector.Visualizer"),
    ):

        result = entry_detector_main(sys.argv[1:])
        assert result == os.EX_OK
