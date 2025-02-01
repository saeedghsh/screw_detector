# pylint: disable=missing-module-docstring, missing-function-docstring

import json
import os
import sys
import tempfile
from unittest import mock

import pytest
from pytest import FixtureRequest

from entry_points.detector_2d import main as entry_detector_2d_main
from entry_points.evaluator import main as entry_evaluator_main
from entry_points.visualizer import main as entry_visualizer_main
from libs.dataset.data_structure import Frame


@pytest.fixture
def dataset_manager_fixture():
    with mock.patch("entry_points.visualizer.DatasetManager") as mock_manager:
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
    with mock.patch("entry_points.visualizer.Visualizer") as mock_visualizer:
        yield mock_visualizer


@mock.patch("entry_points.visualizer.load_config", return_value={"param1": "value1"})
def test_entry_visualizer_dataset_mode(mock_load_config, request: FixtureRequest):
    # pylint: disable=unused-argument
    dataset_manager = request.getfixturevalue("dataset_manager_fixture")
    visualizer = request.getfixturevalue("visualizer_fixture")

    with mock.patch.object(sys, "argv", ["dataset_visualizer", "dataset"]):
        result = entry_visualizer_main(sys.argv[1:])
        assert result == os.EX_OK

    dataset_manager.assert_called_once()
    visualizer.assert_called_once()
    visualizer_instance = visualizer.return_value
    visualizer_instance.visualize_frame.assert_called_once_with(
        dataset_manager.return_value.frame.return_value
    )


@mock.patch("entry_points.visualizer.load_config")
@mock.patch("entry_points.visualizer.Visualizer")
@mock.patch(
    "entry_points.visualizer.load_images",
    return_value={"frame1": mock.MagicMock(), "subdir_frame2": mock.MagicMock()},
)
@mock.patch(
    "entry_points.visualizer.load_pointclouds",
    return_value={"frame1": mock.MagicMock(), "subdir_frame2": mock.MagicMock()},
)
@mock.patch(
    "entry_points.visualizer.load_camera_transforms",
    return_value={"frame1": mock.MagicMock(), "subdir_frame2": mock.MagicMock()},
)
def test_entry_visualizer_direct_mode(
    mock_load_camera_transforms,
    mock_load_pointclouds,
    mock_load_images,
    mock_visualizer,
    mock_load_config,
):
    # pylint: disable=unused-argument, too-many-arguments
    with mock.patch.object(
        sys, "argv", ["entry_visualizer", "direct", "--input-path", "/path/to/data"]
    ):
        result = entry_visualizer_main(sys.argv[1:])
        assert result == os.EX_OK


@mock.patch("entry_points.detector_2d.load_config")
@mock.patch("entry_points.detector_2d.load_cached_split")
@mock.patch("entry_points.detector_2d.DatasetManager")
@mock.patch("entry_points.detector_2d.Visualizer")
@mock.patch("entry_points.detector_2d.HoughCircleDetector")
def test_entry_detector_2d_main(
    mock_detector, mock_visualizer, mock_dataset_manager, mock_load_cached_split, mock_load_config
):
    # pylint: disable=unused-argument
    mock_cached_split = mock.MagicMock()
    mock_cached_split.test_frame_ids = ["mock_frame_1"]  # Ensure non-empty test_frame_ids
    mock_load_cached_split.return_value = mock_cached_split

    mock_frame = mock.MagicMock()
    mock_dataset_manager.return_value.frame.return_value = mock_frame

    with mock.patch.object(
        sys, "argv", ["entry_detector_2d", "dataset", "--cached-split-path", "/path/to/split.json"]
    ):
        result = entry_detector_2d_main(sys.argv[1:])
        assert result == os.EX_OK
    mock_load_cached_split.assert_called_once_with("/path/to/split.json")
    mock_dataset_manager.assert_called_once()
    mock_visualizer.return_value.visualize_frame.assert_called_once_with(mock_frame)


@mock.patch("entry_points.detector_2d.load_config")
@mock.patch("entry_points.detector_2d.Visualizer")
@mock.patch("entry_points.detector_2d.HoughCircleDetector")
@mock.patch(
    "entry_points.detector_2d.load_images",
    return_value={"image1": mock.MagicMock(), "subdir_image2": mock.MagicMock()},
)
def test_entry_detector_2d_main_direct_mode(
    mock_load_images, mock_detector, mock_visualizer, mock_load_config
):
    # pylint: disable=unused-argument
    with mock.patch.object(
        sys, "argv", ["entry_detector_2d", "direct", "--input-path", "/path/to/images"]
    ):
        result = entry_detector_2d_main(sys.argv[1:])
        assert result == os.EX_OK


@mock.patch("entry_points.evaluator.load_config")
@mock.patch("entry_points.evaluator.load_cached_split")
@mock.patch("entry_points.evaluator.DatasetManager")
@mock.patch("entry_points.evaluator.HoughCircleDetector")
@mock.patch("entry_points.evaluator.Evaluator")
def test_entry_evaluator_main(
    mock_evaluator, mock_detector, mock_dataset_manager, mock_load_cached_split, mock_load_config
):
    mock_load_config.side_effect = [{"param1": "value1"}, {"param2": "value2"}]

    mock_cached_split = mock.MagicMock()
    mock_cached_split.test_frame_ids = ["mock/frame/1"]
    mock_load_cached_split.return_value = mock_cached_split

    mock_frame = mock.MagicMock()
    mock_dataset_manager.return_value.frame.return_value = mock_frame

    with tempfile.TemporaryDirectory() as temp_cache_dir:
        sample_split_path = f"{temp_cache_dir}/sample_split.json"
        with open(sample_split_path, "w", encoding="utf-8") as split_file:
            json.dump({"test_frame_ids": ["mock/frame/1"]}, split_file)

        with (
            mock.patch("entry_points.evaluator.CACHE_DIR", temp_cache_dir),
            mock.patch.object(sys, "argv", ["entry_evaluator"]),
        ):
            result = entry_evaluator_main(sys.argv[1:])
            assert result == os.EX_OK

    mock_dataset_manager.assert_called_once()
    mock_load_cached_split.assert_called_once_with(sample_split_path)
    mock_detector.assert_called_once()
    mock_evaluator.assert_called_once()

    evaluator_instance = mock_evaluator.return_value
    evaluator_instance.evaluate.assert_called_once_with(
        detector=mock_detector.return_value,
        dataset_manager=mock_dataset_manager.return_value,
        test_frames=mock_cached_split.test_frame_ids,
    )
