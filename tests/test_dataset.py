# pylint: disable=missing-module-docstring, missing-function-docstring
import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from datumaro.components.media import Image

from libs.dataset.manager import DatasetManager
from libs.dataset.utils import (
    cache_split,
    dataset_stats,
    load_cached_split,
    load_images,
    split_train_test,
)


@pytest.fixture
def dataset_manager_fixture():
    with patch("libs.dataset.manager.datumaro.Dataset.import_from") as mock_import:
        mock_dataset = MagicMock()
        mock_import.return_value = mock_dataset

        # Simulate multiple labels to handle multiple indices
        labels = [MagicMock(name=f"Label{i}") for i in range(3)]
        for i, label in enumerate(labels):
            label.name = f"label_{i}"
        mock_dataset.categories.return_value.get.return_value.items = labels

        # Create a mock annotation with valid label indices
        mock_annotation = MagicMock()
        mock_annotation.label = 0  # Set a valid label index
        mock_annotation.get_bbox.return_value = (0, 0, 10, 10)

        # Create a mock item with Image media type
        mock_item = MagicMock()
        mock_item.id = "battery_pack_1/zone_1/zone_1"
        mock_item.media = Image.from_file(path="mock_path/mock_image.png")
        mock_item.annotations = [mock_annotation]
        mock_dataset.__iter__.return_value = iter([mock_item])
        mock_dataset.__getitem__.side_effect = lambda idx: mock_item

        return DatasetManager()


@patch("libs.dataset.manager.pointcloud_path", return_value="mock_path/mock_image.png")
@patch("libs.dataset.manager.image_path", return_value="mock_path/mock_image.png")
@patch("os.path.isfile", return_value=True)
@patch("libs.dataset.manager.cv2.imread", return_value=np.zeros((100, 100, 3)))
@patch("libs.dataset.manager.o3d.io.read_point_cloud", return_value=MagicMock())
def test_dataset_manager_frame(
    mock_read_point_cloud,
    mock_imread,
    mock_isfile,
    mock_image_path,
    mock_pointcloud_path,
    request: pytest.FixtureRequest,
):
    # pylint: disable=unused-argument, too-many-arguments, too-many-positional-arguments
    dataset_manager = request.getfixturevalue("dataset_manager_fixture")
    frame_id = "battery_pack_1/zone_1/zone_1"
    frame = dataset_manager.frame(frame_id)
    # Check that frame attributes are correctly set
    assert frame.id == frame_id
    assert frame.image is not None
    assert frame.pointcloud is not None
    assert isinstance(frame.annotations, list)
    # Ensure private methods were called
    mock_imread.assert_called_once_with("mock_path/mock_image.png")
    mock_read_point_cloud.assert_called_once_with("mock_path/mock_image.png")


@patch("os.path.isfile", return_value=True)
@patch("libs.dataset.manager.cv2.imread", return_value=np.zeros((100, 100, 3)))
def test_dataset_stats_interface_correctness(
    mock_imread, mock_isfile, request: pytest.FixtureRequest
):
    # pylint: disable=unused-argument
    dataset_manager = request.getfixturevalue("dataset_manager_fixture")
    stats = dataset_stats(dataset_manager)
    assert "num_frames" in stats
    assert "per_battery_pack_count" in stats
    assert "num_labels" in stats
    assert "per_label_count" in stats
    assert "average_bbox_area" in stats
    assert "average_bbox_size" in stats
    assert "smallest_distance_bboxes_pairwise" in stats


@patch("os.path.isfile", return_value=True)
def test_split_train_test_interface_correctness(mock_isfile, request: pytest.FixtureRequest):
    # pylint: disable=unused-argument
    dataset_manager = request.getfixturevalue("dataset_manager_fixture")
    mock_annotations = MagicMock()
    mock_annotations.annotations = []  # Mock empty list of annotations
    with patch.object(dataset_manager, "_annotations", return_value=mock_annotations):
        result = split_train_test(dataset_manager, test_ratio=0.2)
        assert hasattr(result, "train_frame_ids")
        assert hasattr(result, "test_frame_ids")
        assert hasattr(result, "exact_annotation_test_ratio")
        assert isinstance(result.train_frame_ids, list)
        assert isinstance(result.test_frame_ids, list)
        assert isinstance(result.exact_annotation_test_ratio, float)


@patch("os.path.isfile", return_value=True)
@patch("cv2.imread", return_value=np.zeros((100, 100, 3)))
@patch("libs.dataset.utils.datetime")
def test_cache_split_creates_file(
    mock_datetime, mock_imread, mock_isfile, request: pytest.FixtureRequest
):
    # pylint: disable=unused-argument
    dataset_manager = request.getfixturevalue("dataset_manager_fixture")
    # Mock the return value of _annotations to have an annotations attribute
    mock_annotations = MagicMock()
    mock_annotations.annotations = []  # Mock empty annotations list
    with patch.object(dataset_manager, "_annotations", return_value=mock_annotations):
        mock_datetime.now.return_value.strftime.return_value = "20250113T123000"
        split_file = cache_split(dataset_manager, test_ratio=0.2)
    assert os.path.isfile(split_file)


def test_load_cached_split_interface_correctness(tmp_path):
    split_file = tmp_path / "split.json"
    split_data = {
        "train_frame_ids": ["frame_1", "frame_2"],
        "test_frame_ids": ["frame_3"],
        "split_ratio": 0.2,
        "timestamp": "20250113T123000",
    }
    with open(split_file, "w", encoding="utf-8") as file:
        json.dump(split_data, file)
    result = load_cached_split(str(split_file))
    assert result.train_frame_ids == split_data["train_frame_ids"]
    assert result.test_frame_ids == split_data["test_frame_ids"]
    assert result.split_ratio == split_data["split_ratio"]
    assert result.timestamp == split_data["timestamp"]


def test_load_images_single_file():
    with (
        patch("os.path.exists", return_value=True),
        patch("os.path.isfile", return_value=True),
        patch("cv2.imread", return_value=np.zeros((100, 100, 3))),
    ):
        images = load_images("mock_path/image.png")
        assert len(images) == 1
        assert isinstance(images[0], np.ndarray)


def test_load_images_directory():
    mock_file_list = ["mock_path/image1.png", "mock_path/image2.png"]
    with (
        patch("os.path.exists", return_value=True),
        patch("os.path.isfile", return_value=False),
        patch("os.path.isdir", return_value=True),
        patch("glob.glob", return_value=mock_file_list),
        patch("cv2.imread", return_value=np.zeros((100, 100, 3))),
    ):
        images = load_images("mock_path")
        assert len(images) == len(mock_file_list)
        assert all(isinstance(img, np.ndarray) for img in images)


def test_load_images_file_not_found():
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="Input path 'mock_path' does not exist."):
            load_images("mock_path")


def test_load_images_no_images_found():
    with (
        patch("os.path.exists", return_value=True),
        patch("os.path.isfile", return_value=False),
        patch("os.path.isdir", return_value=True),
        patch("glob.glob", return_value=[]),
    ):
        with pytest.raises(FileNotFoundError, match="No images found in 'mock_path'."):
            load_images("mock_path")


def test_load_images_invalid_file():
    with (
        patch("os.path.exists", return_value=True),
        patch("os.path.isfile", return_value=True),
        patch("cv2.imread", return_value=None),
    ):
        with pytest.raises(ValueError, match="Failed to load image from 'mock_path/image.png'."):
            load_images("mock_path/image.png")


def test_load_images_invalid_path():
    with (
        patch("os.path.exists", return_value=True),
        patch("os.path.isfile", return_value=False),
        patch("os.path.isdir", return_value=False),
    ):
        with pytest.raises(
            ValueError, match="Input path 'mock_path' is neither a file nor a directory."
        ):
            load_images("mock_path")
