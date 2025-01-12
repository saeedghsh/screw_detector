# pylint: disable=missing-module-docstring, missing-function-docstring
import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from datumaro.components.media import Image

from libs.dataset.manager import DatasetManager
from libs.dataset.utils import cache_split, dataset_stats, load_cached_split, split_train_test


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


def test_dataset_stats_interface_correctness(request: pytest.FixtureRequest):
    dataset_manager = request.getfixturevalue("dataset_manager_fixture")

    stats = dataset_stats(dataset_manager)
    assert "num_frames" in stats
    assert "per_battery_pack_count" in stats
    assert "num_labels" in stats
    assert "per_label_count" in stats
    assert "average_bbox_area" in stats
    assert "average_bbox_size" in stats
    assert "smallest_distance_bboxes_pairwise" in stats


def test_split_train_test_interface_correctness(request: pytest.FixtureRequest):
    dataset_manager = request.getfixturevalue("dataset_manager_fixture")

    with patch.object(dataset_manager, "frame", return_value=MagicMock(annotations=[])):
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

    mock_datetime.now.return_value.strftime.return_value = "20250113T123000"
    split_file = cache_split(dataset_manager, test_ratio=0.2)
    assert os.path.exists(split_file)
    with open(split_file, "r", encoding="utf-8") as file:
        split_data = json.load(file)
    assert "train_frame_ids" in split_data
    assert "test_frame_ids" in split_data
    assert split_data["split_ratio"] == 0.2
    assert split_data["timestamp"] == "20250113T123000"


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
