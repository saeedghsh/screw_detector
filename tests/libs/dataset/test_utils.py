# pylint: disable=missing-module-docstring, missing-function-docstring

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from libs.dataset.manager import DatasetManager
from libs.dataset.utils import cache_split, dataset_stats, load_cached_split, split_train_test


@pytest.fixture
def mock_dataset_manager() -> DatasetManager:
    # pylint: disable=protected-access
    manager = MagicMock(spec=DatasetManager)
    manager.frame_ids = {
        f"battery_pack_{i}/frame_{j}/frame_{j}": j for i in range(1, 3) for j in range(10)
    }
    manager._dataset = [MagicMock() for _ in range(20)]
    manager._annotations = MagicMock(return_value=[MagicMock(get_bbox=lambda: (0, 0, 10, 10))])
    manager.label_name_mapper = MagicMock(side_effect=lambda idx: f"label_{idx}")
    manager._label_categories = MagicMock(items=[MagicMock(name=f"label_{i}") for i in range(5)])
    return manager


def test_dataset_stats_interface_correctness(request: pytest.FixtureRequest):
    dataset_manager = request.getfixturevalue("mock_dataset_manager")
    stats = dataset_stats(dataset_manager)
    assert "num_frames" in stats
    assert "per_battery_pack_count" in stats
    assert "num_labels" in stats
    assert "per_label_count" in stats
    assert "average_bbox_area" in stats
    assert "average_bbox_size" in stats
    assert "smallest_distance_bboxes_pairwise" in stats


def test_split_train_test_interface_correctness(request: pytest.FixtureRequest):
    dataset_manager = request.getfixturevalue("mock_dataset_manager")
    dataset_manager.frame.return_value.annotations = []
    result = split_train_test(dataset_manager, test_ratio=0.2)
    assert hasattr(result, "train_frame_ids")
    assert hasattr(result, "test_frame_ids")
    assert hasattr(result, "exact_annotation_test_ratio")
    assert isinstance(result.train_frame_ids, list)
    assert isinstance(result.test_frame_ids, list)
    assert isinstance(result.exact_annotation_test_ratio, float)


@patch("libs.dataset.utils.datetime")
def test_cache_split_creates_file(mock_datetime, request: pytest.FixtureRequest):
    dataset_manager = request.getfixturevalue("mock_dataset_manager")
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
