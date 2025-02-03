# pylint: disable=missing-module-docstring, missing-function-docstring
import json
from unittest.mock import patch

import numpy as np
import pytest

from libs.dataset.data_reader import load_images
from libs.dataset.split import load_cached_split


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
        assert "image" in images
        assert isinstance(images["image"], np.ndarray)


def test_load_images_directory():
    mock_file_list = ["mock_path/image1.png", "mock_path/subdir/image2.png"]
    with (
        patch("os.path.exists", return_value=True),
        patch("os.path.isfile", return_value=False),
        patch("os.path.isdir", return_value=True),
        patch("glob.glob", return_value=mock_file_list),
        patch("cv2.imread", return_value=np.zeros((100, 100, 3))),
    ):
        images = load_images("mock_path")
        assert len(images) == len(mock_file_list)
        assert "image1" in images
        assert "subdir_image2" in images
        assert all(isinstance(img, np.ndarray) for img in images.values())


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
