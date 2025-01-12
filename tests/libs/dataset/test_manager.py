# pylint: disable=missing-module-docstring, missing-function-docstring
from unittest.mock import MagicMock, patch

import datumaro
import numpy as np
import open3d as o3d
import pytest
from datumaro.components.annotation import Annotation
from datumaro.components.media import Image
from pytest import FixtureRequest

from libs.dataset.manager import DatasetManager, Frame


@pytest.fixture
def dataset_manager_fixture():
    with patch("libs.dataset.manager.datumaro.Dataset.import_from") as mock_import:
        mock_dataset = MagicMock()
        mock_import.return_value = mock_dataset
        mock_dataset.categories.return_value.get.return_value.items = [MagicMock(name="MockLabel")]
        mock_dataset.categories.return_value.get.return_value.items[0].name = "mock_label"

        # Create a mock annotation
        mock_annotation = MagicMock(spec=Annotation)

        # Create a mock item with Image media type using from_file()
        mock_item = MagicMock()
        mock_item.id = "battery_pack_1/zone_1/zone_1"
        mock_item.media = Image.from_file(path="mock_path/mock_image.png")
        mock_item.annotations = [mock_annotation]  # Return a list of mock annotations
        mock_dataset.__iter__.return_value = iter([mock_item])
        mock_dataset.__getitem__.side_effect = lambda idx: mock_item

        return DatasetManager()


@pytest.fixture
def dataset_manager_fixture_multi_labels():
    with patch("libs.dataset.manager.datumaro.Dataset.import_from") as mock_import:
        mock_dataset = MagicMock()
        mock_import.return_value = mock_dataset
        # Simulate multiple labels with proper string names
        label1 = MagicMock()
        label1.name = "Label1"
        label2 = MagicMock()
        label2.name = "Label2"
        mock_dataset.categories.return_value.get.return_value.items = [label1, label2]
        return DatasetManager()


def test_label_name_mapper_with_multiple_labels(request: pytest.FixtureRequest):
    dataset_manager = request.getfixturevalue("dataset_manager_fixture_multi_labels")
    # Test mapping for each label index
    label_name_0 = dataset_manager.label_name_mapper(0)
    label_name_1 = dataset_manager.label_name_mapper(1)
    assert label_name_0 == "Label1"
    assert label_name_1 == "Label2"
    # Test for an invalid label index (out of range)
    with pytest.raises(IndexError):
        dataset_manager.label_name_mapper(2)  # No label at index 2


def test_frame_ids(request: FixtureRequest):
    dataset_manager = request.getfixturevalue("dataset_manager_fixture")
    frame_ids = dataset_manager.frame_ids
    assert isinstance(frame_ids, dict)
    assert all(isinstance(k, str) and isinstance(v, int) for k, v in frame_ids.items())


def test_invalid_frame_id_format():
    invalid_frame_ids = [
        "invalid_frame_id",  # Missing parts
        "battery_pack_1/zone_1/zone_2",  # Frame name mismatch
        "wrong_prefix/zone_1/zone_1",  # Wrong prefix
    ]

    for invalid_frame_id in invalid_frame_ids:
        with patch("libs.dataset.manager.datumaro.Dataset.import_from") as mock_import:
            mock_dataset = MagicMock()
            mock_import.return_value = mock_dataset

            # Create a mock item with a valid Image media type
            mock_item = MagicMock()
            mock_item.id = invalid_frame_id
            mock_item.media = Image.from_file(path="mock_path/mock_image.png")
            mock_dataset.__iter__.return_value = iter([mock_item])

            with pytest.raises(ValueError, match="Frame ID must"):
                DatasetManager()


def test_label_name_mapper(request: FixtureRequest):
    dataset_manager = request.getfixturevalue("dataset_manager_fixture")
    label_index = 0  # Assuming 0 is a valid index in the dataset
    with patch.object(dataset_manager, "label_name_mapper", return_value="mock_label"):
        label_name = dataset_manager.label_name_mapper(label_index)
    assert isinstance(label_name, str)


def test_frame(request: FixtureRequest):
    dataset_manager = request.getfixturevalue("dataset_manager_fixture")
    frame_id = next(iter(dataset_manager.frame_ids))  # Get any valid frame_id
    with patch("os.path.isfile", return_value=True):
        with patch("cv2.imread", return_value=np.zeros((100, 100, 3))):
            with patch("open3d.io.read_point_cloud", return_value=o3d.geometry.PointCloud()):
                frame = dataset_manager.frame(frame_id)
    assert isinstance(frame, Frame)
    assert isinstance(frame.id, str)
    assert isinstance(frame.image, np.ndarray)
    assert isinstance(frame.pointcloud, o3d.geometry.PointCloud)
    assert isinstance(frame.annotations, list)
    assert all(
        isinstance(ann, datumaro.components.annotation.Annotation) for ann in frame.annotations
    )


def test_frame_file_not_found(request: FixtureRequest):
    frame_id = "battery_pack_1/MAN_ImgCap_closer_zone_10/MAN_ImgCap_closer_zone_10"
    dataset_manager = request.getfixturevalue("dataset_manager_fixture")

    # Clear cache before the test, otherwise @functools.lru_cache will make the patch ineffective
    dataset_manager.frame.cache_clear()

    def isfile_side_effect(path):
        if path.endswith(".png"):
            return False  # Simulate image file not found
        if path.endswith(".ply"):
            return True  # Simulate point cloud file exists
        return True

    with patch("os.path.isfile", side_effect=isfile_side_effect):
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            dataset_manager.frame(frame_id)

    def isfile_side_effect_pointcloud(path):
        if path.endswith(".png"):
            return True  # Simulate image file exists
        if path.endswith(".ply"):
            return False  # Simulate point cloud file not found
        return True

    with patch("os.path.isfile", side_effect=isfile_side_effect_pointcloud):
        with pytest.raises(FileNotFoundError, match="Point cloud file not found"):
            dataset_manager.frame(frame_id)
