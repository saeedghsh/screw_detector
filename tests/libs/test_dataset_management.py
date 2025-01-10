# pylint: disable=missing-module-docstring, missing-function-docstring
from unittest.mock import MagicMock, patch

import datumaro
import numpy as np
import open3d as o3d
import pytest
from pytest import FixtureRequest

from libs.dataset_management import DATASET_PATH, DatasetManager


@pytest.fixture
def dataset_manager_fixture():
    with patch("libs.dataset_management.datumaro.Dataset.import_from") as mock_import:
        mock_dataset = MagicMock()
        mock_import.return_value = mock_dataset
        mock_dataset.categories.return_value.get.return_value.items = [MagicMock(name="MockLabel")]
        mock_dataset.categories.return_value.get.return_value.items[0].name = "mock_label"
        mock_dataset.__iter__.return_value = iter([MagicMock(id="mock_frame_1")])
        mock_dataset.__getitem__.side_effect = lambda idx: MagicMock(
            annotations=[MagicMock(spec=datumaro.components.annotation.Annotation)]
        )
        return DatasetManager(1)


@pytest.fixture
def dataset_manager_fixture_multi_labels():
    with patch("libs.dataset_management.datumaro.Dataset.import_from") as mock_import:
        mock_dataset = MagicMock()
        mock_import.return_value = mock_dataset
        # Simulate multiple labels with proper string names
        label1 = MagicMock()
        label1.name = "Label1"
        label2 = MagicMock()
        label2.name = "Label2"
        mock_dataset.categories.return_value.get.return_value.items = [label1, label2]
        return DatasetManager(1)


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


def test_label_name_mapper(request: FixtureRequest):
    dataset_manager = request.getfixturevalue("dataset_manager_fixture")
    label_index = 0  # Assuming 0 is a valid index in the dataset
    with patch.object(dataset_manager, "label_name_mapper", return_value="mock_label"):
        label_name = dataset_manager.label_name_mapper(label_index)
    assert isinstance(label_name, str)


def test_image_valid():
    frame_id = "battery_pack_1/MAN_ImgCap_closer_zone_10/MAN_ImgCap_closer_zone_10"
    with patch("os.path.isfile", return_value=True):
        with patch("cv2.imread", return_value=np.zeros((100, 100, 3))):
            image = DatasetManager.image(frame_id, DATASET_PATH)
    assert isinstance(image, np.ndarray)


def test_pointcloud_valid():
    frame_id = "battery_pack_1/MAN_ImgCap_closer_zone_10/MAN_ImgCap_closer_zone_10"
    with patch("os.path.isfile", return_value=True):
        with patch("open3d.io.read_point_cloud", return_value=o3d.geometry.PointCloud()):
            pointcloud = DatasetManager.pointcloud(frame_id, DATASET_PATH)
    assert isinstance(pointcloud, o3d.geometry.PointCloud)


def test_frame_annotations(request: FixtureRequest):
    dataset_manager = request.getfixturevalue("dataset_manager_fixture")
    frame_id = next(iter(dataset_manager.frame_ids))  # Get any valid frame_id
    annotations = dataset_manager.frame_annotations(frame_id)
    assert isinstance(annotations, list)
    assert all(isinstance(ann, datumaro.components.annotation.Annotation) for ann in annotations)


def test_invalid_battery_pack():
    with pytest.raises(ValueError, match="Battery pack must be "):
        DatasetManager(3)


def test_image_file_not_found():
    frame_id = "battery_pack_1/MAN_ImgCap_closer_zone_10/MAN_ImgCap_closer_zone_10"
    # Clear cache before the test, otherwise @functools.lru_cache will make the patch ineffective
    DatasetManager.image.cache_clear()  # Clear cache before the test
    with patch("os.path.isfile", return_value=False):  # Return False to trigger FileNotFoundError
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            DatasetManager.image(frame_id, DATASET_PATH)


def test_pointcloud_file_not_found():
    frame_id = "battery_pack_1/MAN_ImgCap_closer_zone_10/MAN_ImgCap_closer_zone_10"
    # Clear cache before the test, otherwise @functools.lru_cache will make the patch ineffective
    DatasetManager.pointcloud.cache_clear()
    with patch("os.path.isfile", return_value=False):  # Return False to trigger FileNotFoundError
        with pytest.raises(FileNotFoundError, match="Point cloud file not found"):
            DatasetManager.pointcloud(frame_id, DATASET_PATH)
