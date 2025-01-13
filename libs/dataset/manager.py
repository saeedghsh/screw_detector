"""
This module provides the `DatasetManager` class for managing datasets related to
battery packs. It includes functionalities for loading datasets, retrieving
frame IDs, accessing image and point cloud files, and obtaining annotations for
specific frames.
"""

import functools
import os
from dataclasses import dataclass

# pylint: disable=no-member
from typing import Dict, List

import cv2
import datumaro
import numpy as np
import open3d as o3d
from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.media import Image

from libs.path_utils import repo_root

DATASET_PATH = os.path.join(repo_root(), "dataset/screw_detection_challenge")
BATTERY_PACKS = [1, 2]


def _annotations_file_path(battery_pack: int) -> str:
    """Return the path to the annotations file for the given battery pack."""
    return os.path.join(DATASET_PATH, f"battery_pack_{battery_pack}_annotations_datumaro.json")


def image_path(frame_id: str) -> str:  # pragma: no cover
    """Return the path to the image file for the given frame ID."""
    p = os.path.join(DATASET_PATH, f"{frame_id}.png")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Image file not found: {p}")
    return p


def pointcloud_path(frame_id: str) -> str:  # pragma: no cover
    """Return the path to the point cloud file for the given frame ID."""
    p = os.path.join(DATASET_PATH, f"{frame_id}.ply")
    if not os.path.isfile(p):  # pragma: no cover
        raise FileNotFoundError(f"Point cloud file not found: {p}")
    return p


@dataclass
class Frame:  # pylint: disable=missing-class-docstring
    id: str
    image: np.ndarray
    pointcloud: o3d.geometry.PointCloud
    annotations: datumaro.components.annotation.Annotations

    def battery_pack(self) -> str:
        """Return the battery pack number."""
        return self.id.split("/")[0]

    def frame_name(self) -> str:
        """Return the frame name."""
        return self.id.split("/")[-1]


class DatasetManager:
    """Manage the dataset."""

    def __init__(self):
        datasets = [
            datumaro.Dataset.import_from(annotation_file, format="datumaro")
            for annotation_file in map(_annotations_file_path, BATTERY_PACKS)
        ]
        self._datasets = datasets
        self._dataset = DatasetManager._merge_datasets(datasets)
        self._label_categories = self._dataset.categories().get(datumaro.AnnotationType.label)
        self._frame_ids = {frame.id: idx for idx, frame in enumerate(self._dataset)}
        self.__post_init__()

    def __post_init__(self):
        for frame_id in self._frame_ids:
            DatasetManager._validate_frame_id(frame_id)

    @staticmethod
    def _validate_frame_id(frame_id: str):
        """Check the frame ID format."""
        parts = frame_id.split("/")
        if len(parts) != 3:  # pragma: no cover
            raise ValueError("Frame ID must be in the format: battery_pack_i/frame_name/frame_name")
        if "battery_pack_" not in parts[0]:  # pragma: no cover
            raise ValueError("Frame ID must start with 'battery_pack_'")
        if parts[1] != parts[2]:  # pragma: no cover
            raise ValueError("Frame ID must have frame name repeated twice in it")

    @staticmethod
    def _merge_datasets(datasets: List[datumaro.Dataset]) -> datumaro.Dataset:
        """Merge multiple datasets into a single dataset.

        NOTE: ideally one would use `item.media` for the media (image) file path
        e.g. ImageFromFile(path='battery_pack_2_annotations_datumaro/...')

        However, the media path is specified w.r.t. annotation file. And also
        due to the issue CVAT had with 3D `ply` file, the references to media
        (image and pointcloud) and functionality to load data are implemented
        directly in DatasetManager class. Hence media_type is considered
        irrelevant and set to Image.
        """
        merged_dataset = datumaro.Dataset(media_type=Image)

        label_categories = LabelCategories()
        for dataset in datasets:
            dataset_label_categories = dataset.categories().get(AnnotationType.label)
            if dataset_label_categories:
                for label in dataset_label_categories.items:
                    if label.name not in label_categories:
                        label_categories.add(label.name, attributes=label.attributes)
        merged_dataset.define_categories({AnnotationType.label: label_categories})

        for dataset in datasets:
            for item in dataset:
                merged_dataset.put(item)

        return merged_dataset

    @property
    def frame_ids(self) -> Dict[str, int]:
        """Return a dictionary of frame IDs.
        keys are string in the form of:
            battery_pack_2/MAN_ImgCap_closer_zone_60/MAN_ImgCap_closer_zone_60
        values are integers starting from 0, representing the index of the frame in the dataset.
        """
        return self._frame_ids

    @functools.lru_cache(maxsize=50)
    def label_name_mapper(self, annotation_label_idx: int) -> str:
        """Return the name of the label with the given index."""
        return self._label_categories.items[annotation_label_idx].name

    @staticmethod
    def _image(frame_id: str) -> np.ndarray:
        """Return the image for the given frame ID."""
        return cv2.imread(image_path(frame_id))

    @staticmethod
    def _pointcloud(frame_id: str) -> o3d.geometry.PointCloud:
        """Return the point cloud for the given frame ID."""
        return o3d.io.read_point_cloud(pointcloud_path(frame_id))

    def _annotations(self, frame_id: str) -> datumaro.components.annotation.Annotations:
        """Return the annotations for the given frame ID."""
        frame_idx = self._frame_ids[frame_id]
        return self._dataset[frame_idx].annotations

    @functools.lru_cache(maxsize=50)
    def frame(self, frame_id: str) -> Frame:
        """Return the frame with the given frame ID."""
        return Frame(
            id=frame_id,
            image=DatasetManager._image(frame_id),
            pointcloud=DatasetManager._pointcloud(frame_id),
            annotations=self._annotations(frame_id),
        )
