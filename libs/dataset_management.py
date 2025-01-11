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
from typing import Dict

import cv2
import datumaro
import numpy as np
import open3d as o3d

from libs.path_utils import repo_root

DATASET_PATH = os.path.join(repo_root(), "dataset/screw_detection_challenge")
BATTERY_PACKS = [1, 2]


def _annotations_file_path(battery_pack: int, dataset_path: str = DATASET_PATH) -> str:
    """Return the path to the annotations file for the given battery pack."""
    return os.path.join(dataset_path, f"battery_pack_{battery_pack}_annotations_datumaro.json")


@dataclass
class Frame:  # pylint: disable=missing-class-docstring
    id: str
    image: np.ndarray
    pointcloud: o3d.geometry.PointCloud
    annotations: datumaro.components.annotation.Annotations

    def output_path_2d(self) -> str:
        """Return the output path for the annotated image."""
        battery_pack, frame_name, _ = self.id.split("/")
        return f"{battery_pack}_{frame_name}_2d.png"


class DatasetManager:
    """Manage the dataset."""

    def __init__(self, battery_pack: int):
        if battery_pack not in BATTERY_PACKS:
            raise ValueError(f"Battery pack must be {BATTERY_PACKS}")
        self._battery_pack = battery_pack
        self._annotation_file = _annotations_file_path(battery_pack)
        self._dataset = datumaro.Dataset.import_from(self._annotation_file, format="datumaro")
        self._frame_ids = {frame.id: idx for idx, frame in enumerate(self._dataset)}
        self._label_categories = self._dataset.categories().get(datumaro.AnnotationType.label)

    @property
    def frame_ids(self) -> Dict[str, int]:
        """Return a dictionary of frame IDs.
        keys are string in the form of:
            battery_pack_2/MAN_ImgCap_closer_zone_60/MAN_ImgCap_closer_zone_60
        values are integers starting from 0, representing the index of the frame in the dataset.
        """
        return self._frame_ids

    def frame(self, frame_id: str) -> Frame:
        """Return the frame with the given frame ID."""
        return Frame(
            id=frame_id,
            image=self.image(frame_id),
            pointcloud=self.pointcloud(frame_id),
            annotations=self.frame_annotations(frame_id),
        )

    @functools.lru_cache(maxsize=50)
    def label_name_mapper(self, annotation_label_idx: int) -> str:
        """Return the name of the label with the given index."""
        return self._label_categories.items[annotation_label_idx].name

    @staticmethod
    @functools.lru_cache(maxsize=50)
    def _image_path(frame_id: str, dataset_path: str) -> str:
        """Return the path to the image file for the given frame ID."""
        return os.path.join(dataset_path, f"{frame_id}.png")

    @staticmethod
    @functools.lru_cache(maxsize=50)
    def _pointcloud_path(frame_id: str, dataset_path: str) -> str:
        """Return the path to the point cloud file for the given frame ID."""
        return os.path.join(dataset_path, f"{frame_id}.ply")

    @staticmethod
    @functools.lru_cache(maxsize=50)
    def image(frame_id: str, dataset_path: str = DATASET_PATH) -> np.ndarray:
        """Return the image for the given frame ID."""
        image_path = DatasetManager._image_path(frame_id, dataset_path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return cv2.imread(image_path)

    @staticmethod
    @functools.lru_cache(maxsize=50)
    def pointcloud(frame_id: str, dataset_path: str = DATASET_PATH) -> o3d.geometry.PointCloud:
        """Return the point cloud for the given frame ID."""
        pointcloud_path = DatasetManager._pointcloud_path(frame_id, dataset_path)
        if not os.path.isfile(pointcloud_path):
            raise FileNotFoundError(f"Point cloud file not found: {pointcloud_path}")
        return o3d.io.read_point_cloud(pointcloud_path)

    def frame_annotations(self, frame_id: str) -> datumaro.components.annotation.Annotations:
        """Return the annotations for the given frame ID."""
        frame_idx = self._frame_ids[frame_id]
        return self._dataset[frame_idx].annotations
