"""
This module provides the `DatasetManager` class for managing datasets related to
battery packs. It includes functionalities for loading datasets, retrieving
frame IDs, accessing image and point cloud files, and obtaining annotations for
specific frames.
"""

import functools
import logging
import os
import random
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

# pylint: disable=no-member
from typing import Dict, List, Optional

import cv2
import datumaro
import numpy as np
import open3d as o3d
from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.media import Image
from scipy.spatial.distance import pdist

from libs.path_utils import repo_root

DATASET_PATH = os.path.join(repo_root(), "dataset/screw_detection_challenge")
BATTERY_PACKS = [1, 2]


def _annotations_file_path(battery_pack: int) -> str:
    """Return the path to the annotations file for the given battery pack."""
    return os.path.join(DATASET_PATH, f"battery_pack_{battery_pack}_annotations_datumaro.json")


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
        if len(parts) != 3:
            raise ValueError("Frame ID must be in the format: battery_pack_i/frame_name/frame_name")
        if "battery_pack_" not in parts[0]:
            raise ValueError("Frame ID must start with 'battery_pack_'")
        if parts[1] != parts[2]:
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
    def _image(frame_id: str, dataset_path: str = DATASET_PATH) -> np.ndarray:
        """Return the image for the given frame ID."""
        image_path = os.path.join(dataset_path, f"{frame_id}.png")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return cv2.imread(image_path)

    @staticmethod
    def _pointcloud(frame_id: str, dataset_path: str = DATASET_PATH) -> o3d.geometry.PointCloud:
        """Return the point cloud for the given frame ID."""
        pointcloud_path = os.path.join(dataset_path, f"{frame_id}.ply")
        if not os.path.isfile(pointcloud_path):
            raise FileNotFoundError(f"Point cloud file not found: {pointcloud_path}")
        return o3d.io.read_point_cloud(pointcloud_path)

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


def dataset_stats(
    dataset_manger: DatasetManager,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, int]:  # pragma: no cover
    # pylint: disable=protected-access
    """Return statistics about the dataset."""

    def _frame_ids_in_battery_pack(battery_pack: int) -> List[str]:
        return [
            frame_id
            for frame_id in dataset_manger.frame_ids.keys()
            if f"battery_pack_{battery_pack}" in frame_id
        ]

    def _per_battery_pack_frame_counts() -> Dict[str, int]:
        return {
            f"battery_pack_{battery_pack}": len(_frame_ids_in_battery_pack(battery_pack))
            for battery_pack in BATTERY_PACKS
        }

    def _per_label_count() -> Dict[str, int]:
        label_counts = {}
        for frame_id in dataset_manger.frame_ids.keys():
            for annotation in dataset_manger._annotations(frame_id):
                annotation_label = dataset_manger.label_name_mapper(annotation.label)
                if annotation_label not in label_counts:
                    label_counts[annotation_label] = 0
                label_counts[annotation_label] += 1
        return label_counts

    def _average_bbox_area() -> float:
        total_area = 0
        annotation_count = 0
        for frame_id in dataset_manger.frame_ids.keys():
            for annotation in dataset_manger._annotations(frame_id):
                _, _, w, h = annotation.get_bbox()
                total_area += w * h
                annotation_count += 1
        return total_area / annotation_count

    def _smallest_distance_bboxes_pairwise() -> float:

        def _bboxes_x_y(frame_id):
            return np.array([a.get_bbox()[:2] for a in dataset_manger._annotations(frame_id)])

        min_distances = []
        for frame_id in dataset_manger.frame_ids.keys():
            distances = pdist(_bboxes_x_y(frame_id), metric="euclidean")
            min_distances.append(np.min(distances))
        return np.min(min_distances)

    stats = {}
    num_frames = len(dataset_manger._dataset)
    if num_frames == 0 and logger:
        logger.warning("No frames found in the dataset")

    if num_frames > 0:
        average_bbox_area = _average_bbox_area()
        stats = {
            "num_frames": num_frames,
            "per_battery_pack_count": _per_battery_pack_frame_counts(),
            "num_labels": len(dataset_manger._label_categories.items),
            "per_label_count": _per_label_count(),
            "average_bbox_area": average_bbox_area,
            "average_bbox_size": np.sqrt(average_bbox_area),
            "smallest_distance_bboxes_pairwise": _smallest_distance_bboxes_pairwise(),
        }
    if logger:
        for key, value in stats.items():
            logger.info("Dataset statistics: %s: %s", key, value)

    return stats


def split_train_test(  # pylint: disable=too-many-locals
    dataset_manager: DatasetManager, test_ratio: float
) -> SimpleNamespace:  # pragma: no cover
    """
    Splits the dataset into train and test sets based on the specified ratio of annotations
    for the labels "screw_head" and "screw_hole".

    The split is frame-based, ensuring that the ratio of annotations with the specified labels
    in each set is approximately equal to the given split ratio. Frames are randomly assigned
    to train or test sets while maintaining the desired ratio of annotations.

    Args:
        dataset_manager (DatasetManager): The dataset manager containing frames and annotations.
        test_ratio (float): The desired ratio of annotations in the test set (between 0 and 1).

    Returns:
        SimpleNamespace: An object containing:
            - train_frame_ids (list of str): List of frame IDs in the train set.
            - test_frame_ids (list of str): List of frame IDs in the test set.
            - exact_annotation_test_ratio (float): The exact ratio of annotations in the test set.
    """
    annotation_labels = {"screw_head", "screw_hole"}

    # Count annotations for each label across all frames
    frame_label_counts = {}
    total_label_counts: Counter[str] = Counter()

    for frame_id in dataset_manager.frame_ids.keys():
        label_counts: Counter[str] = Counter()
        for annotation in dataset_manager.frame(frame_id).annotations:
            label_name = dataset_manager.label_name_mapper(annotation.label)
            if label_name in annotation_labels:
                label_counts[label_name] += 1
                total_label_counts[label_name] += 1
        frame_label_counts[frame_id] = label_counts

    # Desired number of annotations per label for test set
    desired_test_counts = {
        label: int(count * test_ratio) for label, count in total_label_counts.items()
    }

    # Randomly shuffle frames for unbiased selection
    frames = list(frame_label_counts.keys())
    random.shuffle(frames)

    train_frame_ids, test_frame_ids = [], []
    current_test_counts: Counter[str] = Counter()

    # Assign frames to train or test to achieve the desired ratio
    for frame_id in frames:
        label_counts = frame_label_counts[frame_id]
        if all(
            current_test_counts[label] + label_counts[label] <= desired_test_counts[label]
            for label in annotation_labels
        ):
            test_frame_ids.append(frame_id)
            current_test_counts.update(label_counts)
        else:
            train_frame_ids.append(frame_id)

    # Compute the exact annotation ratio achieved
    total_test_annotations = sum(current_test_counts.values())
    total_annotations = sum(total_label_counts.values())
    exact_test_ratio = total_test_annotations / total_annotations if total_annotations > 0 else 0

    return SimpleNamespace(
        train_frame_ids=train_frame_ids,
        test_frame_ids=test_frame_ids,
        exact_annotation_test_ratio=exact_test_ratio,
    )
