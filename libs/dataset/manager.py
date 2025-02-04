"""
This module provides the `DatasetManager` class for managing datasets related to
battery packs. It includes functionalities for loading datasets, retrieving
frame IDs, accessing image and point cloud files, and obtaining annotations for
specific frames.
"""

import functools
import os

# pylint: disable=no-member
from typing import Dict, List

import datumaro
from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.media import Image

from libs.dataset.data_reader import read_camera_transform, read_image, read_pointcloud
from libs.dataset.data_structure import Frame
from libs.path_utils import repo_root

DATASET_PATH = os.path.join(repo_root(), "dataset/screw_detection_challenge")
BATTERY_PACKS = [1, 2]


def _filter_label_categories(
    label_categories: LabelCategories,
) -> LabelCategories:  # pragma: no cover
    """Filter the label categories to keep only the labels of interest.
    The annotation project in CVAT has more labels that we are not interested in."""
    labels_to_keep = ["screw_head", "screw_hole"]
    filtered = LabelCategories(attributes=label_categories.attributes)
    for label in label_categories.items:
        if label.name in labels_to_keep:
            filtered.add(label.name, attributes=label.attributes)
    return filtered


def _annotations_file_path(battery_pack: int) -> str:
    """Return the path to the annotations file for the given battery pack."""
    return os.path.join(DATASET_PATH, f"battery_pack_{battery_pack}_annotations_datumaro.json")


class DatasetManager:
    """Manage the dataset."""

    def __init__(self):
        datasets = [
            datumaro.Dataset.import_from(annotation_file, format="datumaro")
            for annotation_file in map(_annotations_file_path, BATTERY_PACKS)
        ]
        self._datasets = datasets
        self._dataset = DatasetManager._merge_datasets(datasets)
        self._label_categories = _filter_label_categories(
            self._dataset.categories().get(datumaro.AnnotationType.label)
        )
        self._frame_ids = {frame.id: idx for idx, frame in enumerate(self._dataset)}
        self.__post_init__()

    def __post_init__(self):
        for frame_id in self._frame_ids:
            DatasetManager._validate_frame_id(frame_id)

    @staticmethod
    def _validate_frame_id(frame_id: str):  # pragma: no cover
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

    def frame_count(self) -> int:  # pragma: no cover
        """Return the number of frames in the dataset."""
        return len(self._dataset)

    def label_count(self) -> int:  # pragma: no cover
        """Return the number of labels in the dataset."""
        return len(self._label_categories.items)

    @functools.lru_cache(maxsize=50)
    def label_name_mapper(self, annotation_label_idx: int) -> str:  # pragma: no cover
        """Return the name of the label with the given index."""
        if annotation_label_idx < 0 or annotation_label_idx >= len(self._label_categories.items):
            return "UNKNOWN"
        return self._label_categories.items[annotation_label_idx].name

    def _annotations(self, frame_id: str) -> datumaro.components.annotation.Annotations:
        """Return the annotations for the given frame ID."""
        frame_idx = self._frame_ids[frame_id]
        return self._dataset[frame_idx].annotations

    @functools.lru_cache(maxsize=50)
    def frame(self, frame_id: str) -> Frame:
        """Return the frame with the given frame ID."""
        return Frame(
            id=frame_id,
            image=read_image(DATASET_PATH, frame_id),
            pointcloud=read_pointcloud(DATASET_PATH, frame_id),
            camera_transform=read_camera_transform(DATASET_PATH, frame_id),
            annotations=self._annotations(frame_id),
        )
