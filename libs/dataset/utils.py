"""
This module provides the `DatasetManager` class for managing datasets related to
battery packs. It includes functionalities for loading datasets, retrieving
frame IDs, accessing image and point cloud files, and obtaining annotations for
specific frames.
"""

import glob
import json
import logging
import os
import random
from collections import Counter
from datetime import datetime
from types import SimpleNamespace

# pylint: disable=no-member
from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy.spatial.distance import pdist

from libs.dataset.manager import BATTERY_PACKS, DATASET_PATH, DatasetManager

CACHE_DIR = os.path.join(DATASET_PATH, "data_split_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def dataset_stats(
    dataset_manager: DatasetManager, logger: Optional[logging.Logger] = None
) -> Dict[str, int]:
    """Return statistics about the dataset."""

    def _frame_ids_in_battery_pack(battery_pack: int) -> List[str]:
        return [
            frame_id
            for frame_id in dataset_manager.frame_ids.keys()
            if f"battery_pack_{battery_pack}" in frame_id
        ]

    def _per_battery_pack_frame_counts() -> Dict[str, int]:
        return {
            f"battery_pack_{battery_pack}": len(_frame_ids_in_battery_pack(battery_pack))
            for battery_pack in BATTERY_PACKS
        }

    def _per_label_count() -> Dict[str, int]:
        label_counts = {}
        for frame_id in dataset_manager.frame_ids.keys():
            if (annotations := dataset_manager.frame(frame_id).annotations) is not None:
                for annotation in annotations:
                    annotation_label = dataset_manager.label_name_mapper(annotation.label)
                    if annotation_label not in label_counts:
                        label_counts[annotation_label] = 0
                    label_counts[annotation_label] += 1

        return label_counts

    def _average_bbox_area() -> float:
        total_area = 0
        annotation_count = 0
        for frame_id in dataset_manager.frame_ids.keys():
            if (annotations := dataset_manager.frame(frame_id).annotations) is not None:
                for annotation in annotations:
                    _, _, w, h = annotation.get_bbox()
                    total_area += w * h
                    annotation_count += 1
        return total_area / annotation_count

    def _smallest_distance_bboxes_pairwise() -> float:
        def _bboxes_x_y(frame_id):
            if (annotations := dataset_manager.frame(frame_id).annotations) is not None:
                return np.array([a.get_bbox()[:2] for a in annotations])
            return np.empty((0, 2))  # pragma: no cover

        min_distances = []
        for frame_id in dataset_manager.frame_ids.keys():
            bboxes = _bboxes_x_y(frame_id)
            if bboxes.size == 0:  # Skip if no bounding boxes
                continue  # pragma: no cover
            distances = pdist(bboxes, metric="euclidean")
            if distances.size > 0:
                min_distances.append(np.min(distances))  # pragma: no cover

        return np.min(min_distances) if min_distances else float("inf")

    stats = {}
    num_frames = dataset_manager.frame_count()
    if num_frames == 0 and logger:
        logger.warning("No frames found in the dataset")

    if num_frames > 0:
        average_bbox_area = _average_bbox_area()
        stats = {
            "num_frames": num_frames,
            "per_battery_pack_count": _per_battery_pack_frame_counts(),
            "num_labels": dataset_manager.label_count(),
            "per_label_count": _per_label_count(),
            "average_bbox_area": average_bbox_area,
            "average_bbox_size": np.sqrt(average_bbox_area),
            "smallest_distance_bboxes_pairwise": _smallest_distance_bboxes_pairwise(),
        }
    if logger:
        for key, value in stats.items():
            logger.info("Dataset statistics: %s: %s", key, value)  # pragma: no cover

    return stats


def split_train_test(  # pylint: disable=too-many-locals
    dataset_manager: DatasetManager, test_ratio: float
) -> SimpleNamespace:
    """
    Splits the dataset into train and test sets based on the specified ratio of annotations
    for the labels "screw_head" and "screw_hole".

    The split is frame-based, ensuring that the ratio of annotations with the specified labels
    in each set is approximately equal to the given split ratio. Frames are randomly assigned
    to train or test sets while maintaining the desired ratio of annotations.
    """
    annotation_labels = {"screw_head", "screw_hole"}

    # Count annotations for each label across all frames
    frame_label_counts = {}
    total_label_counts: Counter[str] = Counter()

    for frame_id in dataset_manager.frame_ids.keys():
        label_counts: Counter[str] = Counter()
        if (annotations := dataset_manager.frame(frame_id).annotations) is not None:
            for annotation in annotations:  # pragma: no cover
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
            current_test_counts.get(label, 0) + label_counts.get(label, 0)
            <= desired_test_counts.get(label, 0)
            for label in annotation_labels
        ):
            test_frame_ids.append(frame_id)
            current_test_counts.update(label_counts)
        else:  # pragma: no cover
            train_frame_ids.append(frame_id)

    # Compute the exact annotation ratio achieved
    total_test_annotations = sum(current_test_counts.values())
    total_annotations = sum(total_label_counts.values())
    exact_test_ratio = total_test_annotations / total_annotations if total_annotations > 0 else 0.0

    return SimpleNamespace(
        train_frame_ids=train_frame_ids,
        test_frame_ids=test_frame_ids,
        exact_annotation_test_ratio=exact_test_ratio,
    )


def cache_split(dataset_manager: DatasetManager, test_ratio: float) -> str:
    """Split the dataset and store the split result in a timestamped file."""
    split_result = split_train_test(dataset_manager, test_ratio)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    split_file = os.path.join(CACHE_DIR, f"{timestamp}_{test_ratio}_split.json")

    split_data = {
        "train_frame_ids": split_result.train_frame_ids,
        "test_frame_ids": split_result.test_frame_ids,
        "split_ratio": test_ratio,
        "timestamp": timestamp,
    }

    with open(split_file, "w", encoding="utf-8") as file:
        json.dump(split_data, file, indent=4)

    return split_file


def load_cached_split(file_path: str) -> SimpleNamespace:
    """Load a cached split from a given file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cached split file not found: {file_path}")  # pragma: no cover
    with open(file_path, "r", encoding="utf-8") as file:
        split_data = json.load(file)
    return SimpleNamespace(**split_data)


def load_images(input_path: str) -> List[np.ndarray]:
    """
    Load images from a single file or a directory.

    :param input_path: Path to an image file or a directory containing images.
    :return: List of loaded images as numpy arrays.
    :raises FileNotFoundError: If the input path does not exist or no images are found.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path '{input_path}' does not exist.")

    images = []
    if os.path.isfile(input_path):
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Failed to load image from '{input_path}'.")
        images.append(image)
    elif os.path.isdir(input_path):
        for file_path in sorted(glob.glob(os.path.join(input_path, "**", "*.png"), recursive=True)):
            image = cv2.imread(file_path)
            if image is not None:
                images.append(image)
            else:
                raise ValueError(f"Failed to load image from '{file_path}'.")  # pragma: no cover
    else:
        raise ValueError(f"Input path '{input_path}' is neither a file nor a directory.")

    if not images:
        raise FileNotFoundError(f"No images found in '{input_path}'.")

    return images
