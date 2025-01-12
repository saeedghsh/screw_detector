"""
This module provides the `DatasetManager` class for managing datasets related to
battery packs. It includes functionalities for loading datasets, retrieving
frame IDs, accessing image and point cloud files, and obtaining annotations for
specific frames.
"""

import logging
import random
from collections import Counter
from types import SimpleNamespace

# pylint: disable=no-member
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.distance import pdist

from libs.dataset.manager import BATTERY_PACKS, DatasetManager


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
