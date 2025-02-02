"""This module provides functions to compute statistics about the dataset."""

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.distance import pdist

from libs.dataset.manager import BATTERY_PACKS, DatasetManager


def _frame_ids_in_battery_pack(dataset_manager: DatasetManager, battery_pack: int) -> List[str]:
    return [
        frame_id
        for frame_id in dataset_manager.frame_ids.keys()
        if f"battery_pack_{battery_pack}" in frame_id
    ]


def _per_battery_pack_frame_counts(dataset_manager: DatasetManager) -> Dict[str, int]:
    return {
        f"battery_pack_{battery_pack}": len(
            _frame_ids_in_battery_pack(dataset_manager, battery_pack)
        )
        for battery_pack in BATTERY_PACKS
    }


def _per_label_count(dataset_manager: DatasetManager) -> Dict[str, int]:
    label_counts = {}
    for frame_id in dataset_manager.frame_ids.keys():
        if (annotations := dataset_manager.frame(frame_id).annotations) is not None:
            for annotation in annotations:
                annotation_label = dataset_manager.label_name_mapper(annotation.label)
                if annotation_label not in label_counts:
                    label_counts[annotation_label] = 0
                label_counts[annotation_label] += 1

    return label_counts


def _average_bbox_area(dataset_manager: DatasetManager) -> float:
    total_area = 0
    annotation_count = 0
    for frame_id in dataset_manager.frame_ids.keys():
        if (annotations := dataset_manager.frame(frame_id).annotations) is not None:
            for annotation in annotations:
                _, _, w, h = annotation.get_bbox()
                total_area += w * h
                annotation_count += 1
    return total_area / annotation_count


def _smallest_distance_bboxes_pairwise(dataset_manager: DatasetManager) -> float:
    def _bboxes_x_y(frame_id):
        if (annotations := dataset_manager.frame(frame_id).annotations) is not None:
            return np.array([a.get_bbox()[:2] for a in annotations])
        return np.empty((0, 2))

    min_distances = []
    for frame_id in dataset_manager.frame_ids.keys():
        bboxes = _bboxes_x_y(frame_id)
        if bboxes.size == 0:  # Skip if no bounding boxes
            continue
        distances = pdist(bboxes, metric="euclidean")
        if distances.size > 0:
            min_distances.append(np.min(distances))

    return np.min(min_distances) if min_distances else float("inf")


def dataset_stats(
    dataset_manager: DatasetManager, logger: Optional[logging.Logger] = None
) -> Dict[str, int]:
    """Return statistics about the dataset."""

    stats = {}
    num_frames = dataset_manager.frame_count()
    if num_frames == 0 and logger:
        logger.warning("No frames found in the dataset")

    if num_frames > 0:
        average_bbox_area = _average_bbox_area(dataset_manager)
        stats = {
            "num_frames": num_frames,
            "per_battery_pack_count": _per_battery_pack_frame_counts(dataset_manager),
            "num_labels": dataset_manager.label_count(),
            "per_label_count": _per_label_count(dataset_manager),
            "average_bbox_area": average_bbox_area,
            "average_bbox_size": np.sqrt(average_bbox_area),
            "smallest_distance_bboxes_pairwise": _smallest_distance_bboxes_pairwise(
                dataset_manager
            ),
        }
    if logger:
        for key, value in stats.items():
            logger.info("Dataset statistics: %s: %s", key, value)

    return stats
