"""This module provides functions for splitting a dataset into subsets

Can be used for different splits (client side responsibility):
1. Split the whole dataset into train and test subsets set
2. Split the train subset into training and validation subsets
The split ratio applies to annotation counts, not frame counts.
"""

import json
import os
import random
from collections import Counter
from types import SimpleNamespace
from typing import Dict, List, Set, Tuple

from libs.dataset.manager import DATASET_PATH, DatasetManager


def data_split_cache_path(ensure_exist: bool = False) -> str:
    """Return path to the cache directory"""
    cache_dir = os.path.join(DATASET_PATH, "data_split_cache")
    if ensure_exist:
        os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def save_split(split_result: SimpleNamespace, split_file: str) -> None:
    """Save the split result to a file."""
    with open(split_file, "w", encoding="utf-8") as file:
        json.dump(split_result.__dict__, file, indent=4)


def load_cached_split(file_path: str) -> SimpleNamespace:
    """Load a cached split from a given file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cached split file not found: {file_path}")  # pragma: no cover
    with open(file_path, "r", encoding="utf-8") as file:
        split_data = json.load(file)
    return SimpleNamespace(**split_data)


def _verify_split(
    subsets_counts: Dict[str, Counter[str]],
    annotation_labels: Set[str],
    subset_names: Tuple[str, str],
):
    """Verify that each subset contains at least one annotation of each label."""
    for subset_name in subset_names:
        for label in annotation_labels:
            if label not in subsets_counts[subset_name] or subsets_counts[subset_name][label] == 0:
                raise ValueError(
                    f"Subset '{subset_name}' does not contain any annotations of label '{label}'."
                )


def _data_split_stats(
    subsets_counts: Dict[str, Counter[str]],
    total_labels_counts: Counter[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return statistics about the split.

    Include the count and ratio of annotations per label in each subset.

    NOTE: The ratio is exact as opposed to the desired split ratio.
    """
    stats: Dict[str, Dict[str, Dict[str, float]]] = {"count": {}, "ratio": {}}
    for subset_name, counts in subsets_counts.items():
        for label, count in counts.items():
            if label not in stats["count"]:
                stats["count"][label] = {}
                stats["ratio"][label] = {}
            stats["count"][label][subset_name] = count
            stats["ratio"][label][subset_name] = float(count) / total_labels_counts[label]
    return stats


def _counts_labels(
    dataset_manager: DatasetManager, frame_ids: List[str], annotation_labels: Set[str]
) -> SimpleNamespace:
    """
    Return the count of [annotation] labels per label in the given frames.

    The counts are computed for the specified labels and across all frames. of
    what dataset_manager contains (in terms of labels and frames), the counts
    are computed for only the labels in annotation_labels and across frames
    specified in frame_ids.

    * frames_labels_counts: the count of annotations per label per frame
    * total_labels_counts: the total count of annotations per label across all
      frames
    """
    frames_labels_counts: Dict[str, Counter[str]] = {}
    total_labels_counts: Counter[str] = Counter()
    for frame_id in frame_ids:
        labels_counts: Counter[str] = Counter()
        if (annotations := dataset_manager.frame(frame_id).annotations) is not None:
            for annotation in annotations:
                label_name = dataset_manager.label_name_mapper(annotation.label)
                if label_name in annotation_labels:
                    labels_counts[label_name] += 1
                    total_labels_counts[label_name] += 1
            frames_labels_counts[frame_id] = labels_counts
    return SimpleNamespace(
        frames_labels_counts=frames_labels_counts,
        total_labels_counts=total_labels_counts,
    )


def split(
    dataset_manager: DatasetManager,
    frame_ids: List[str],
    annotation_labels: Set[str],
    subset_names: Tuple[str, str],
    desired_split_ratio: float,
) -> SimpleNamespace:
    """
    Return a split of the frames into two subsets based on the specified ratio.

    NOTE: The split is frame-based, but the ratio applies to annotation counts.
    It ensure that the ratio of annotations with the specified labels in each
    subset is approximately equal to the given split ratio. Frames are randomly
    assigned to subsets until the desired ratio for at least one labels is
    reached.
    """
    # Count annotations for each label across all frames
    count_result = _counts_labels(dataset_manager, frame_ids, annotation_labels)
    frames_labels_counts = count_result.frames_labels_counts
    total_labels_counts = count_result.total_labels_counts

    # Desired number of annotations per label according to the split ratio
    desired_split_count = {
        label: int(count * desired_split_ratio) for label, count in total_labels_counts.items()
    }

    # Assign frames to subset to achieve the desired ratio
    # desired_split_ratio:       corresponds to the first subset in subset_names
    # (1 - desired_split_ratio): corresponds to the second subset
    random.shuffle(frame_ids)
    subsets: Dict[str, list] = {n: [] for n in subset_names}
    subsets_counts: Dict[str, Counter[str]] = {n: Counter() for n in subset_names}
    first_subset_full = False
    for frame_id in frame_ids:
        subset_name = subset_names[0] if not first_subset_full else subset_names[1]
        subsets[subset_name].append(frame_id)
        subsets_counts[subset_name].update(frames_labels_counts[frame_id])
        first_subset_full = any(
            subsets_counts[subset_names[0]].get(label, 0) > desired_split_count.get(label, 0)
            for label in annotation_labels
        )

    _verify_split(subsets_counts, annotation_labels, subset_names)

    return SimpleNamespace(
        **subsets,
        desired_split_ratio=desired_split_ratio,
        stats=_data_split_stats(subsets_counts, total_labels_counts),
    )
