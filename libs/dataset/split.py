"""This module provides functions for splitting a dataset into subsets

Can be used for different splits (client side responsibility): 1) Split the
whole dataset into train and test subsets set, and 2) Split the train subset
into training and validation subsets The split ratio applies to annotation
counts, not frame counts.

NOTE: the number of annotation labels is arbitrary and can vary. But the subsets
into which data is split has to be exactly two.
"""

import heapq
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


def _choose_subset(
    label: str,
    subsets_counts: Dict[str, Counter[str]],
    desired_split_count: Dict[str, int],
    total_labels_counts: Counter[str],
    subset_names: Tuple[str, str],
) -> str:
    """Determine which subset needs correction for the given label."""
    subset1_deficit = desired_split_count[label] - subsets_counts[subset_names[0]].get(label, 0)
    subset2_deficit = (total_labels_counts[label] - desired_split_count[label]) - subsets_counts[
        subset_names[1]
    ].get(label, 0)
    return subset_names[0] if subset1_deficit > subset2_deficit else subset_names[1]


def _build_heap(
    annotation_labels: Set[str],
    subsets_counts: Dict[str, Counter[str]],
    desired_split_count: Dict[str, int],
    subset_names: Tuple[str, str],
) -> List[Tuple[int, str]]:
    """Rebuild the Min-Heap based on per-label deviation."""
    heap: List[Tuple[int, str]] = []
    for label in annotation_labels:
        deviation = abs(
            subsets_counts[subset_names[0]].get(label, 0) - desired_split_count.get(label, 0)
        )
        heapq.heappush(heap, (-deviation, label))  # Max deviation first
    return heap


def split(
    dataset_manager: DatasetManager,
    frame_ids: List[str],
    annotation_labels: Set[str],
    subset_names: Tuple[str, str],
    desired_split_ratio: float,
) -> SimpleNamespace:
    """
    Return a split of the frames into two subsets based on the specified ratio.

    The split is frame-based, but the ratio applies to annotation counts.
    It ensures that the ratio of annotations with the specified labels in each
    subset is approximately equal to the given split ratio.

    ### **Implementation Notes**
    - **Enhanced Min-Heap Balancing** prioritizes per-label ratio deviations.
    - **Frames are assigned dynamically** based on the label needing correction the most.
    - This ensures per-label ratios are closer to `desired_split_ratio`.
    """
    # Count annotations for each label across all frames
    count_result = _counts_labels(dataset_manager, frame_ids, annotation_labels)
    frames_labels_counts = count_result.frames_labels_counts
    total_labels_counts = count_result.total_labels_counts

    # Compute desired count per label
    desired_split_count = {
        label: int(count * desired_split_ratio) for label, count in total_labels_counts.items()
    }

    # Assign frames to subset to achieve the desired ratio
    # desired_split_ratio:       corresponds to the first subset in subset_names
    # (1 - desired_split_ratio): corresponds to the second subset
    random.shuffle(frame_ids)
    subsets: Dict[str, list] = {n: [] for n in subset_names}
    subsets_counts: Dict[str, Counter[str]] = {n: Counter() for n in subset_names}
    heap = _build_heap(annotation_labels, subsets_counts, desired_split_count, subset_names)
    for frame_id in frame_ids:
        _, most_imbalanced_label = heapq.heappop(heap)
        subset_name = _choose_subset(
            most_imbalanced_label,
            subsets_counts,
            desired_split_count,
            total_labels_counts,
            subset_names,
        )
        subsets[subset_name].append(frame_id)
        subsets_counts[subset_name].update(frames_labels_counts[frame_id])
        heap = _build_heap(annotation_labels, subsets_counts, desired_split_count, subset_names)

    _verify_split(subsets_counts, annotation_labels, subset_names)

    return SimpleNamespace(
        **subsets,
        desired_split_ratio=desired_split_ratio,
        stats=_data_split_stats(subsets_counts, total_labels_counts),
    )
