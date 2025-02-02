"""This module provides functions for splitting a dataset into train, validation
and test sets with custom criteria."""

import json
import os
import random
from collections import Counter
from datetime import datetime
from types import SimpleNamespace

from libs.dataset.manager import DATASET_PATH, DatasetManager

CACHE_DIR = os.path.join(DATASET_PATH, "data_split_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


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
