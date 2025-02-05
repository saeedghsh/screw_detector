"""This module is the entry point for splitting dataset into train and test and
store the result into a file for consistent evaluation."""

# pylint: disable=no-member
import os
import sys
from datetime import datetime
from typing import Sequence

from libs.config_reader import load_config
from libs.dataset.manager import DatasetManager
from libs.dataset.split import data_split_cache_path, save_split, split
from libs.logger import setup_logging

logger = setup_logging(name_appendix="data-splitter")


def main(_: Sequence[str]) -> int:  # pragma: no cover
    """Main entry point for splitting and caching the dataset."""
    dataset_split_config = load_config("dataset_split")
    logger.info(
        "Splitting dataset into train and test sets with split ratio %.2f",
        dataset_split_config["test_split_ratio"],
    )

    annotation_labels = {"screw_head", "screw_hole"}
    dataset_manager = DatasetManager()
    split_result = split(
        dataset_manager,
        frame_ids=list(dataset_manager.frame_ids.keys()),
        annotation_labels=annotation_labels,
        subset_names=("test", "train"),
        desired_split_ratio=dataset_split_config["test_split_ratio"],
    )
    for label in annotation_labels:
        logger.info(
            "Counts of label %s: train=%d, test=%d",
            label,
            split_result.stats["count"][label]["train"],
            split_result.stats["count"][label]["test"],
        )

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    cache_dir_path = data_split_cache_path(ensure_exist=True)
    split_file = os.path.join(cache_dir_path, f"{timestamp}.json")
    save_split(split_result, split_file)
    return os.EX_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
