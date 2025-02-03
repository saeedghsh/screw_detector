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
    dataset_manager = DatasetManager()
    split_result = split(
        dataset_manager,
        frame_ids=list(dataset_manager.frame_ids.keys()),
        annotation_labels={"screw_head", "screw_hole"},
        subset_names=("test", "train"),
        desired_split_ratio=dataset_split_config["test_split_ratio"],
    )
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    cache_dir_path = data_split_cache_path(ensure_exist=True)
    split_file = os.path.join(cache_dir_path, f"{timestamp}.json")
    save_split(split_result, split_file)
    return os.EX_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
