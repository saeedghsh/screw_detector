"""This module is the entry point for splitting dataset into train and test and
store the result into a file for consistent evaluation."""

# pylint: disable=no-member
import os
import sys
from typing import Sequence

from libs.config_reader import load_config
from libs.dataset.manager import DatasetManager
from libs.dataset.split import cache_split
from libs.logger import setup_logging

logger = setup_logging(name_appendix="data-splitter")


def main(_: Sequence[str]) -> int:  # pragma: no cover
    """Main entry point for splitting and caching the dataset."""
    dataset_manager = DatasetManager()
    dataset_split_config = load_config("dataset_split")
    cache_split(dataset_manager, dataset_split_config["test_split_ratio"])
    return os.EX_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
