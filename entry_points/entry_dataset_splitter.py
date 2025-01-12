"""This module is the entry point for splitting dataset into train and test and
store the result into a file for consistent evaluation."""

# pylint: disable=no-member
# pragma: no cover
import os
import sys
from typing import Sequence

from libs.dataset.manager import DatasetManager
from libs.dataset.utils import cache_split
from libs.logger import setup_logging

logger = setup_logging(name_appendix="data-splitter")


def main(_: Sequence[str]) -> int:
    """Main entry point for splitting and caching the dataset."""
    dataset_manager = DatasetManager()
    cache_split(dataset_manager, 0.2)
    return os.EX_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
