"""This module is the entry point for the data inspector application."""

# pylint: disable=no-member

import logging
import os
import sys
from types import SimpleNamespace
from typing import Sequence

from libs.dataset_management import DatasetManager, dataset_stats, split_train_test
from libs.logger import setup_logging
from libs.visualization import Visualizer

logger = setup_logging(name_appendix="data-inspector", level=logging.DEBUG)


CONFIG = SimpleNamespace(
    image_resize_factor=0.5,
    visualize_2d=False,
    visualize_3d=False,
    show_output=False,
    save_output=False,
    output_dir="output",
)


def main(_: Sequence[str]) -> int:
    """Main entry point for the data inspector application."""
    dataset_manager = DatasetManager()
    dataset_stats(dataset_manager, logger)

    __ = split_train_test(dataset_manager, test_ratio=0.2)

    visualizer = Visualizer(CONFIG, dataset_manager.label_name_mapper)

    for frame_id in dataset_manager.frame_ids.keys():
        frame = dataset_manager.frame(frame_id)
        visualizer.visualize_frame(frame)
    return os.EX_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
