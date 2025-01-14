"""This module is the entry point for the data inspector application."""

# pylint: disable=no-member

import logging
import os
import sys
from typing import Sequence

from libs.config_reader import load_config
from libs.dataset.manager import DatasetManager
from libs.dataset.utils import dataset_stats
from libs.logger import setup_logging
from libs.visualization import Visualizer

logger = setup_logging(name_appendix="Visualizer", level=logging.DEBUG)


def main(_: Sequence[str]) -> int:
    """Main entry point for the data inspector application."""
    dataset_manager = DatasetManager()
    dataset_stats(dataset_manager, logger)

    visualizer_config = load_config("visualizer")
    visualizer = Visualizer(visualizer_config, dataset_manager.label_name_mapper)

    for frame_id in dataset_manager.frame_ids.keys():
        frame = dataset_manager.frame(frame_id)
        visualizer.visualize_frame(frame)

    return os.EX_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
