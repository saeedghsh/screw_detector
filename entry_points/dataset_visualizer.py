"""This module is the entry point for the data inspector application."""

# pylint: disable=no-member

import logging
import os
import sys
from types import SimpleNamespace
from typing import Sequence

from libs.dataset_management import BATTERY_PACKS, DatasetManager
from libs.logger import setup_logging
from libs.visualization import Visualizer

logger = setup_logging(name_appendix="data-inspector", level=logging.DEBUG)


def _config() -> SimpleNamespace:
    config = SimpleNamespace(
        battery_pack=2,
        image_resize_factor=0.5,
        visualize_2d=True,
        visualize_3d=False,
        show_output=False,
        save_output=True,
        output_dir="output",
    )
    if config.battery_pack not in BATTERY_PACKS:  # pragma: no cover
        raise ValueError(f"Battery pack must be one of {BATTERY_PACKS}")
    return config


def main(_: Sequence[str]) -> int:
    """Main entry point for the data inspector application."""

    config = _config()
    dataset_manager = DatasetManager(config.battery_pack)
    visualizer = Visualizer(config, dataset_manager.label_name_mapper)

    for frame_id in dataset_manager.frame_ids.keys():
        frame = dataset_manager.frame(frame_id)
        visualizer.visualize_frame(frame)

    return os.EX_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
