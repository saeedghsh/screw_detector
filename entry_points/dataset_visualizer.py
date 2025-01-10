"""This module is the entry point for the data inspector application."""

# pylint: disable=no-member

import argparse
import logging
import os
import sys
from typing import Sequence

from libs.dataset_management import DatasetManager
from libs.logger import setup_logging
from libs.visualization import ANNOTATION_DRAW_MODE, Visualizer, VisualizerConfig

logger = setup_logging(name_appendix="data-inspector", level=logging.DEBUG)


def _str_to_bool(s: str) -> bool:
    """Convert a string to a boolean."""
    return s.lower() == "true"


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Data Inspector Application")
    parser.add_argument("--battery-pack", default=2, type=int, choices=[1, 2])
    parser.add_argument("--image-resize-factor", default=0.5, type=float)
    parser.add_argument(
        "--draw-annotation-as", default="mask_contour", type=str, choices=ANNOTATION_DRAW_MODE
    )
    parser.add_argument("--visualize-2d", type=_str_to_bool, default=True, help="Visualize 2D data")
    parser.add_argument("--visualize-3d", type=_str_to_bool, default=True, help="Visualize 3D data")
    parsed_args = parser.parse_args(argv)
    logger.info("Data inspector started with arguments: %s", parsed_args)
    return parsed_args


def main(argv: Sequence[str]) -> int:
    """Main entry point for the data inspector application."""
    parsed_args = _parse_args(argv)

    dataset_manager = DatasetManager(parsed_args.battery_pack)
    visualizer_config = VisualizerConfig(
        parsed_args.image_resize_factor,
        parsed_args.draw_annotation_as,
        parsed_args.visualize_2d,
        parsed_args.visualize_3d,
    )
    visualizer = Visualizer(visualizer_config, dataset_manager.label_name_mapper)

    for frame_id in dataset_manager.frame_ids.keys():
        image = dataset_manager.image(frame_id)
        pointcloud = dataset_manager.pointcloud(frame_id)
        visualizer.visualize_frame(
            image,
            pointcloud,
            dataset_manager.frame_annotations(frame_id),
        )

    return os.EX_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
