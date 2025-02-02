"""This module is the entry point for the data inspector application."""

# pylint: disable=no-member

import argparse
import os
import sys
from typing import Sequence

from libs.config_reader import load_config
from libs.dataset.data_reader import load_camera_transforms, load_images, load_pointclouds
from libs.dataset.data_structure import Frame
from libs.dataset.manager import DatasetManager
from libs.dataset.stats import dataset_stats
from libs.logger import setup_logging
from libs.visualization import Visualizer

logger = setup_logging(name_appendix="Visualizer")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic Visualizer.")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation")

    # Dataset mode parser
    dataset_parser = subparsers.add_parser("dataset", help="Use dataset manager and cached split.")
    dataset_parser.set_defaults(func=_handle_dataset_mode)

    # Direct mode parser
    direct_parser = subparsers.add_parser("direct", help="Direct image or directory input.")
    direct_parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to input image or directory.",
    )
    direct_parser.set_defaults(func=_handle_direct_mode)

    args = parser.parse_args(argv)
    return args


def _handle_dataset_mode(_: argparse.Namespace):
    logger.info("Instantiating DatasetManager\n")
    dataset_manager = DatasetManager()
    dataset_stats(dataset_manager, logger)

    visualizer_config = load_config("visualizer")
    visualizer = Visualizer(visualizer_config, dataset_manager.label_name_mapper)

    for frame_id in dataset_manager.frame_ids.keys():
        frame = dataset_manager.frame(frame_id)
        visualizer.visualize_frame(frame)


def _handle_direct_mode(
    args: argparse.Namespace,
):
    images = load_images(args.input_path)
    pointclouds = load_pointclouds(args.input_path)
    camera_transforms = load_camera_transforms(args.input_path)
    visualizer = Visualizer(config=load_config("visualizer"))

    for frame_id, image in images.items():
        logger.info("Processing frame: %s\n", frame_id)
        frame = Frame(image=image, id=frame_id)
        frame.pointcloud = pointclouds.get(frame_id, None)
        frame.camera_transform = camera_transforms.get(frame_id, None)
        visualizer.visualize_frame(frame)


def main(argv: Sequence[str]) -> int:
    # pylint: disable=missing-function-docstring
    # pragma: no cover
    args = _parse_args(argv)
    logger.info("Entry point args:\n%s\n", args)
    args.func(args)
    return os.EX_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
