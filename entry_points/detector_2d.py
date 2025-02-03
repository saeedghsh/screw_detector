"""Entry point for the detector module."""

import argparse
import os
import sys
from typing import Sequence

from libs.config_reader import load_config
from libs.dataset.data_reader import load_images
from libs.dataset.data_structure import Frame
from libs.dataset.manager import DATASET_PATH, DatasetManager
from libs.dataset.split import load_cached_split
from libs.detection.detector_2d import Detection2D, Detector2D
from libs.detection.hough_circle_detector import HoughCircleDetector
from libs.logger import setup_logging
from libs.visualization import Visualizer

logger = setup_logging(name_appendix="Detector2D")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect screws in images or dataset.")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation")

    # Dataset mode parser
    dataset_parser = subparsers.add_parser("dataset", help="Use dataset manager and cached split.")
    dataset_parser.add_argument(
        "--cached-split-path",
        type=str,
        default=f"{DATASET_PATH}/data_split_cache/20250203T021622.json",
        help="Path to cached split.",
    )
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


def _handle_dataset_mode(args: argparse.Namespace, detector: Detector2D):
    dataset_manager = DatasetManager()
    cached_split = load_cached_split(args.cached_split_path)
    visualizer = Visualizer(
        config=load_config("visualizer"),
        annotation_label_mapper=dataset_manager.label_name_mapper,
        detection_label_mapper=Detection2D.label_name_mapper,
    )
    for frame_id in cached_split.test_frame_ids:
        frame = dataset_manager.frame(frame_id)
        frame.detections = detector.detect(frame.image)
        visualizer.visualize_frame(frame)


def _handle_direct_mode(args: argparse.Namespace, detector: Detector2D):
    visualizer = Visualizer(
        config=load_config("visualizer"),
        detection_label_mapper=Detection2D.label_name_mapper,
    )
    images = load_images(args.input_path)
    frames = [Frame(image=image, id=key) for key, image in images.items()]
    for frame in frames:
        detections = detector.detect(frame.image)
        frame.detections = detections
        visualizer.visualize_frame(frame)


def main(argv: Sequence[str]) -> int:  # pylint: disable=missing-function-docstring
    args = _parse_args(argv)
    hough_circle_detector_config = load_config("hough_circle_detector")
    circle_detector = HoughCircleDetector(hough_circle_detector_config)
    args.func(args, circle_detector)
    return os.EX_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
