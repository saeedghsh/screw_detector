"""Entry point for the detector module."""

# pylint: disable=no-member, missing-function-docstring
import argparse
import os
import sys
from typing import Sequence

from libs.config_reader import load_config
from libs.dataset.data_structure import Frame
from libs.dataset.manager import DATASET_PATH, DatasetManager
from libs.dataset.utils import load_cached_split, load_images
from libs.detection.detector import Detection, Detector
from libs.detection.hough_circle_detector import HoughCircleDetector
from libs.logger import setup_logging
from libs.visualization import Visualizer

logger = setup_logging(name_appendix="Detector")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect screws in images or dataset.")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation")

    # Dataset mode parser
    dataset_parser = subparsers.add_parser("dataset", help="Use dataset manager and cached split.")
    dataset_parser.add_argument(
        "--cached-split-path",
        type=str,
        default=f"{DATASET_PATH}/data_split_cache/20250112T232216_0.2_split.json",
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


def _handle_dataset_mode(args: argparse.Namespace, detector: Detector):
    dataset_manager = DatasetManager()
    cached_split = load_cached_split(args.cached_split_path)
    visualizer = Visualizer(
        config=load_config("visualizer"),
        annotation_label_mapper=dataset_manager.label_name_mapper,
        detection_label_mapper=Detection.label_name_mapper,
    )
    for frame_id in cached_split.test_frame_ids:
        frame = dataset_manager.frame(frame_id)
        frame.detections = detector.detect(frame.image)
        visualizer.visualize_frame(frame)


def _handle_direct_mode(args: argparse.Namespace, detector: Detector):
    images = load_images(args.input_path)
    visualizer = Visualizer(
        config=load_config("visualizer"),
        detection_label_mapper=Detection.label_name_mapper,
    )
    frames = [Frame(image=image) for image in images]
    for frame in frames:
        detections = detector.detect(frame.image)
        frame.detections = detections
        visualizer.visualize_frame(frame)


def main(argv: Sequence[str]) -> int:
    args = _parse_args(argv)
    hough_circle_detector_config = load_config("hough_circle_detector")
    circle_detector = HoughCircleDetector(hough_circle_detector_config)
    args.func(args, circle_detector)
    return os.EX_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
