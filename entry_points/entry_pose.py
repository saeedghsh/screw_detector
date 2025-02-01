"""Entry point for the 2D detection and 3D pose estimation."""

import argparse
import os
import sys
from typing import Sequence

from libs.config_reader import load_config
from libs.dataset.data_structure import Frame
from libs.dataset.manager import DATASET_PATH, DatasetManager
from libs.dataset.utils import (
    load_cached_split,
    load_camera_transforms,
    load_images,
    load_pointclouds,
)
from libs.detection.detector_2d import Detector2D
from libs.detection.hough_circle_detector import HoughCircleDetector
from libs.logger import setup_logging
from libs.pose_estimate import PointCloudProcessor, PoseEstimator
from libs.visualization import visualize_detections_3d

logger = setup_logging(name_appendix="3D pose estimation")


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


def _handle_dataset_mode(
    args: argparse.Namespace,
    pose_estimator: PoseEstimator,
    detector_2d: Detector2D,
    pointcloud_processor: PointCloudProcessor,
):
    logger.info("Instantiating DatasetManager\n")
    dataset_manager = DatasetManager()

    logger.info("Loading cached split from: %s\n", args.cached_split_path)
    cached_split = load_cached_split(args.cached_split_path)

    for frame_id in cached_split.test_frame_ids:
        logger.info("Processing frame: %s\n", frame_id)
        frame = dataset_manager.frame(frame_id)
        frame.detections = detector_2d.detect(frame.image)
        processed_pcd, global_indices = pointcloud_processor.processes(frame.pointcloud)
        frame.detections_3d = pose_estimator.find_points_for_detections(
            pcd=processed_pcd,
            frame=frame,
            selected_indices=global_indices,
            transform_to_robot_frame=pose_estimator.configuration["transform_to_robot_frame"],
        )
        visualize_detections_3d(pointcloud=processed_pcd, frame=frame)


def _handle_direct_mode(
    args: argparse.Namespace,
    pose_estimator: PoseEstimator,
    detector_2d: Detector2D,
    pointcloud_processor: PointCloudProcessor,
):
    images = load_images(args.input_path)
    pointclouds = load_pointclouds(args.input_path)
    camera_transforms = load_camera_transforms(args.input_path)
    for frame_id, image in images.items():
        logger.info("Processing frame: %s\n", frame_id)
        frame = Frame(image=image, id=frame_id)
        frame.pointcloud = pointclouds[frame_id]
        frame.camera_transform = camera_transforms[frame_id]
        frame.detections = detector_2d.detect(frame.image)
        processed_pcd, global_indices = pointcloud_processor.processes(frame.pointcloud)
        frame.detections_3d = pose_estimator.find_points_for_detections(
            pcd=processed_pcd,
            frame=frame,
            selected_indices=global_indices,
            transform_to_robot_frame=pose_estimator.configuration["transform_to_robot_frame"],
        )
        visualize_detections_3d(pointcloud=processed_pcd, frame=frame)


def main(argv: Sequence[str]) -> int:
    # pylint: disable=missing-function-docstring
    # pragma: no cover
    args = _parse_args(argv)
    logger.info("Entry point args:\n%s\n", args)

    pose_estimator_config = load_config("pose_estimator")
    logger.info("Instantiating PoseEstimator:\n%s\n", pose_estimator_config)
    pose_estimator = PoseEstimator(pose_estimator_config)

    hough_circle_detector_config = load_config("hough_circle_detector")
    logger.info("Instantiating HoughCircleDetector:\n%s\n", hough_circle_detector_config)
    detector_2d = HoughCircleDetector(hough_circle_detector_config)

    pointcloud_processor_config = load_config("pointcloud_processor")
    logger.info("Instantiating PointCloudProcessor:\n%s\n", pointcloud_processor_config)
    pointcloud_processor = PointCloudProcessor(pointcloud_processor_config)

    args.func(args, pose_estimator, detector_2d, pointcloud_processor)

    return os.EX_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
