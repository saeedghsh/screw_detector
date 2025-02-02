"""This module provides functions for loading images, camera transforms, and
point without using the dataset manager."""

import glob
import json
import os

# pylint: disable=no-member
from typing import Dict

import cv2
import numpy as np
import open3d as o3d


def _fake_frame_id_single_file(input_path) -> str:
    return os.path.splitext(os.path.basename(input_path))[0]


def _fake_frame_id_multi_file(relative_path) -> str:
    return os.path.splitext(relative_path)[0].replace(os.sep, "_")


def load_images(input_path: str) -> Dict[str, np.ndarray]:
    """
    Load images from a single file or a directory and return them as a
    dictionary. The keys are derived from the image paths relative to the
    input_path, with file extensions removed and path separators replaced by
    underscores (faking frame_id).

    NOTE: this is used for loading camera transforms in the direct mode of the entry points.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path '{input_path}' does not exist.")

    images = {}

    if os.path.isfile(input_path):
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Failed to load image from '{input_path}'.")

        key = _fake_frame_id_single_file(input_path)
        images[key] = image

    elif os.path.isdir(input_path):
        for file_path in sorted(glob.glob(os.path.join(input_path, "**", "*.png"), recursive=True)):
            image = cv2.imread(file_path)
            if image is not None:
                # Create key by removing input_path, stripping extension, and replacing separators
                relative_path = os.path.relpath(file_path, input_path)
                key = _fake_frame_id_multi_file(relative_path)
                images[key] = image
            else:
                raise ValueError(f"Failed to load image from '{file_path}'.")  # pragma: no cover
    else:
        raise ValueError(f"Input path '{input_path}' is neither a file nor a directory.")

    if not images:
        raise FileNotFoundError(f"No images found in '{input_path}'.")

    return images


def load_camera_transforms(input_path: str) -> Dict[str, np.ndarray]:
    """
    Load camera transforms from a single JSON file or a directory and return them
    as a dictionary. The keys are derived from the JSON file paths relative to the
    input_path, with file extensions removed and path separators replaced by underscores.

    Each file must contain a 4x4 transformation matrix stored in JSON format.

    NOTE: this is used for loading camera transforms in the direct mode of the entry points.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path '{input_path}' does not exist.")

    transforms = {}

    if os.path.isfile(input_path):
        with open(input_path, "r", encoding="utf-8") as f:
            matrix = np.array(json.load(f), dtype=np.float64)
            if matrix.shape != (4, 4):
                raise ValueError(f"Invalid 4x4 matrix in '{input_path}'.")

        key = _fake_frame_id_single_file(input_path)
        transforms[key] = matrix

    elif os.path.isdir(input_path):
        for file_path in sorted(
            glob.glob(os.path.join(input_path, "**", "*.json"), recursive=True)
        ):
            with open(file_path, "r", encoding="utf-8") as f:
                matrix = np.array(json.load(f), dtype=np.float64)
                if matrix.shape != (4, 4):
                    raise ValueError(f"Invalid 4x4 matrix in '{file_path}'.")  # pragma: no cover

            relative_path = os.path.relpath(file_path, input_path)
            key = _fake_frame_id_multi_file(relative_path)
            transforms[key] = matrix
    else:
        raise ValueError(f"Input path '{input_path}' is neither a file nor a directory.")

    if not transforms:
        raise FileNotFoundError(f"No camera transforms found in '{input_path}'.")

    return transforms


def load_pointclouds(input_path: str) -> Dict[str, o3d.geometry.PointCloud]:
    """
    Load point clouds from a single PLY file or a directory and return them as a
    dictionary. The keys are derived from the PLY file paths relative to the
    input_path, with file extensions removed and path separators replaced by underscores.

    NOTE: this is used for loading camera transforms in the direct mode of the entry points.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path '{input_path}' does not exist.")

    pointclouds = {}

    if os.path.isfile(input_path):
        pcd = o3d.io.read_point_cloud(input_path)
        if pcd.is_empty():
            raise ValueError(f"Failed to load valid point cloud from '{input_path}'.")

        key = _fake_frame_id_single_file(input_path)
        pointclouds[key] = pcd

    elif os.path.isdir(input_path):
        for file_path in sorted(glob.glob(os.path.join(input_path, "**", "*.ply"), recursive=True)):
            pcd = o3d.io.read_point_cloud(file_path)
            if not pcd.is_empty():
                relative_path = os.path.relpath(file_path, input_path)
                key = _fake_frame_id_multi_file(relative_path)
                pointclouds[key] = pcd
            else:
                raise ValueError(
                    f"Failed to load valid point cloud from '{file_path}'."
                )  # pragma: no cover
    else:
        raise ValueError(f"Input path '{input_path}' is neither a file nor a directory.")

    if not pointclouds:
        raise FileNotFoundError(f"No point clouds found in '{input_path}'.")

    return pointclouds
