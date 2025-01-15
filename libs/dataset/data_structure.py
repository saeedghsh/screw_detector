"""Data structures for the dataset."""

from dataclasses import dataclass

# pylint: disable=no-member
from typing import Any, List, Optional, Tuple

import datumaro
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


class Detection2D:
    """A simple container for detection results."""

    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        label: int,
        confidence: float = 1.0,
    ):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.confidence = confidence

    def __repr__(self):
        return (
            f"Detection2D(label={self.label}, x={self.x}, y={self.y}, "
            f"width={self.width}, height={self.height}, confidence={self.confidence})"
        )

    def get_bbox(self) -> Any:
        """Return the bounding box as a tuple (x, y, w, h)."""
        return self.x, self.y, self.width, self.height

    @staticmethod
    def label_name_mapper(label: int) -> str:
        """Map a label (index) to a human-readable name."""
        label_names = ["screw_head", "screw_hole"]
        if label >= len(label_names) or label < 0:
            return "UNKNOWN"
        return label_names[label]


@dataclass
class Pose3D:
    """
    6D pose in 3D space, represented by a 3D translation vector and
    a quaternion (x, y, z, w).

    Note:
        - 'transform()' re-expresses this pose in another coordinate frame,
          given a 4x4 transform that goes from the current frame to the new frame.
        - In an ideal scenario, a dual/quaternion approach might handle
          rotation + translation more rigorously, but this single quaternion
          is acceptable for most applications.
    """

    translation: np.ndarray  # shape (3,)
    quaternion: np.ndarray  # shape (4,) in [x, y, z, w] format

    def transform(self, transform_matrix: np.ndarray) -> "Pose3D":
        """
        Re-express this pose in a new frame using the provided 4x4 transform.

        Args:
            T (np.ndarray): 4x4 transformation matrix that converts a pose
                            from the *old* frame to the *new* frame:
                            new_pose = T * old_pose.

        Returns:
            A new Pose3D in the updated frame.
        """
        rot_mat = transform_matrix[0:3, 0:3]  # 3×3 rotation
        trans_vec = transform_matrix[0:3, 3]  # 3×1 translation
        new_translation = rot_mat @ self.translation + trans_vec
        old_rot = Rotation.from_quat(self.quaternion)  # convert to scipy Rotation
        new_rot_mat = rot_mat @ old_rot.as_matrix()
        new_quat = Rotation.from_matrix(new_rot_mat).as_quat()  # returns [x, y, z, w]
        return Pose3D(translation=new_translation, quaternion=new_quat)


@dataclass
class Detection3D:
    """
    Represents a single detection in 3D space, derived from a 2D bounding box.

    Attributes:
        detection_2d: Reference to the 2D detection (bounding box, etc.).
        points_3d:    Nx3 array of all 3D points (in the processed point cloud)
                      that fall within the detection_2d bounding box.
        centroid_3d:  The mean (x, y, z) of points_3d.
        pose_3d:      6D pose (e.g., [x, y, z, roll, pitch, yaw]) for the detected object.
    """

    detection_2d: Detection2D
    points_3d: Optional[np.ndarray] = None
    centroid_3d: Optional[np.ndarray] = None
    pose_3d: Optional[Pose3D] = None


@dataclass
class Frame:  # pylint: disable=missing-class-docstring
    image: np.ndarray
    id: str
    pointcloud: Optional[o3d.geometry.PointCloud] = None
    annotations: Optional[datumaro.components.annotation.Annotations] = None
    detections: Optional[List[Detection2D]] = None
    detections_3d: Optional[List[Detection3D]] = None
    camera_transform: Optional[np.ndarray] = None

    def file_name_from_id(self) -> str:
        """Return the file name from the frame ID."""
        return self.id.replace("/", "_") if self.id else ""

    def annotations_count(self) -> int:  # pragma: no cover
        """Return the number of annotations."""
        return len(self.annotations) if self.annotations else 0

    def detections_count(self) -> int:  # pragma: no cover
        """Return the number of detections."""
        return len(self.detections) if self.detections else 0


@dataclass
class BoundingBox:  # pylint: disable=missing-class-docstring, missing-function-docstring
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return int(self.x), int(self.y), int(self.w), int(self.h)

    def resize(self, factor: float) -> "BoundingBox":
        return BoundingBox(
            int(self.x * factor),
            int(self.y * factor),
            int(self.w * factor),
            int(self.h * factor),
        )
