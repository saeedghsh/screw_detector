"""
This module provides a Visualizer class for visualizing images and point clouds
with annotations from a dataset. The Visualizer supports drawing annotations as
bounding boxes or mask contours on images, and it can display the annotated
images using OpenCV.
"""

# pylint: disable=no-member

import copy
from types import SimpleNamespace
from typing import Any, Callable, List, Optional, Tuple, cast

import cv2
import numpy as np
import open3d as o3d
from datumaro.components.annotation import Annotations

from libs.dataset.data_structure import BoundingBox, Detection2D, Frame


def _colors(idx: Optional[int], clip_to_unit: bool = False) -> Tuple[int, int, int]:
    """Return a list of colors for drawing annotations.
    If the index is not provided, return the last color in the list."""
    colors = {
        "Blue": (255, 0, 0),  # Blue
        "Green": (0, 255, 0),  # Green
        "Red": (0, 0, 255),  # Red
        "Cyan": (255, 255, 0),  # Cyan
        "Yellow": (0, 255, 255),  # Yellow
        "Black": (0, 0, 0),  # Black
        "White": (255, 255, 255),  # White
        "Magenta": (255, 0, 255),  # Magenta
    }
    i = idx % len(colors) if idx is not None else -1
    color: Tuple[int, int, int] = list(colors.values())[i]
    if clip_to_unit:  # pragma: no cover
        # Explicit cast to suppress mypy error
        color = cast(Tuple[int, int, int], tuple(int(c / 255) for c in color))
    return color


def _text_font_args() -> dict:
    """Return the configuration for the visualizer."""
    return {"fontFace": cv2.FONT_HERSHEY_SIMPLEX, "fontScale": 0.5, "thickness": 1}


def _write_label_with_prefix(
    annotated_image: np.ndarray,
    prefix: str,
    label_name: str,
    label_coordinates: SimpleNamespace,
    color: Tuple[int, int, int],
):
    """Write the prefixed label name on the image."""
    full_label = f"[{prefix}]-{label_name}"
    org = (label_coordinates.x, label_coordinates.y - 10)
    cv2.putText(img=annotated_image, text=full_label, org=org, color=color, **_text_font_args())


def _draw_bbox(
    annotated_image: np.ndarray, bbox: BoundingBox, color: Tuple[int, int, int], filled: bool
):
    """Draw a bounding box with either filled transparency or empty rectangle style."""
    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
    if filled:
        overlay = annotated_image.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness=cv2.FILLED)
        cv2.addWeighted(overlay, 0.3, annotated_image, 0.7, 0, annotated_image)
    else:
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)


def _draw_with_custom_camera_view(geometries: List[Any]):
    """Draw geometries with a custom camera view."""
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geom in geometries:
        vis.add_geometry(geom)
    # Set custom view aligned with camera Z-axis
    view_control = vis.get_view_control()
    camera = view_control.convert_to_pinhole_camera_parameters()
    # Set eye position slightly behind the origin, looking at [0, 0, 0], with Z-axis as up direction
    camera.extrinsic = np.array(
        [
            [1, 0, 0, 0],  # X-axis
            [0, 1, 0, 0],  # Y-axis
            [0, 0, 1, +500],  # Z-axis, set 500 mm behind the origin
            [0, 0, 0, 1],
        ]
    )
    # Apply the new camera parameters
    view_control.convert_from_pinhole_camera_parameters(camera)
    # Run visualization
    vis.run()
    vis.destroy_window()


def _custom_camera_frame() -> o3d.geometry.LineSet:
    """Create a custom camera frame with X, Y, and Z axes."""
    coord_frame_size = 1000.0  # mm
    axis_points = [
        [0, 0, 0],
        [coord_frame_size, 0, 0],
        [0, 0, 0],
        [0, coord_frame_size, 0],
        [0, 0, 0],
        [0, 0, coord_frame_size],
    ]
    axis_lines = [[0, 1], [2, 3], [4, 5]]
    axis_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(axis_points)
    axis.lines = o3d.utility.Vector2iVector(axis_lines)
    axis.colors = o3d.utility.Vector3dVector(axis_colors)
    return axis


def _custom_coordinate_frame(
    translation: np.ndarray, quaternion: np.ndarray
) -> o3d.geometry.LineSet:
    """Create a coordinate frame at the given pose (translation + orientation)
    using a LineSet."""
    coord_frame_size = 100.0  # mm
    axis_points = [
        [0, 0, 0],  # Origin
        [coord_frame_size, 0, 0],  # X-axis endpoint
        [0, coord_frame_size, 0],  # Y-axis endpoint
        [0, 0, coord_frame_size],  # Z-axis endpoint
    ]
    axis_lines = [[0, 1], [0, 2], [0, 3]]  # Connect origin to each axis
    axis_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # RGB for X, Y, Z
    # Create a LineSet for the axes
    coord_frame = o3d.geometry.LineSet()
    coord_frame.points = o3d.utility.Vector3dVector(axis_points)
    coord_frame.lines = o3d.utility.Vector2iVector(axis_lines)
    coord_frame.colors = o3d.utility.Vector3dVector(axis_colors)
    # Apply rotation and translation
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
    transformation = np.eye(4)
    transformation[:3, :3] = rotation_matrix
    transformation[:3, 3] = translation
    coord_frame.transform(transformation)
    return coord_frame


def visualize_detections_3d(pointcloud: o3d.geometry.PointCloud, frame: Frame):
    """Visualize the point cloud with:
    * detection poses represented as coordinate frames
    * detection bounding boxes
    """
    pcd_display = copy.deepcopy(pointcloud)
    geometries = [pcd_display]
    if frame.detections_3d is not None:
        for i, detections_3d in enumerate(frame.detections_3d):
            # draw bounding box
            points_array = detections_3d.points_3d
            if points_array is not None and len(points_array) != 0:
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(points_array)
                bbox = cloud.get_axis_aligned_bounding_box()
                bbox.color = _colors(i, clip_to_unit=True)
                geometries.append(bbox)
            # draw coordinate frame
            pose = detections_3d.pose_3d
            if pose is not None:
                coord_frame = _custom_coordinate_frame(pose.translation, pose.quaternion)
                geometries.append(coord_frame)
    camera_frame = _custom_camera_frame()
    geometries.append(camera_frame)
    _draw_with_custom_camera_view(geometries)


class Visualizer:  # pylint: disable=too-few-public-methods
    """Visualize images and point clouds from the dataset."""

    def __init__(
        self,
        config: dict,
        annotation_label_mapper: Callable = str,
        detection_label_mapper: Callable = str,
    ):
        self._config = config
        self._annotation_label_mapper = annotation_label_mapper
        self._detection_label_mapper = detection_label_mapper

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize the image according to the configuration."""
        if self._config["image_resize_factor"] != 1.0:
            return cv2.resize(
                image,
                (0, 0),
                fx=self._config["image_resize_factor"],
                fy=self._config["image_resize_factor"],
            )
        return image

    def _draw_annotations(self, annotated_image: np.ndarray, annotations: Annotations):
        """Draw annotations as filled rectangles"""
        for annotation in annotations:
            color = _colors(annotation.label)
            label_name = self._annotation_label_mapper(annotation.label)
            bbox = BoundingBox(*annotation.get_bbox()).resize(self._config["image_resize_factor"])
            _draw_bbox(annotated_image, bbox, color, filled=True)
            label_coordinates = SimpleNamespace(x=bbox.x, y=bbox.y)
            _write_label_with_prefix(annotated_image, "A", label_name, label_coordinates, color)

    def _draw_detections(self, annotated_image: np.ndarray, detections: List[Detection2D]):
        """Draw detections as empty rectangles."""
        for detection in detections:
            color = _colors(detection.label)
            label_name = self._detection_label_mapper(detection.label)
            bbox = BoundingBox(*detection.get_bbox()).resize(self._config["image_resize_factor"])
            _draw_bbox(annotated_image, bbox, color, filled=False)
            label_coordinates = SimpleNamespace(x=bbox.x, y=bbox.y)
            _write_label_with_prefix(annotated_image, "D", label_name, label_coordinates, color)

    def _visualize_3d(self, frame: Frame):
        """Visualize a point cloud."""
        if self._config["show_output"]:
            o3d.visualization.draw_geometries([frame.pointcloud], window_name="Point Cloud")

    def _visualize_2d(self, frame: Frame):
        """Visualize a frame with its image and annotations."""
        annotated_image = self._resize_image(frame.image.copy())
        if frame.annotations is not None:
            self._draw_annotations(annotated_image, frame.annotations)
        if frame.detections is not None:
            self._draw_detections(annotated_image, frame.detections)

        if self._config["show_output"]:
            cv2.imshow("Annotated Image", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if self._config["save_output"]:
            output_path_2d = f"{frame.file_name_from_id()}_2d.png"
            output_path = f"{self._config['output_dir']}/{output_path_2d}"
            cv2.imwrite(output_path, annotated_image)

    def visualize_frame(self, frame: Frame):
        """Visualize a frame with its image, point cloud, and annotations."""
        if self._config["visualize_3d"]:
            self._visualize_3d(frame)
        if self._config["visualize_2d"]:
            self._visualize_2d(frame)
