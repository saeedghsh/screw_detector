"""
This module provides a Visualizer class for visualizing images and point clouds
with annotations from a dataset. The Visualizer supports drawing annotations as
bounding boxes or mask contours on images, and it can display the annotated
images using OpenCV.
"""

# pylint: disable=no-member

from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
from datumaro.components.annotation import Annotations

from libs.dataset.data_structure import BoundingBox, Frame
from libs.detection.detector import Detection


def _colors(idx: Optional[int]) -> Tuple[int, int, int]:
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
    return list(colors.values())[i]


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

    def _draw_detections(self, annotated_image: np.ndarray, detections: List[Detection]):
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
