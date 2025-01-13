"""
This module provides a Visualizer class for visualizing images and point clouds
with annotations from a dataset. The Visualizer supports drawing annotations as
bounding boxes or mask contours on images, and it can display the annotated
images using OpenCV.
"""

# pylint: disable=no-member


from types import SimpleNamespace
from typing import Callable, Tuple

import cv2
import numpy as np
import open3d as o3d
from datumaro.components.annotation import Annotation

from libs.dataset.manager import Frame


def _colors(idx: int) -> Tuple[int, int, int]:
    """Return a list of colors for drawing annotations."""
    colors = {
        "Blue": (255, 0, 0),  # Blue
        "Green": (0, 255, 0),  # Green
        "Red": (0, 0, 255),  # Red
        "Cyan": (255, 255, 0),  # Cyan
        "Magenta": (255, 0, 255),  # Magenta
        "Yellow": (0, 255, 255),  # Yellow
        "Black": (0, 0, 0),  # Black
        "White": (255, 255, 255),  # White
    }
    return list(colors.values())[idx % len(colors)]


def _text_font_args() -> dict:
    """Return the configuration for the visualizer."""
    return {"fontFace": cv2.FONT_HERSHEY_SIMPLEX, "fontScale": 0.5, "thickness": 1}


class Visualizer:  # pylint: disable=too-few-public-methods
    """Visualize images and point clouds from the dataset."""

    def __init__(self, config: SimpleNamespace, label_name_mapper: Callable):
        self._config = config
        self._label_name_mapper = label_name_mapper

    def _write_annotation_label(
        self,
        annotated_image: np.ndarray,
        label_name: str,
        label_coordinates: SimpleNamespace,
        color: Tuple[int, int, int],
    ):
        """Write the label name on the image next to annotation."""
        org = (label_coordinates.x, label_coordinates.y - 10)
        cv2.putText(img=annotated_image, text=label_name, org=org, color=color, **_text_font_args())

    def _draw_annotation_as_bbox(
        self,
        annotated_image: np.ndarray,
        annotation: Annotation,
        label_name: str,
        color: Tuple[int, int, int],
    ):
        """Draws an annotation on the image as a bounding box."""
        x, y, w, h = annotation.get_bbox()
        x, y, w, h = (
            int(x * self._config.image_resize_factor),
            int(y * self._config.image_resize_factor),
            int(w * self._config.image_resize_factor),
            int(h * self._config.image_resize_factor),
        )
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
        label_coordinates = SimpleNamespace(x=int(x), y=int(y))
        self._write_annotation_label(annotated_image, label_name, label_coordinates, color)

    def _visualize_3d(self, frame: Frame):
        """Visualize a point cloud."""
        if self._config.show_output:
            o3d.visualization.draw_geometries([frame.pointcloud], window_name="Point Cloud")

    def _visualize_2d(self, frame: Frame):
        """Visualize a frame with its image and annotations."""
        annotated_image = frame.image.copy()
        if self._config.image_resize_factor != 1.0:
            annotated_image = cv2.resize(
                annotated_image,
                (0, 0),
                fx=self._config.image_resize_factor,
                fy=self._config.image_resize_factor,
            )
        for annotation in frame.annotations:
            self._draw_annotation_as_bbox(
                annotated_image=annotated_image,
                annotation=annotation,
                label_name=self._label_name_mapper(annotation.label),
                color=_colors(annotation.label),
            )

        if self._config.show_output:
            cv2.imshow("Annotated Image", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if self._config.save_output:
            output_path_2d = f"{frame.battery_pack()}_{frame.frame_name()}_2d.png"
            output_path = f"{self._config.output_dir}/{output_path_2d}"
            cv2.imwrite(output_path, annotated_image)

    def visualize_frame(self, frame: Frame):
        """Visualize a frame with its image, point cloud, and annotations."""
        if self._config.visualize_3d:
            self._visualize_3d(frame)
        if self._config.visualize_2d:
            self._visualize_2d(frame)
