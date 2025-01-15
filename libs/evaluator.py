"""Module for evaluating object detection models."""

import json
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from datumaro.components.annotation import Annotation
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from libs.dataset.data_structure import Detection2D, Frame
from libs.dataset.manager import DatasetManager
from libs.detection.detector_2d import Detector2D


def compute_iou(ann: Annotation, det: Detection2D) -> float:  # pylint: disable=too-many-locals
    """Compute the Intersection over Union (IoU) of two bounding boxes."""
    ann_x, ann_y, ann_width, ann_height = ann.get_bbox()
    x1_min, y1_min, x1_max, y1_max = ann_x, ann_y, ann_x + ann_width, ann_y + ann_height
    x2_min, y2_min, x2_max, y2_max = det.x, det.y, det.x + det.width, det.y + det.height
    inter_x_min, inter_y_min = max(x1_min, x2_min), max(y1_min, y2_min)
    inter_x_max, inter_y_max = min(x1_max, x2_max), min(y1_max, y2_max)
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    union_area = ann_width * ann_height + det.width * det.height - inter_area
    return inter_area / union_area if union_area > 0 else 0


def match_detections(
    frame: Frame,
    iou_threshold: float,
    annotation_label_mapper: Callable,
    detection_label_mapper: Callable,
) -> List[Optional[int]]:
    """Match detections to ground truths based on IoU and labels."""
    matched: List[Optional[int]] = [None] * frame.annotations_count()
    if not frame.detections or not frame.annotations:
        return matched
    for det_idx, det in enumerate(frame.detections):
        for ann_idx, ann in enumerate(frame.annotations):
            ann_label = annotation_label_mapper(ann.label)
            det_label = detection_label_mapper(det.label)
            if compute_iou(ann, det) >= iou_threshold and ann_label == det_label:
                matched[ann_idx] = det_idx
                break
    return matched


def compute_distance(ann: Annotation, det: Detection2D) -> float:
    """Compute the Euclidean distance between the centers of two bounding boxes."""
    ann_x, ann_y, ann_width, ann_height = ann.get_bbox()
    return np.linalg.norm(
        [
            (ann_x + ann_width / 2) - (det.x + det.width / 2),
            (ann_y + ann_height / 2) - (det.y + det.height / 2),
        ]
    ).astype(float)


class Evaluator:  # pylint: disable=too-few-public-methods
    """Evaluate object detection models."""

    def __init__(self, config: dict):
        self._config = config

    def evaluate(
        self, detector: Detector2D, dataset_manager: DatasetManager, test_frames: List[str]
    ) -> Dict[str, Any]:
        # pylint: disable=too-many-locals
        """Evaluate a detector on a dataset."""
        y_true = []
        y_pred = []
        total_localization_error = 0.0
        true_positive_count = 0
        total_false_positives = 0

        dataset_labels = [
            dataset_manager.label_name_mapper(idx) for idx in range(dataset_manager.label_count())
        ]

        for frame_id in test_frames:
            frame = dataset_manager.frame(frame_id)
            frame.detections = detector.detect(frame.image)

            matched = match_detections(
                frame,
                self._config["iou_threshold"],
                dataset_manager.label_name_mapper,
                Detection2D.label_name_mapper,
            )

            # Collect true positives, false positives, and false negatives
            for ann_idx, det_idx in enumerate(matched):
                if det_idx is not None:  # True Positive: matched annotation-detection pair
                    y_true.append(frame.annotations[ann_idx].label)  # type: ignore
                    y_pred.append(frame.detections[det_idx].label)
                else:  # False negative: unmatched annotation
                    y_true.append(frame.annotations[ann_idx].label)  # type: ignore
                    y_pred.append(-1)

            # indices to detections that are false positive
            false_positives = [i for i in range(len(frame.detections)) if i not in matched]
            for det_idx in false_positives:
                y_true.append(-1)  # No matching annotation
                y_pred.append(frame.detections[det_idx].label)

            total_false_positives += len(false_positives)

            for ann_idx, det_idx in enumerate(matched):
                if det_idx is not None:
                    det = frame.detections[det_idx]
                    closest_gt = frame.annotations[ann_idx]  # type: ignore
                    total_localization_error += compute_distance(closest_gt, det)
                    true_positive_count += 1

        labels = list(range(-1, len(dataset_labels)))
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
        mean_localization_error = (
            total_localization_error / true_positive_count if true_positive_count > 0 else 0
        )

        results = {
            "precision": dict(zip(dataset_labels, precision)),
            "recall": dict(zip(dataset_labels, recall)),
            "f1_score": dict(zip(dataset_labels, f1_score)),
            "mean_localization_error": mean_localization_error,
            "confusion_matrix": conf_matrix.tolist(),
            "total_false_positives": total_false_positives,
            "detector_config": detector.__dict__,
            "evaluator_config": self.__dict__,
        }
        self._store_results(results)
        return results

    def _store_results(self, results: Dict[str, Any]):
        """Store evaluation results with configuration in a timestamped JSON file."""
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        dir_path = self._config["results_dir_path"]
        os.makedirs(dir_path, exist_ok=True)
        results_file = os.path.join(dir_path, f"{timestamp}.json")
        try:
            with open(results_file, "w", encoding="utf-8") as file:
                json.dump(results, file, indent=4)
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error writing results to file: {e}")
