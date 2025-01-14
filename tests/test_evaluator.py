# pylint: disable=missing-module-docstring, missing-function-docstring
from unittest import mock

from libs.detection.detector import Detection
from libs.evaluator import compute_distance, compute_iou, match_detections


def test_compute_iou():
    mock_annotation = mock.MagicMock()
    mock_annotation.get_bbox.return_value = (0, 0, 10, 10)
    detection = Detection(5, 5, 10, 10, label=0)
    iou = compute_iou(mock_annotation, detection)
    assert 0.14 < iou < 0.15


def label_mapper(label: int) -> str:
    return ["screw_head", "screw_hole"][label]


def test_match_detections_all_matched():
    frame = mock.MagicMock()
    frame.annotations_count.return_value = 2
    frame.detections = [
        Detection(x=0, y=0, width=10, height=10, label=0),
        Detection(x=20, y=20, width=10, height=10, label=1),
    ]
    frame.annotations = [
        mock.MagicMock(label=0, get_bbox=mock.Mock(return_value=(0, 0, 10, 10))),
        mock.MagicMock(label=1, get_bbox=mock.Mock(return_value=(20, 20, 10, 10))),
    ]

    matched = match_detections(
        frame,
        iou_threshold=0.5,
        annotation_label_mapper=label_mapper,
        detection_label_mapper=label_mapper,
    )
    assert matched == [0, 1]  # Both annotations matched to respective detections


def test_match_detections_partial_match():
    frame = mock.MagicMock()
    frame.annotations_count.return_value = 2
    frame.detections = [
        Detection(x=0, y=0, width=10, height=10, label=0),
    ]
    frame.annotations = [
        mock.MagicMock(label=0, get_bbox=mock.Mock(return_value=(0, 0, 10, 10))),
        mock.MagicMock(label=1, get_bbox=mock.Mock(return_value=(20, 20, 10, 10))),
    ]

    matched = match_detections(
        frame,
        iou_threshold=0.5,
        annotation_label_mapper=label_mapper,
        detection_label_mapper=label_mapper,
    )

    assert matched == [0, None]  # First annotation matched, second unmatched


def test_match_detections_no_match():
    frame = mock.MagicMock()
    frame.annotations_count.return_value = 2
    frame.detections = [
        Detection(x=50, y=50, width=10, height=10, label=0),
    ]
    frame.annotations = [
        mock.MagicMock(label=0, get_bbox=mock.Mock(return_value=(0, 0, 10, 10))),
        mock.MagicMock(label=1, get_bbox=mock.Mock(return_value=(20, 20, 10, 10))),
    ]

    matched = match_detections(
        frame,
        iou_threshold=0.5,
        annotation_label_mapper=label_mapper,
        detection_label_mapper=label_mapper,
    )
    assert matched == [None, None]  # No annotations matched


def test_match_detections_empty_frame():
    frame = mock.MagicMock()
    frame.annotations_count.return_value = 0
    frame.detections = []
    frame.annotations = []

    matched = match_detections(
        frame,
        iou_threshold=0.5,
        annotation_label_mapper=label_mapper,
        detection_label_mapper=label_mapper,
    )
    assert matched == []  # No annotations or detections


def test_compute_distance():
    mock_annotation = mock.MagicMock()
    mock_annotation.get_bbox.return_value = (0, 0, 10, 10)
    detection = Detection(5, 5, 10, 10, label=0)
    distance = compute_distance(mock_annotation, detection)
    assert isinstance(distance, float)
    assert distance > 0
