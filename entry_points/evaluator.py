"""Entry point for the performance evaluator module."""

# pylint: disable=missing-function-docstring

import os
import sys
from pathlib import Path
from typing import Sequence

from libs.config_reader import load_config
from libs.dataset.manager import DatasetManager
from libs.dataset.utils import CACHE_DIR, load_cached_split
from libs.detection.hough_circle_detector import HoughCircleDetector
from libs.evaluator import Evaluator
from libs.logger import setup_logging

logger = setup_logging(name_appendix="Detector2D Evaluation")


def main(_: Sequence[str]) -> int:
    dataset_manger = DatasetManager()
    circle_detector = HoughCircleDetector(load_config("hough_circle_detector"))
    evaluator = Evaluator(load_config("evaluator"))

    cached_data_split_files = list(Path(CACHE_DIR).rglob("*.json"))
    for f in cached_data_split_files:
        data_split_cache = load_cached_split(str(f))
        logger.info("Cached data split file: %s", f)
        evaluation_results = evaluator.evaluate(
            detector=circle_detector,
            dataset_manager=dataset_manger,
            test_frames=data_split_cache.test_frame_ids,
        )
        logger.info(
            "Precision: %.4f, Recall: %.4f",
            evaluation_results["precision"],
            evaluation_results["recall"],
        )

    return os.EX_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
