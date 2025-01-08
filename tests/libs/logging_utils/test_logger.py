# pylint: disable=missing-module-docstring, missing-function-docstring
import glob
import logging
import os

import pytest
from pytest import FixtureRequest

from libs.logging_utils.logger import setup_logging


@pytest.fixture
def log_directory(tmp_path: str) -> str:
    """
    Return a temporary directory for log files.

    This fixture creates a temporary directory named "logs" within the pytest-provided
    temporary path. The directory is used to store log files during testing.
    """
    log_dir = os.path.join(tmp_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def test_setup_logging_creates_log_directory(request: FixtureRequest):
    log_dir = request.getfixturevalue("log_directory")
    setup_logging(dir_path=log_dir)
    assert os.path.exists(log_dir)


def test_setup_logging_creates_log_file(request: FixtureRequest):
    log_dir = request.getfixturevalue("log_directory")
    _ = setup_logging(dir_path=log_dir)
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    assert len(log_files) == 1
    assert os.path.isfile(log_files[0])


def test_setup_logging_logs_message_correctly(
    request: FixtureRequest, caplog: pytest.LogCaptureFixture
):
    log_dir = request.getfixturevalue("log_directory")  # Explicitly get the fixture value
    logger = setup_logging(level=logging.INFO, dir_path=log_dir)
    test_message = "This is a test log message."
    with caplog.at_level(logging.INFO):
        logger.info(test_message)
    assert test_message in caplog.text


def test_setup_logging_log_file_content(request: FixtureRequest):
    log_dir = request.getfixturevalue("log_directory")
    logger = setup_logging(level=logging.INFO, dir_path=log_dir)
    test_message = "This is a test log message."
    logger.info(test_message)
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    with open(log_files[0], "r", encoding="utf-8") as log_file:
        log_content = log_file.read()
    assert test_message in log_content


def test_setup_logging_name_appendix(request: FixtureRequest):
    log_dir = request.getfixturevalue("log_directory")
    name_appendix = "test_appendix"
    _ = setup_logging(level=logging.INFO, name_appendix=name_appendix, dir_path=log_dir)
    log_files = glob.glob(os.path.join(log_dir, f"*_{name_appendix}.log"))
    assert len(log_files) == 1
    assert os.path.isfile(log_files[0])
