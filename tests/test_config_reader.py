# pylint: disable=missing-module-docstring, missing-function-docstring
import os

import pytest
import toml

from libs.config_reader import load_config


@pytest.fixture
def sample_config(tmp_path: str) -> str:
    config_data = {"dp": 1.2, "min_dist": 20}
    config_path = os.path.join(tmp_path, "hough_circle_detector.toml")
    with open(config_path, "w", encoding="utf-8") as file:
        toml.dump(config_data, file)
    return tmp_path


def test_load_config_valid(request: pytest.FixtureRequest):
    config_dir = request.getfixturevalue("sample_config")
    config = load_config("hough_circle_detector", config_dir=config_dir)
    assert config["dp"] == 1.2
    assert config["min_dist"] == 20


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config")
