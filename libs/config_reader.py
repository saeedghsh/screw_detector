"""Module for reading configuration files in TOML format."""

import os

import toml


def load_config(config_name: str, config_dir: str = "config") -> dict:
    """Load a TOML configuration file from the specified directory.
    Usage example: config = load_config("hough_circle_detector")
    """
    config_path = os.path.join(config_dir, f"{config_name}.toml")
    if not os.path.exists(config_path):  # pragma: no cover
        raise FileNotFoundError(f"Config file '{config_name}.toml' not found in '{config_dir}'")
    return toml.load(config_path)
