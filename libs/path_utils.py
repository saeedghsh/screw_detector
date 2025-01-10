"""Utility functions for working with file paths."""

import subprocess


def repo_root() -> str:
    """Return the full path to the root of the repository."""
    try:
        repo_root_path = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Error: Not in a Git repository or unable to determine the repo root."
        ) from exc
    return repo_root_path
