# pylint: disable=missing-module-docstring, missing-function-docstring
import subprocess
from unittest.mock import patch

import pytest

from libs.path_utils import repo_root


def test_repo_root_returns_correct_path():
    # Mock `subprocess.check_output` to return a fake repo path with a trailing newline.
    fake_path = "/fake/path/to/repo"
    with patch("subprocess.check_output", return_value=f"{fake_path}\n") as mock_subprocess:
        result = repo_root()
        assert result == fake_path
        mock_subprocess.assert_called_once_with(["git", "rev-parse", "--show-toplevel"], text=True)


def test_repo_root_raises_runtime_error_when_git_fails():
    # Mock `subprocess.check_output` to raise a CalledProcessError.
    with patch("subprocess.check_output") as mock_subprocess:
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="git rev-parse --show-toplevel"
        )
        # Verify that a RuntimeError is raised.
        with pytest.raises(RuntimeError, match="Not in a Git repository"):
            repo_root()
