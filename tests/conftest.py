"""Automatically found by pytest and adds the path to root.

This module is automatically discovered by pytest and ensures that the project
root directory is added to the Python path. This is necessary because pytest
does not automatically add the root directory to the Python path when executing
tests with the `-m` option.

The script determines the project root path by navigating two levels up from the
current file's directory and inserts this path at the beginning of the system
path list (`sys.path`). This allows for the proper import of project modules
during testing.

With the entry points we execute with -m and things are OK. But pytest fails at
adding the root to python path, so we enforce it via this file.
"""

import os
import sys

project_root_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root_path)
