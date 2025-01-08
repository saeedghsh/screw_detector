"""This module is the entry point for the data inspector application."""

import argparse
import logging
import os
import sys
from typing import Sequence

from libs.logging_utils.logger import setup_logging

logger = setup_logging(name_appendix="data-inspector", level=logging.DEBUG)


def parse_args(args: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Data Inspector Application")
    return parser.parse_args(args)


def main(args: Sequence[str]) -> int:
    """Main entry point for the data inspector application."""
    parsed_args = parse_args(args)
    logger.info("Data inspector started with arguments: %s", parsed_args)
    return os.EX_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
