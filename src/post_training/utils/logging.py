"""Structured logging helpers."""

from __future__ import annotations

import logging
import os
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a clean format.

    Call once at the beginning of every entry-point script.
    """
    rank = int(os.environ.get("RANK", 0))
    if level == logging.INFO and rank > 0:
        # Suppress INFO logs from non-main ranks to reduce log spam.
        logging.basicConfig(handlers=[logging.NullHandler()], force=True)
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
