"""Structured logging configuration for the project."""

import logging
import os
import sys
from typing import Optional


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """Configure structured logging for the entire project.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
               Defaults to LOG_LEVEL env var, or INFO.
        log_file: Optional path to write logs to a file in addition to stdout.
    """
    log_level = level or os.getenv("LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
