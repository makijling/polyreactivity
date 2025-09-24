"""Logging utilities."""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER_NAME = "polyreact"


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project logger."""

    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        logger.setLevel(level)
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def write_logfile(path: str | Path, *, level: int = logging.INFO) -> logging.Logger:
    """Add a file handler to the project logger."""

    logger = configure_logging(level)
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
