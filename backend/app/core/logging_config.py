"""Structured logging configuration with rotating file and console handlers."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from app.core.config import settings


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    settings.ensure_dirs()
    log_file = settings.log_dir / "app.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger("missing_person_system")
    if root_logger.handlers:
        return root_logger

    root_logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return root_logger


logger = setup_logging()
