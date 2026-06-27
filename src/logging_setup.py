"""Loguru logging configuration with rich formatting."""

import sys

from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    """Configure structured application logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=level.upper(),
        colorize=True,
    )
