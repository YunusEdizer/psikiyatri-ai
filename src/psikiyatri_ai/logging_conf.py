"""Proje geneli loglama yapılandırması (loguru)."""

from __future__ import annotations

import sys

from loguru import logger

from .config import get_settings

_configured = False


def setup_logging() -> None:
    """Loglayıcıyı ayarlardaki seviyeye göre yapılandırır (idempotent)."""
    global _configured
    if _configured:
        return

    settings = get_settings()
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
        ),
        backtrace=True,
        diagnose=settings.env != "production",
    )
    _configured = True


__all__ = ["logger", "setup_logging"]
