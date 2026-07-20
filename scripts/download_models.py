"""Betimsel gözlem için gereken açık model paketlerini indirir.

Şu an yalnızca MediaPipe **FaceLandmarker** paketini (betimsel yüz
gözlemi; bkz. ``modalities/vision/observations.py``) indirir. Model
``models/`` altına yazılır (git'e girmez).

KULLANIM:
    python scripts/download_models.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from psikiyatri_ai.config import get_settings
from psikiyatri_ai.logging_conf import logger, setup_logging

# MediaPipe resmi model deposu (float16 FaceLandmarker, ~3.8 MB).
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def download_face_landmarker(dest_dir: Path | None = None) -> Path:
    """FaceLandmarker paketini indirir (varsa atlar)."""
    dest_dir = dest_dir or get_settings().model_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "face_landmarker.task"
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("FaceLandmarker zaten mevcut: {}", dest)
        return dest
    logger.info("FaceLandmarker indiriliyor → {}", dest)
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, dest)  # noqa: S310
    logger.info("İndirildi: {} ({} bayt)", dest, dest.stat().st_size)
    return dest


def main() -> None:
    setup_logging()
    download_face_landmarker()


if __name__ == "__main__":
    main()
