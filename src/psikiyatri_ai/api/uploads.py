"""Yüklenen medya dosyalarını geçici dosyalara alma yardımcıları.

`/session/analyze` ve `/patients/{id}/interviews` aynı yükleme akışını
paylaşır: gelen video/ses/giyilebilir kayıtları geçici dosyalara yazılır,
analiz sonrası ``cleanup`` ile silinir (ham medya kalıcı saklanmaz).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import UploadFile


async def collect_uploads(
    video: UploadFile | None,
    audio: UploadFile | None,
    wearable: UploadFile | None,
) -> tuple[list[dict], dict[str, Path], list[str]]:
    """Yüklemeleri geçici dosyalara yazar.

    Returns:
        (alınan dosya meta listesi, modalite->geçici yol, geçici dosya adları)
    """
    received: list[dict] = []
    media_paths: dict[str, Path] = {}
    tmp_files: list[str] = []
    uploads = (
        ("vision", video, ".mp4"),
        ("audio", audio, ".wav"),
        ("wearable", wearable, ".csv"),
    )
    for modality, upload, default_suffix in uploads:
        if upload is not None and upload.filename:
            data = await upload.read()
            received.append(
                {"modality": modality, "filename": upload.filename, "size": len(data)}
            )
            suffix = Path(upload.filename).suffix or default_suffix
            fd, tmp_name = tempfile.mkstemp(suffix=suffix)
            # Yazma başarısız olsa bile cleanup silebilsin diye adı
            # hemen (yazmadan önce) listeye ekle.
            tmp_files.append(tmp_name)
            media_paths[modality] = Path(tmp_name)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
    return received, media_paths, tmp_files


def cleanup(tmp_files: list[str]) -> None:
    """Geçici dosyaları en-iyi-çaba siler."""
    for name in tmp_files:
        try:
            os.unlink(name)
        except OSError:  # pragma: no cover
            pass


__all__ = ["cleanup", "collect_uploads"]
