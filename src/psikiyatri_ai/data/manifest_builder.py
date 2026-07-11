"""Ham seans klasörlerinden anonim manifest üretimi (üretim aracı).

Beklenen ham dizin yapısı::

    <raw_root>/
        <session_id>/
            video.mp4         (opsiyonel)
            audio.wav         (opsiyonel)
            transcript.txt    (opsiyonel)
            meta.json         -> {"patient_id": "...", "label": "depresyon"}

Bu araç, ``meta.json`` içindeki doğrudan tanımlayıcıyı (``patient_id``)
geri döndürülemez takma-kimliğe çevirir ve transkripti PII açısından
maskeler. Çıktı manifest, ham medyayı **kopyalamaz**; yalnızca yollarını
(veri köküne göreli) kaydeder.

> Gizlilik: ``patient_id`` ve ham tanımlayıcılar manifest'e asla yazılmaz.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import get_settings
from ..logging_conf import logger
from . import audit
from .anonymization import mask_pii, pseudonymize
from .schema import DiagnosisLabel, Sample

_MEDIA = {
    "video_path": "video.mp4",
    "audio_path": "audio.wav",
    "wearable_path": "wearable.csv",
}


def _read_meta(session_dir: Path) -> dict:
    meta_file = session_dir / "meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"meta.json bulunamadı: {session_dir}")
    return json.loads(meta_file.read_text(encoding="utf-8"))


def build_manifest_from_raw(
    raw_root: Path,
    *,
    out_path: Path | None = None,
    data_root: Path | None = None,
) -> Path:
    """Ham seans klasörlerini tarayıp anonim bir manifest üretir.

    Returns:
        Yazılan manifest yolu. Etiketsiz/eksik meta olan seanslar atlanır.
    """
    settings = get_settings()
    raw_root = Path(raw_root)
    data_root = Path(data_root) if data_root else settings.data_dir
    out_path = Path(out_path) if out_path else settings.processed_data_dir / "manifest.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    skipped = 0
    for session_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        try:
            meta = _read_meta(session_dir)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning("Seans atlandı ({}): {}", session_dir.name, exc)
            skipped += 1
            continue

        patient_id = meta.get("patient_id")
        label = meta.get("label")
        if not patient_id or not label:
            logger.warning("Seans atlandı ({}): patient_id/label eksik", session_dir.name)
            skipped += 1
            continue

        pseudonym = pseudonymize(str(patient_id))
        fields: dict = {
            "pseudonym": pseudonym,
            "session_id": session_dir.name,
            "label": DiagnosisLabel(label),
            # Açık rıza (KVKK) — meta.json'dan taşınır
            "consent": bool(meta.get("consent", False)),
            "consent_ref": meta.get("consent_ref"),
        }

        # Medya yolları (varsa) — veri köküne göreli kaydedilir
        for attr, filename in _MEDIA.items():
            media = session_dir / filename
            if media.exists():
                try:
                    fields[attr] = media.relative_to(data_root)
                except ValueError:
                    fields[attr] = media

        # Transkript — PII maskelemesinden geçirilir
        transcript_file = session_dir / "transcript.txt"
        if transcript_file.exists():
            fields["transcript"] = mask_pii(transcript_file.read_text(encoding="utf-8"))

        lines.append(Sample(**fields).model_dump_json(exclude_none=True))
        audit.log_event(
            "manifest_ingest",
            subject=pseudonym,
            detail={"session": session_dir.name, "consent": fields["consent"]},
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Manifest yazıldı: {} ({} seans, {} atlandı)", out_path, len(lines), skipped)
    return out_path


__all__ = ["build_manifest_from_raw"]
