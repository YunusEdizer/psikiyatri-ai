"""Erişim ve işleme denetim günlüğü (audit log).

KVKK/GDPR *hesap verebilirlik* ilkesi gereği, kişisel veriye erişim ve
işleme olayları geriye dönük izlenebilir olmalıdır. Bu modül olayları
yapılandırılabilir bir dosyaya (varsayılan: ``<data>/audit/audit.jsonl``)
satır-JSON (JSONL) olarak yazar.

Gizlilik: Günlüğe yalnızca takma-kimlik (pseudonym) ve olay meta verisi
yazılır; doğrudan tanımlayıcı (ad, TC, dosya no) **asla** yazılmaz.
Yazma en iyi çaba (best-effort) yapılır: bir hata isteği/işlemi kesmez.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ..config import get_settings
from ..logging_conf import logger


def _audit_path() -> Path:
    settings = get_settings()
    if settings.audit_log_path is not None:
        return Path(settings.audit_log_path)
    return settings.data_dir / "audit" / "audit.jsonl"


def log_event(
    action: str,
    *,
    subject: str = "-",
    detail: dict | None = None,
    path: Path | None = None,
) -> None:
    """Bir denetim olayını JSONL dosyasına ekler (best-effort).

    Args:
        action: olay türü (ör. ``"session_analyze"``, ``"manifest_build"``).
        subject: ilgili takma-kimlik veya ``"-"`` (doğrudan tanımlayıcı değil).
        detail: ek meta veri (kişisel veri içermemelidir).
        path: hedef dosya (varsayılan: yapılandırmadaki denetim yolu).
    """
    entry = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "action": action,
        "subject": subject,
        "detail": detail or {},
    }
    fp = Path(path) if path is not None else _audit_path()
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        with fp.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:  # pragma: no cover - günlükleme isteği kesmemeli
        logger.warning("Denetim günlüğü yazılamadı: {}", exc)


def read_events(path: Path | None = None) -> list[dict]:
    """Denetim günlüğündeki olayları liste olarak okur (yoksa boş liste)."""
    fp = Path(path) if path is not None else _audit_path()
    if not fp.exists():
        return []
    lines = fp.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


__all__ = ["log_event", "read_events"]
