"""Araştırma veri seti (NLP korpusu) gözatma uç noktaları.

``data/processed/nlp_manifest.jsonl`` içindeki anonimleştirilmiş
araştırma korpusunu klinik panelde incelemek için salt-okunur API.
Kayıtlar zaten takma-kimlikli ve PII-maskeli olduğundan ek bir
anonimleştirme gerekmez; yine de erişim kimlik doğrulamalıdır ve
denetim günlüğüne işlenir (KVKK).

Uç noktalar:
- ``GET /dataset/summary``            : toplam/etiket dağılımı + kaynak bilgisi
- ``GET /dataset/records``            : sayfalı kayıt listesi (filtre: etiket, arama)
- ``GET /dataset/records/{session_id}``: tek kaydın tam transkripti
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..config import get_settings
from ..data import audit
from ..db.models import Role, User
from ..logging_conf import logger
from .security import require_roles

router = APIRouter(prefix="/dataset", tags=["veri seti"])

_PREVIEW_CHARS = 220

# Manifest bellekte bir kez yüklenir; dosya değişirse (mtime) tazelenir.
_cache_lock = threading.Lock()
_cache: dict = {"mtime": None, "records": [], "by_id": {}}


def _manifest_path() -> Path:
    return get_settings().processed_data_dir / "nlp_manifest.jsonl"


def _load_records() -> list[dict]:
    """Manifest kayıtlarını (mtime ile) önbellekten döndürür."""
    path = _manifest_path()
    if not path.exists():
        return []
    mtime = path.stat().st_mtime
    with _cache_lock:
        if _cache["mtime"] == mtime:
            return _cache["records"]
        records: list[dict] = []
        by_id: dict[str, dict] = {}
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records.append(rec)
                by_id[rec.get("session_id", "")] = rec
        _cache.update(mtime=mtime, records=records, by_id=by_id)
        logger.info("Veri seti önbelleğe alındı: {} kayıt", len(records))
        return records


class DatasetSummary(BaseModel):
    available: bool
    total: int
    label_counts: dict[str, int]
    source: str
    consent_ref: str | None = None
    note: str


class DatasetRecordPreview(BaseModel):
    session_id: str
    pseudonym: str
    label: str
    transcript_preview: str
    transcript_chars: int


class DatasetRecordDetail(DatasetRecordPreview):
    transcript: str
    consent_ref: str | None = None


class DatasetPage(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[DatasetRecordPreview]


def _preview(rec: dict) -> DatasetRecordPreview:
    text = rec.get("transcript", "")
    return DatasetRecordPreview(
        session_id=rec.get("session_id", ""),
        pseudonym=rec.get("pseudonym", ""),
        label=rec.get("label", ""),
        transcript_preview=text[:_PREVIEW_CHARS] + ("…" if len(text) > _PREVIEW_CHARS else ""),
        transcript_chars=len(text),
    )


@router.get("/summary", response_model=DatasetSummary)
def dataset_summary(user: User = Depends(require_roles(Role.YONETICI))) -> DatasetSummary:
    """Korpus özeti: toplam kayıt ve etiket dağılımı."""
    records = _load_records()
    counts: dict[str, int] = {}
    for rec in records:
        lbl = rec.get("label", "belirsiz")
        counts[lbl] = counts.get(lbl, 0) + 1
    return DatasetSummary(
        available=bool(records),
        total=len(records),
        label_counts=dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        source=_manifest_path().name,
        consent_ref=records[0].get("consent_ref") if records else None,
        note=(
            "Kayıtlar takma-kimlikli ve PII-maskelidir; transkriptler yalnızca "
            "hasta-sesi anket yanıtlarından kurulur (tanı adı sızıntısı temizlenmiştir)."
        ),
    )


@router.get("/records", response_model=DatasetPage)
def dataset_records(
    label: str | None = Query(None, description="Etikete göre filtrele"),
    q: str | None = Query(None, min_length=2, description="Transkript içinde ara"),
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=100),
    user: User = Depends(require_roles(Role.YONETICI)),
) -> DatasetPage:
    """Sayfalı, filtrelenebilir kayıt listesi (önizleme metniyle)."""
    records = _load_records()
    if label:
        records = [r for r in records if r.get("label") == label]
    if q:
        needle = q.casefold()
        records = [r for r in records if needle in r.get("transcript", "").casefold()]
    page = records[offset : offset + limit]
    return DatasetPage(
        total=len(records),
        offset=offset,
        limit=limit,
        items=[_preview(r) for r in page],
    )


@router.get("/records/{session_id}", response_model=DatasetRecordDetail)
def dataset_record_detail(
    session_id: str, user: User = Depends(require_roles(Role.YONETICI))
) -> DatasetRecordDetail:
    """Tek kaydın tam (maskeli) transkripti; erişim denetime işlenir."""
    _load_records()
    rec = _cache["by_id"].get(session_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Kayıt bulunamadı: {session_id}")
    audit.log_event(
        "dataset_record_view",
        subject=rec.get("pseudonym", "-"),
        detail={"session": session_id, "user": user.username},
    )
    return DatasetRecordDetail(
        **_preview(rec).model_dump(),
        transcript=rec.get("transcript", ""),
        consent_ref=rec.get("consent_ref"),
    )


__all__ = ["router"]
