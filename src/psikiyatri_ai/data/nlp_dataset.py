"""Psikiyatrik NLP Excel veri setini anonim manifest'e çevirir.

Bu modül, dış kaynaklı Türkçe psikiyatrik NLP veri setini
(``psychiatric_nlp_dataset_*.xlsx``) projenin eğitim hattının beklediği
anonimleştirilmiş **JSONL manifest**'ine (bkz. ``data/schema.py`` ve
``docs/veri-formati.md``) dönüştürür.

Tasarım ilkeleri
----------------
1. **KVKK / anonimlik.** ``Hasta_ID`` geri döndürülemez takma-kimliğe
   (``pseudonymize``) çevrilir; oluşturulan transkript PII maskelemesinden
   (``mask_pii``) geçer. Ham Excel git'e girmez (``data/external/`` yok
   sayılır).
2. **Etiket sızıntısını (label leakage) önleme.** Veri setinin doktor
   notu / uzun anlatı / kamera-ses analizi sütunları tanının adını
   *açıkça* içerir. Metin modeli hastalık adını ezberlemesin diye
   transkript yalnızca **hasta-sesi anket cevaplarından (Q1–Q24)** ve
   sızıntısı temizlenmiş uzun anlatıdan kurulur; tanı adları ``[TANI]``
   ile maskelenir.
3. **Sadık taksonomi.** Veri setindeki sekiz tanı, projenin genişletilmiş
   ``DiagnosisLabel`` taksonomisine birebir eşlenir (granülerlik korunur).

Kullanım::

    from psikiyatri_ai.data.nlp_dataset import build_manifest_from_nlp_dataset
    build_manifest_from_nlp_dataset("data/external/psychiatric_nlp_dataset_20260627.xlsx")
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..logging_conf import logger
from . import audit
from .anonymization import mask_pii, pseudonymize
from .schema import DiagnosisLabel, Sample

#: Excel'de hasta yanıtlarını taşıyan çalışma sayfası.
DATA_SHEET = "Psikiyatrik NLP Verisi"

#: Bu veri seti tümüyle metin (NLP) modalitesidir; başka modalite yoktur.
#: De-identifiye araştırma korpusu olarak işlenir (bireysel rıza değil,
#: veri-seti düzeyinde etik/kaynak referansı).
CONSENT_REF = "ARASTIRMA-VERI-SETI/psychiatric_nlp_dataset (de-identifiye)"

# --------------------------------------------------------------------------- #
# Sütun adları — veri seti sürümleri arasında değiştiğinden aday listeleriyle
# çözümlenir (ilk eşleşen başlık kullanılır). 20260627 (ilk) ve 20260719
# (klinik) sürümleri desteklenir.
# --------------------------------------------------------------------------- #
_ID_COL = "Hasta_ID"
_DIAGNOSIS_COLS = ("Hastalık tanısı", "Yenilenmiş_Tanı")
_SEVERITY_COLS = ("Şiddet_Düzeyi", "Yenilenmiş_Şiddet_Düzeyi")
_COUNTRY_COL = "Ülke"
_LONG_TEXT_COLS = ("Hastanın_Uzun_Anlatımı", "Uzun_Metinli_Cevaplar")


def _resolve(header: list[str], candidates: tuple[str, ...]) -> str | None:
    """Başlıkta aday sütun adlarından ilk eşleşeni döndürür."""
    hset = set(header)
    return next((c for c in candidates if c in hset), None)

# --------------------------------------------------------------------------- #
# Etiket eşlemesi (veri seti tanısı -> kanonik taksonomi)
# --------------------------------------------------------------------------- #
#: Sadık eşleme: her klinik tanı kendi sınıfına gider (granülerlik korunur).
#: ``anksiyete`` etiketi yaygın anksiyete (YAB) çekirdeğini temsil eder.
DIAGNOSIS_MAP: dict[str, DiagnosisLabel] = {
    "Majör Depresif Bozukluk": DiagnosisLabel.DEPRESYON,
    "Yaygın Anksiyete Bozukluğu": DiagnosisLabel.ANKSIYETE,
    "Sosyal Anksiyete Bozukluğu": DiagnosisLabel.SOSYAL_ANKSIYETE,
    "Panik Bozukluk": DiagnosisLabel.PANIK,
    "Obsesif Kompulsif Bozukluk": DiagnosisLabel.OKB,
    "Travma Sonrası Stres Bozukluğu": DiagnosisLabel.TSSB,
    "Bipolar Bozukluk": DiagnosisLabel.BIPOLAR,
    "Şizofreni": DiagnosisLabel.SIZOFRENI,
}

# --------------------------------------------------------------------------- #
# Etiket sızıntısı temizliği
# --------------------------------------------------------------------------- #
#: Transkriptten temizlenecek tanı adları / kısaltmaları. Metin modelinin
#: hastalık adını doğrudan okuyup ezberlemesini engeller.
_LEAK_TERMS = [
    "Yaygın Anksiyete Bozukluğu",
    "Majör Depresif Bozukluk",
    "Obsesif Kompulsif Bozukluk",
    "Sosyal Anksiyete Bozukluğu",
    "Travma Sonrası Stres Bozukluğu",
    "Bipolar Bozukluk",
    "Panik Bozukluk",
    "Depresyon",
    "Şizofreni",
    "OKB",
    "TSSB",
    "YAB",
]
_LEAK_RE = re.compile(
    "|".join(re.escape(t) for t in sorted(_LEAK_TERMS, key=len, reverse=True)),
    re.IGNORECASE,
)


def scrub_diagnosis_leak(text: str) -> str:
    """Serbest metinden tanı adı/kısaltması sızıntısını maskeler."""
    return _LEAK_RE.sub("[TANI]", text)


def _q_columns(header: list[str]) -> list[str]:
    """Anket (Q1–Q24) sütunlarını başlık sırasında döndürür."""

    def _qnum(name: str) -> int:
        m = re.match(r"Q(\d+)", name)
        return int(m.group(1)) if m else 0

    qs = [h for h in header if isinstance(h, str) and re.match(r"Q\d+", h)]
    return sorted(qs, key=_qnum)


def build_transcript(
    row: dict[str, Any], q_cols: list[str], long_text_col: str | None = None
) -> str:
    """Bir satırdan hasta-sesi transkripti kurar (sızıntısız, maskeli).

    Yalnızca anket cevapları (Q1–Q24) ve tanı adı temizlenmiş uzun anlatı
    kullanılır; klinisyen notu / kamera / ses analiz sütunları tanıyı
    açıkça içerdiğinden **dahil edilmez**.
    """
    parts: list[str] = []
    for col in q_cols:
        val = row.get(col)
        if val is not None and str(val).strip():
            parts.append(str(val).strip())

    long_text = row.get(long_text_col) if long_text_col else None
    if long_text is not None and str(long_text).strip():
        parts.append(scrub_diagnosis_leak(str(long_text).strip()))

    transcript = "\n".join(parts)
    # Kalan tanı sızıntısını (Q'larda beklenmez ama güvence) temizle, sonra
    # PII maskele.
    transcript = scrub_diagnosis_leak(transcript)
    return mask_pii(transcript)


@dataclass
class IngestStats:
    """Aktarım özeti."""

    total: int = 0
    written: int = 0
    skipped_no_label: int = 0
    skipped_unknown_label: int = 0
    skipped_empty_text: int = 0
    label_counts: Counter[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.label_counts is None:
            self.label_counts = Counter()


def _load_rows(xlsx_path: Path) -> tuple[list[str], Iterator[tuple]]:
    """Excel veri sayfasını (başlık, satır üreteci) olarak açar."""
    try:
        import openpyxl  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "NLP veri setini okumak için 'openpyxl' gerekli. "
            "Kurulum: pip install openpyxl"
        ) from exc

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    if DATA_SHEET not in wb.sheetnames:
        raise ValueError(
            f"'{DATA_SHEET}' sayfası bulunamadı. Mevcut sayfalar: {wb.sheetnames}"
        )
    ws = wb[DATA_SHEET]
    rows = ws.iter_rows(values_only=True)
    header = [str(c) if c is not None else "" for c in next(rows)]
    return header, rows


def iter_samples(
    xlsx_path: Path,
    *,
    salt: str | None = None,
    limit: int | None = None,
    stats: IngestStats | None = None,
) -> Iterator[Sample]:
    """Excel satırlarını anonimleştirilmiş ``Sample`` örneklerine çevirir.

    Etiketsiz, bilinmeyen etiketli ya da boş metinli satırlar atlanır
    (``stats`` verilirse sayaçlara işlenir).
    """
    xlsx_path = Path(xlsx_path)
    header, rows = _load_rows(xlsx_path)
    q_cols = _q_columns(header)
    if not q_cols:
        raise ValueError("Anket (Q1–Q24) sütunları bulunamadı — beklenen şema uyuşmuyor.")

    diag_col = _resolve(header, _DIAGNOSIS_COLS)
    if diag_col is None:
        raise ValueError(
            f"Tanı sütunu bulunamadı. Aranan: {_DIAGNOSIS_COLS}; başlık: {header[:5]}…"
        )
    long_text_col = _resolve(header, _LONG_TEXT_COLS)

    st = stats if stats is not None else IngestStats()
    for i, values in enumerate(rows):
        if limit is not None and st.written >= limit:
            break
        st.total += 1
        row = dict(zip(header, values, strict=False))

        raw_diag = row.get(diag_col)
        if not raw_diag or not str(raw_diag).strip():
            st.skipped_no_label += 1
            continue
        label = DIAGNOSIS_MAP.get(str(raw_diag).strip())
        if label is None:
            st.skipped_unknown_label += 1
            logger.warning("Bilinmeyen tanı atlandı: {!r}", raw_diag)
            continue

        transcript = build_transcript(row, q_cols, long_text_col)
        if not transcript.strip():
            st.skipped_empty_text += 1
            continue

        raw_id = row.get(_ID_COL) or f"NLP_{i + 1:06d}"
        session_id = str(raw_id).strip()
        sample = Sample(
            pseudonym=pseudonymize(session_id, salt=salt),
            session_id=session_id,
            transcript=transcript,
            label=label,
            consent=True,
            consent_ref=CONSENT_REF,
        )
        st.written += 1
        st.label_counts[label.value] += 1
        yield sample


def build_manifest_from_nlp_dataset(
    xlsx_path: str | Path,
    *,
    out_path: Path | None = None,
    limit: int | None = None,
    salt: str | None = None,
) -> tuple[Path, IngestStats]:
    """Excel veri setinden anonim JSONL manifest üretir.

    Returns:
        (yazılan manifest yolu, aktarım istatistikleri).
    """
    settings = get_settings()
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Veri seti bulunamadı: {xlsx_path}")

    out_path = (
        Path(out_path)
        if out_path
        else settings.processed_data_dir / "nlp_manifest.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = IngestStats()
    with out_path.open("w", encoding="utf-8") as fh:
        for sample in iter_samples(xlsx_path, salt=salt, limit=limit, stats=stats):
            fh.write(sample.model_dump_json(exclude_none=True))
            fh.write("\n")

    audit.log_event(
        "nlp_dataset_ingest",
        subject=xlsx_path.name,
        detail={
            "written": stats.written,
            "skipped": (
                stats.skipped_no_label
                + stats.skipped_unknown_label
                + stats.skipped_empty_text
            ),
            "labels": dict(stats.label_counts),
        },
    )
    logger.info(
        "NLP manifest yazıldı: {} ({} örnek, {} atlandı)",
        out_path,
        stats.written,
        stats.total - stats.written,
    )
    return out_path, stats


__all__ = [
    "CONSENT_REF",
    "DIAGNOSIS_MAP",
    "IngestStats",
    "build_manifest_from_nlp_dataset",
    "build_transcript",
    "iter_samples",
    "scrub_diagnosis_leak",
]
