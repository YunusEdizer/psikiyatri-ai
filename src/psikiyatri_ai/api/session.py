"""Seans değerlendirme mantığı (klinik panel için).

Yüklenen kayıtları (video/ses/transkript) alır, transkripti KVKK gereği
maskeler ve sözlük-tabanlı metin ön analizini çalıştırır. Ses/görüntü için
otomatik skorlama tam derin öğrenme hattını gerektirdiğinden, bu prototipte
bu modaliteler "alındı" olarak raporlanır.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ..data.anonymization import mask_pii
from ..modalities.text.heuristics import analyze_text_risk
from ..modalities.text.preprocess import normalize_text

DISCLAIMER = (
    "Bu çıktı klinik karar desteği amaçlıdır ve kesin teşhis yerine geçmez. "
    "Nihai karar yetkili sağlık uzmanına aittir."
)
PROTOTYPE_NOTE = (
    "Prototip (TRL6): metin ön analizi sözlük tabanlıdır; ses ve görüntü "
    "modalitelerinin otomatik skorlanması tam derin öğrenme hattı ile "
    "gerçekleştirilecektir."
)


def _risk_level(prob: float, high_risk: bool) -> str:
    if high_risk or prob >= 0.66:
        return "yüksek"
    if prob >= 0.40:
        return "orta"
    return "düşük"


def analyze_session(
    pseudonym: str,
    evaluator: str,
    transcript_text: str,
    received: list[dict],
) -> dict:
    """Bir seansı değerlendirir ve klinik rapor sözlüğü döndürür."""
    now = datetime.now(UTC)
    session_id = "SES-" + now.strftime("%Y%m%d-%H%M%S")

    modalities = [r["modality"] for r in received]
    has_transcript = bool(transcript_text and transcript_text.strip())
    if has_transcript:
        modalities.append("text")

    files = [
        {
            "modality": r["modality"],
            "filename": r.get("filename") or "-",
            "size_kb": round(r.get("size", 0) / 1024, 1),
        }
        for r in received
    ]

    report: dict = {
        "session_id": session_id,
        "created_at": now.isoformat(timespec="seconds"),
        "pseudonym": pseudonym.strip() or "anonim",
        "evaluator": evaluator.strip() or "-",
        "received": files,
        "modalities": modalities,
        "matched_indicators": {},
        "transcript_preview": None,
        "probabilities": None,
        "predicted_label": None,
        "risk_level": None,
        "disclaimer": DISCLAIMER,
        "prototype_note": PROTOTYPE_NOTE,
    }

    if has_transcript:
        masked = mask_pii(normalize_text(transcript_text))
        risk = analyze_text_risk(masked)
        pd = risk.depression_probability
        report.update(
            analysis_available=True,
            method="Sözlük tabanlı metin ön analizi (prototip)",
            probabilities={"kontrol": round(1 - pd, 4), "depresyon": pd},
            predicted_label="depresyon" if pd >= 0.5 else "kontrol",
            risk_level=_risk_level(pd, risk.high_risk),
            high_risk=risk.high_risk,
            matched_indicators=risk.matched,
            transcript_preview=masked[:500],
        )
    else:
        report.update(
            analysis_available=False,
            method="-",
            high_risk=False,
            note="Otomatik skorlama için transkript gerekli; ses/görüntü tam "
            "hatta işlenecektir.",
        )

    return report


__all__ = ["analyze_session"]
