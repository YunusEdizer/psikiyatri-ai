"""Seans değerlendirme mantığı (klinik panel için).

Yüklenen kayıtları (video/ses/giyilebilir/transkript) alır, transkripti
KVKK gereği maskeler, sözlük-tabanlı metin ön analizini çalıştırır ve
**ilgili modalite bağımlılıkları kuruluysa** ses/görüntü/giyilebilir
kayıtları gerçek derin öğrenme füzyon modelinden geçirir. Ağır
bağımlılıklar (mediapipe/librosa) yoksa skorlama zarifçe metin ön
analizine geri düşer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..data import audit
from ..data.anonymization import mask_pii
from ..modalities.text.heuristics import analyze_text_risk
from ..modalities.text.preprocess import normalize_text

DISCLAIMER = (
    "Bu çıktı klinik karar desteği amaçlıdır ve kesin teşhis yerine geçmez. "
    "Nihai karar yetkili sağlık uzmanına aittir."
)
PROTOTYPE_NOTE = (
    "Prototip (TRL6): eğitilmiş model yoksa çıktı rastgele-başlatılmış ağla "
    "üretilir. Ses/görüntü/giyilebilir skorlaması ilgili modalite ekstraları "
    "(vision/audio) kurulu olduğunda etkinleşir; aksi hâlde metin ön analizi "
    "kullanılır."
)


def _risk_level(prob: float, high_risk: bool) -> str:
    if high_risk or prob >= 0.66:
        return "yüksek"
    if prob >= 0.40:
        return "orta"
    return "düşük"


def score_media(media_paths: dict[str, Path]) -> dict | None:
    """Ses/görüntü/giyilebilir kayıtları gerçek füzyon modelinden geçirir.

    İlgili modalite bağımlılığı yoksa (ör. mediapipe/librosa kurulu değil)
    ya da çıkarım başarısızsa ``None`` döner (çağıran metin analizine düşer).
    """
    import torch

    from ..config import get_settings
    from ..data.batch import SampleFeatures
    from ..data.datamodule import FeatureExtractor, collate_features
    from ..data.schema import Sample
    from ..logging_conf import logger
    from ..training.checkpoint import labels_for_state, load_metadata
    from ..training.model import PsikiyatriModel

    # Sadece derin öğrenme kodlayıcısı olan modaliteler (metin ayrı işlenir).
    candidates = [m for m in ("vision", "audio", "wearable") if media_paths.get(m)]
    if not candidates:
        return None

    sample = Sample(
        pseudonym="anonim",
        session_id="canli",
        video_path=media_paths.get("vision"),
        audio_path=media_paths.get("audio"),
        wearable_path=media_paths.get("wearable"),
    )

    feats: dict = {}
    scored: list[str] = []
    for modality in candidates:
        try:
            part = FeatureExtractor(modalities=(modality,)).extract(sample)
        except Exception as exc:  # noqa: BLE001 - eksik bağımlılık/bozuk medya
            logger.warning("'{}' modalitesi skorlanamadı: {}", modality, exc)
            continue
        if part:
            feats.update(part)
            scored.append(modality)

    if not scored:
        return None

    settings = get_settings()
    ckpt = settings.model_dir / "psikiyatri_model.pt"
    state = torch.load(ckpt, map_location="cpu") if ckpt.exists() else None
    meta = load_metadata(settings.model_dir)
    if meta:
        classes = list(meta["classes"])
    elif state is not None:
        classes = labels_for_state(state, ["kontrol", "depresyon"])
    else:
        classes = ["kontrol", "depresyon"]
    model_cfg = settings.model.model_copy(update={"num_classes": len(classes)})
    model = PsikiyatriModel(
        model_cfg,
        enabled_modalities=tuple(scored),
        use_egemaps="audio_egemaps" in feats,
    )
    trained = state is not None
    if state is not None:
        model.load_state_dict(state, strict=False)
    model.eval()

    batch, _ = collate_features([SampleFeatures(label=0, **feats)])
    with torch.no_grad():
        out = model(batch)
    probs = torch.softmax(out.logits, dim=-1).squeeze(0)
    present = batch.present_modalities
    return {
        "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))},
        "predicted_label": classes[int(probs.argmax())],
        "incongruence": float(out.incongruence[0]),
        "modality_contributions": [
            {"modality": m, "weight": float(out.attention[0, i])}
            for i, m in enumerate(present)
        ],
        "scored_modalities": scored,
        "trained": trained,
    }


def analyze_session(
    pseudonym: str,
    evaluator: str,
    transcript_text: str,
    received: list[dict],
    media_paths: dict[str, Path] | None = None,
) -> dict:
    """Bir seansı değerlendirir ve klinik rapor sözlüğü döndürür."""
    now = datetime.now(UTC)
    session_id = "SES-" + now.strftime("%Y%m%d-%H%M%S")
    media_paths = media_paths or {}

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
        "incongruence": None,
        "modality_contributions": [],
        "scored_modalities": [],
        "disclaimer": DISCLAIMER,
        "prototype_note": PROTOTYPE_NOTE,
    }

    # 1) Metin ön analizi (sözlük tabanlı) — her zaman denenir.
    text_risk = None
    if has_transcript:
        masked = mask_pii(normalize_text(transcript_text))
        text_risk = analyze_text_risk(masked)
        report["transcript_preview"] = masked[:500]
        report["matched_indicators"] = text_risk.matched

    # 2) Ses/görüntü/giyilebilir derin öğrenme skorlaması (bağımlılık varsa).
    media_result = None
    if media_paths:
        try:
            media_result = score_media(media_paths)
        except Exception:  # noqa: BLE001 - skorlama isteği asla kesmemeli
            media_result = None

    high_risk = bool(text_risk.high_risk) if text_risk else False

    if media_result is not None:
        probs = media_result["probabilities"]
        finding_prob = 1.0 - probs.get("kontrol", 0.0)
        method = "Derin öğrenme füzyonu (" + " + ".join(media_result["scored_modalities"]) + ")"
        if not media_result["trained"]:
            method += " — eğitilmemiş model (demo)"
        report.update(
            analysis_available=True,
            method=method,
            probabilities={k: round(v, 4) for k, v in probs.items()},
            predicted_label=media_result["predicted_label"],
            incongruence=round(media_result["incongruence"], 4),
            modality_contributions=media_result["modality_contributions"],
            scored_modalities=media_result["scored_modalities"],
            risk_level=_risk_level(finding_prob, high_risk),
            high_risk=high_risk,
            trained=media_result["trained"],
        )
    elif text_risk is not None:
        pd = text_risk.depression_probability
        report.update(
            analysis_available=True,
            method="Sözlük tabanlı metin ön analizi (prototip)",
            probabilities={"kontrol": round(1 - pd, 4), "depresyon": pd},
            predicted_label="depresyon" if pd >= 0.5 else "kontrol",
            risk_level=_risk_level(pd, high_risk),
            high_risk=high_risk,
            trained=False,
        )
    else:
        report.update(
            analysis_available=False,
            method="-",
            high_risk=False,
            trained=False,
            note="Otomatik skorlama için transkript girin veya ses/görüntü "
            "modalite ekstralarını (vision/audio) kurun.",
        )

    audit.log_event(
        "session_analyze",
        subject=report["pseudonym"],
        detail={
            "modalities": modalities,
            "analysis": report["analysis_available"],
            "scored": report["scored_modalities"],
        },
    )
    return report


__all__ = ["analyze_session", "score_media"]
