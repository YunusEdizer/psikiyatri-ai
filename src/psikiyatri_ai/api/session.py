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
    "Prototip (TRL6): Tanı, eğitilmiş metin modeliyle üretilir. Ses ve görüntü "
    "kodlayıcıları henüz etiketli medya ile eğitilmediğinden tanı füzyonuna "
    "KATILMAZ (rastgele kodlayıcı tanıyı bozacağından); kaydedilen ses "
    "betimsel akustik gözlem olarak sunulur, yüz analizi hattı geliştirme "
    "aşamasındadır. Çok-modaliteli tanı ve uyumsuzluk analizi, etiketli "
    "ses/görüntü verisi toplandığında etkinleşecektir."
)


# Çekimserlik eşikleri (kontrol sınıfı olmayan model için):
# - _ABSTAIN_TOP1: en yüksek olasılık bunun altındaysa her koşulda "belirsiz"
#   (8 sınıfta şans taban çizgisi 0.125; 0.35 ≈ şansın ~3 katı güven ister).
# - _ABSTAIN_NO_INDICATOR: sözlük-tabanlı ön analiz transkriptte HİÇBİR
#   belirti ifadesi bulamamışsa modelden daha yüksek güven istenir; iki
#   bağımsız kanıt hattı da zayıfken tanı öne sürmek yanıltıcı olur.
_ABSTAIN_TOP1 = 0.35
_ABSTAIN_NO_INDICATOR = 0.60


def _risk_level(prob: float, high_risk: bool) -> str:
    if high_risk or prob >= 0.66:
        return "yüksek"
    if prob >= 0.40:
        return "orta"
    return "düşük"


# Yüklü çıkarım modeli önbelleği: (modaliteler, egemaps, ckpt mtime) -> model.
# Metin kodlayıcı (BERTurk) ve checkpoint yüklemesi pahalıdır; aynı modalite
# kümesiyle gelen ardışık isteklerde yeniden kurulmasın.
_model_cache: dict[tuple, tuple] = {}


def _get_inference_model(scored: tuple[str, ...], use_egemaps: bool):
    """Eğitilmiş füzyon modelini (varsa) önbellekten kurar/döndürür."""
    import torch

    from ..config import get_settings
    from ..training.checkpoint import labels_for_state, load_metadata
    from ..training.model import PsikiyatriModel

    settings = get_settings()
    ckpt = settings.model_dir / "psikiyatri_model.pt"
    mtime = ckpt.stat().st_mtime if ckpt.exists() else None
    key = (scored, use_egemaps, mtime)
    if key in _model_cache:
        return _model_cache[key]

    state = torch.load(ckpt, map_location="cpu") if ckpt.exists() else None
    meta = load_metadata(settings.model_dir)
    if meta:
        classes = list(meta["classes"])
    elif state is not None:
        classes = labels_for_state(state, ["kontrol", "depresyon"])
    else:
        classes = ["kontrol", "depresyon"]
    model_cfg = settings.model.model_copy(update={"num_classes": len(classes)})
    model = PsikiyatriModel(model_cfg, enabled_modalities=scored, use_egemaps=use_egemaps)
    trained = state is not None
    if state is not None:
        model.load_state_dict(state, strict=False)
    model.eval()

    _model_cache.clear()  # tek giriş yeterli; eski checkpoint'i bellekte tutma
    _model_cache[key] = (model, classes, trained)
    return model, classes, trained


def score_media(media_paths: dict[str, Path], transcript: str | None = None) -> dict | None:
    """Kayıtları (ve varsa transkripti) gerçek füzyon modelinden geçirir.

    Metin, ``transformers`` kuruluysa BERTurk kodlayıcısıyla skorlanır;
    ilgili modalite bağımlılığı yoksa (ör. mediapipe/librosa kurulu değil)
    o modalite atlanır. Hiçbir modalite skorlanamazsa ``None`` döner
    (çağıran sözlük-tabanlı metin analizine düşer).
    """
    from ..data.batch import SampleFeatures
    from ..data.datamodule import FeatureExtractor, collate_features
    from ..data.schema import Sample
    from ..logging_conf import logger

    candidates = [m for m in ("vision", "audio", "wearable") if media_paths.get(m)]
    has_text = bool(transcript and transcript.strip())
    if has_text:
        candidates.append("text")
    if not candidates:
        return None

    sample = Sample(
        pseudonym="anonim",
        session_id="canli",
        video_path=media_paths.get("vision"),
        audio_path=media_paths.get("audio"),
        wearable_path=media_paths.get("wearable"),
        transcript=transcript if has_text else None,
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

    import torch

    model, classes, trained = _get_inference_model(
        tuple(scored), "audio_egemaps" in feats
    )

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
        "audio_observation": None,
        "vision_observation": None,
        "vision_pending": False,
        "disclaimer": DISCLAIMER,
        "prototype_note": PROTOTYPE_NOTE,
    }

    # 1) Metin ön analizi (sözlük tabanlı) — her zaman denenir; risk
    #    göstergeleri (kendine zarar vb.) derin skorlamadan bağımsız işler.
    text_risk = None
    masked = None
    if has_transcript:
        masked = mask_pii(normalize_text(transcript_text))
        text_risk = analyze_text_risk(masked)
        report["transcript_preview"] = masked[:500]
        report["matched_indicators"] = text_risk.matched

    # 2) Derin öğrenme skorlaması. ÖNEMLİ: ses/görüntü kodlayıcıları henüz
    #    etiketli medya ile EĞİTİLMEDİĞİNDEN (korpus metin-only) tanı füzyonuna
    #    yalnızca eğitilmiş modaliteler (metin; + varsa giyilebilir) verilir.
    #    Ampirik olarak doğrulandı: rastgele-başlatılmış ses kodlayıcısı metin
    #    tanısını bozuyor (ör. depresyon 0.78 → 0.43). Ses/görüntü aşağıda
    #    (adım 3) BETİMSEL gözlem olarak işlenir; tanıya karıştırılmaz.
    diagnostic_paths = {k: v for k, v in media_paths.items() if k == "wearable"}
    media_result = None
    if diagnostic_paths or has_transcript:
        try:
            media_result = score_media(diagnostic_paths, transcript=masked)
        except Exception:  # noqa: BLE001 - skorlama isteği asla kesmemeli
            media_result = None

    # 3) Betimsel akustik gözlem (tanı DEĞİL): kaydedilen sesten perde,
    #    enerji ve seslilik gibi doğrudan yorumlanabilir sinyaller.
    if media_paths.get("audio"):
        try:
            from ..modalities.audio.observations import (
                describe_audio,
                observation_to_dict,
            )

            report["audio_observation"] = observation_to_dict(
                describe_audio(media_paths["audio"])
            )
        except Exception:  # noqa: BLE001 - gözlem hattı tanıyı kesmemeli
            report["audio_observation"] = None
    if media_paths.get("vision"):
        # Betimsel yüz gözlemi (tanı DEĞİL): MediaPipe FaceLandmarker
        # blendshape'lerinden yüz saptanma oranı, göz kırpma, ifade
        # değişkenliği, gülümseme, kaş hareketi. Tanı füzyonuna katılmaz.
        try:
            from ..modalities.vision.observations import (
                describe_video,
            )
            from ..modalities.vision.observations import (
                observation_to_dict as face_obs_to_dict,
            )

            obs = face_obs_to_dict(describe_video(media_paths["vision"]))
            report["vision_observation"] = obs
            report["vision_pending"] = obs is None  # model/okuma yoksa beklemede
        except Exception:  # noqa: BLE001 - gözlem hattı tanıyı kesmemeli
            report["vision_pending"] = True

    high_risk = bool(text_risk.high_risk) if text_risk else False

    if media_result is not None:
        probs = media_result["probabilities"]
        predicted = media_result["predicted_label"]
        top1 = float(max(probs.values()))
        # 'kontrol' sınıfı olmayan modelde (yalnız tanı sınıfları) bulgu
        # olasılığı en güçlü tanının olasılığıyla temsil edilir.
        if "kontrol" in probs:
            finding_prob = 1.0 - probs.get("kontrol", 0.0)
        else:
            finding_prob = top1
            # Çekimserlik (abstention): model "sağlıklı" diyemediğinden,
            # düşük güvenli bir tahmini tanı gibi sunmak yanıltıcı olur.
            # Sözlük analizi de belirti bulamamışsa (bağımsız ikinci kanıt
            # hattı boş) daha yüksek güven aranır. "Belirsiz" durumda dağılım
            # yine gösterilir; nihai karar uzmana aittir.
            no_text_evidence = (
                text_risk is not None
                and "text" in media_result["scored_modalities"]
                and not text_risk.matched
            )
            if top1 < _ABSTAIN_TOP1 or (no_text_evidence and top1 < _ABSTAIN_NO_INDICATOR):
                predicted = "belirsiz"
        method = "Derin öğrenme füzyonu (" + " + ".join(media_result["scored_modalities"]) + ")"
        if not media_result["trained"]:
            method += " — eğitilmemiş model (demo)"
        elif predicted == "belirsiz":
            method += " — düşük güven, sonuç belirsiz"
        report.update(
            analysis_available=True,
            method=method,
            probabilities={k: round(v, 4) for k, v in probs.items()},
            predicted_label=predicted,
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
