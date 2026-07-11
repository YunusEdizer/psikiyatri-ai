"""Klinik karar destek motoru (kural tabanlı, açıklanabilir katman).

Model çıktısını (olasılıklar, uyumsuzluk, metin göstergeleri) görüşmede
girilen yapılandırılmış klinik verilerle (semptom süresi, vaka şiddeti,
doktor notu) birleştirir ve uzmana yönelik **açıklanabilir** bir tanı
profili üretir:

- olası tanı profili + her tanıya destek veren bulgular,
- risk düzeyi ve risk etkenleri,
- güven düzeyi + belirsizlik uyarıları (düşük güvenli sonuç mekanizması),
- klinik öneriler.

Çıktı kesin tanı değildir; uzman kararını destekleyen bir öneridir ve
her profile bu uyarı eklenir.
"""

from __future__ import annotations

from ..modalities.text.heuristics import analyze_text_risk

DISCLAIMER = (
    "Bu profil klinik karar desteği amaçlıdır ve kesin teşhis yerine geçmez. "
    "Nihai karar yetkili sağlık uzmanına aittir."
)

# Olasılık farkı bu eşiğin altındaysa sonuç "belirsiz" kabul edilir.
_AMBIGUITY_GAP = 0.15
# Uyumsuzluk bu eşiğin üstündeyse modaliteler arası çelişki bildirilir.
_INCONGRUENCE_THRESHOLD = 0.60
# Kronikleşme uyarısı için semptom süresi eşiği (~6 ay).
_CHRONIC_WEEKS = 26

_RISK_ORDER = ["düşük", "orta", "yüksek"]

_LABEL_TR = {
    "kontrol": "Kontrol (belirgin bulgu yok)",
    "depresyon": "Depresyon",
    "anksiyete": "Anksiyete",
    "bipolar": "Bipolar bozukluk",
    "sizofreni": "Şizofreni",
}


def _bump_risk(level: str, minimum: str) -> str:
    """Risk düzeyini en az `minimum` seviyesine yükseltir."""
    cur = _RISK_ORDER.index(level) if level in _RISK_ORDER else 0
    floor = _RISK_ORDER.index(minimum)
    return _RISK_ORDER[max(cur, floor)]


def generate_decision(
    *,
    probabilities: dict[str, float] | None,
    predicted_label: str | None,
    risk_level: str | None,
    high_risk: bool,
    incongruence: float | None,
    scored_modalities: list[str],
    matched_indicators: dict[str, list[str]],
    trained: bool,
    symptom_duration_weeks: int | None = None,
    severity: str | None = None,
    doctor_note_masked: str | None = None,
    has_transcript: bool = False,
) -> dict:
    """Model + klinik girdilerden açıklanabilir karar destek profili üretir."""
    findings: list[str] = []
    warnings: list[str] = []
    risk_factors: list[str] = []
    recommendations: list[str] = []

    # --- Tanı profili -------------------------------------------------------
    profile: list[dict] = []
    if probabilities:
        for label, prob in sorted(probabilities.items(), key=lambda kv: -kv[1]):
            support: list[str] = []
            if label == predicted_label:
                support.append("model tahmini")
            if label != "kontrol" and matched_indicators:
                cats = ", ".join(matched_indicators)
                support.append(f"metin göstergeleri: {cats}")
            profile.append(
                {
                    "label": label,
                    "label_tr": _LABEL_TR.get(label, label),
                    "probability": round(float(prob), 4),
                    "support": support,
                }
            )

    # --- Belirsizlik / güven ------------------------------------------------
    modality_count = len(scored_modalities) + (1 if has_transcript else 0)
    if not trained:
        warnings.append(
            "Eğitilmiş model ağırlığı yok — skorlar demo amaçlıdır, klinik değeri sınırlıdır."
        )
    if probabilities and len(probabilities) >= 2:
        top2 = sorted(probabilities.values(), reverse=True)[:2]
        if top2[0] - top2[1] < _AMBIGUITY_GAP:
            warnings.append(
                "En yüksek iki olasılık birbirine çok yakın "
                f"(fark < {_AMBIGUITY_GAP:.2f}) — sonuç belirsiz; ek klinik "
                "değerlendirme önerilir."
            )
    if modality_count <= 1:
        warnings.append(
            "Değerlendirme tek veri kaynağına dayanıyor — çoklu modalite "
            "(ses/görüntü/giyilebilir) eklenmesi güveni artırır."
        )
    if incongruence is not None and incongruence > _INCONGRUENCE_THRESHOLD:
        findings.append(
            "Modaliteler arası belirgin uyumsuzluk saptandı (ör. söylenen içerik "
            "ile ses tonu/yüz ifadesi çelişiyor olabilir) — bastırılmış duygulanım "
            "veya içsel çatışma yönünden görüşmede derinleştirin."
        )

    if not warnings and trained and modality_count >= 2:
        confidence = "yüksek"
    elif not trained or (
        probabilities
        and len(probabilities) >= 2
        and sorted(probabilities.values(), reverse=True)[0]
        - sorted(probabilities.values(), reverse=True)[1]
        < _AMBIGUITY_GAP
    ):
        confidence = "düşük"
    else:
        confidence = "orta"

    # --- Yapılandırılmış klinik girdiler -------------------------------------
    if symptom_duration_weeks is not None:
        if symptom_duration_weeks >= _CHRONIC_WEEKS:
            findings.append(
                f"Belirti süresi {symptom_duration_weeks} hafta — kronikleşme "
                "riski; tedavi planında süregenlik göz önüne alınmalı."
            )
            risk_factors.append("uzun belirti süresi (≥6 ay)")
        elif symptom_duration_weeks >= 2:
            findings.append(
                f"Belirti süresi {symptom_duration_weeks} hafta — depresif "
                "epizod süre ölçütüyle (≥2 hafta) uyumlu."
            )
        else:
            findings.append(
                "Belirti süresi 2 haftadan kısa — epizod süre ölçütü henüz "
                "karşılanmıyor olabilir; izlem önerilir."
            )

    if severity:
        sev = severity.strip().lower()
        if sev in ("şiddetli", "siddetli", "ağır", "agir"):
            risk_factors.append("klinisyen değerlendirmesi: şiddetli vaka")
        elif sev == "orta":
            findings.append("Klinisyen değerlendirmesi: orta şiddette vaka.")

    if doctor_note_masked and doctor_note_masked.strip():
        note_risk = analyze_text_risk(doctor_note_masked)
        if note_risk.matched:
            cats = ", ".join(note_risk.matched)
            findings.append(f"Klinisyen notunda belirti ifadeleri saptandı: {cats}.")
        if note_risk.high_risk:
            risk_factors.append("klinisyen notunda yüksek-risk ifadesi")

    if high_risk:
        risk_factors.append("konuşma içeriğinde yüksek-risk (kendine zarar) ifadesi")

    # --- Risk düzeyi ----------------------------------------------------------
    level = risk_level or "düşük"
    if high_risk or "klinisyen notunda yüksek-risk ifadesi" in risk_factors:
        level = _bump_risk(level, "yüksek")
    elif "klinisyen değerlendirmesi: şiddetli vaka" in risk_factors:
        level = _bump_risk(level, "orta")

    # --- Öneriler -------------------------------------------------------------
    if level == "yüksek":
        recommendations.append(
            "Güvenlik planı ve intihar riski değerlendirme protokolü uygulanmalı; "
            "gerekiyorsa acil psikiyatrik konsültasyon."
        )
    if confidence != "yüksek":
        recommendations.append(
            "Düşük/orta güvenli sonuç: bulguları yapılandırılmış klinik "
            "görüşme ve ölçeklerle (ör. HAM-D, GAD-7) doğrulayın."
        )
    if predicted_label and predicted_label != "kontrol":
        recommendations.append(
            "Model bulgusu ayırıcı tanı için başlangıç noktasıdır; komorbidite "
            "ve organik nedenler dışlanmalıdır."
        )
    recommendations.append("Nihai tanı ve tedavi kararı yetkili uzmana aittir.")

    return {
        "diagnosis_profile": profile,
        "risk": {"level": level, "factors": risk_factors},
        "confidence": {"level": confidence, "warnings": warnings},
        "clinical_findings": findings,
        "recommendations": recommendations,
        "disclaimer": DISCLAIMER,
    }


__all__ = ["generate_decision"]
