"""Karar destek motoru testleri."""

from psikiyatri_ai.decision import generate_decision


def _base(**overrides) -> dict:
    """Tipik bir çağrının varsayılan girdileri."""
    kwargs = dict(
        probabilities={"kontrol": 0.2, "depresyon": 0.8},
        predicted_label="depresyon",
        risk_level="orta",
        high_risk=False,
        incongruence=0.3,
        scored_modalities=["audio"],
        matched_indicators={"çökkün duygudurum": ["umutsuz"]},
        trained=True,
        has_transcript=True,
    )
    kwargs.update(overrides)
    return generate_decision(**kwargs)


def test_profile_sorted_and_labeled():
    d = _base()
    profile = d["diagnosis_profile"]
    assert profile[0]["label"] == "depresyon"
    assert profile[0]["probability"] == 0.8
    assert "model tahmini" in profile[0]["support"]
    assert d["disclaimer"]


def test_high_risk_bumps_risk_and_recommends_protocol():
    d = _base(high_risk=True, risk_level="düşük")
    assert d["risk"]["level"] == "yüksek"
    assert any("Güvenlik planı" in r for r in d["recommendations"])


def test_ambiguous_probabilities_lower_confidence():
    d = _base(probabilities={"kontrol": 0.48, "depresyon": 0.52})
    assert d["confidence"]["level"] == "düşük"
    assert any("belirsiz" in w for w in d["confidence"]["warnings"])


def test_untrained_model_warns():
    d = _base(trained=False)
    assert any("Eğitilmiş model" in w for w in d["confidence"]["warnings"])
    assert d["confidence"]["level"] == "düşük"


def test_single_modality_warns():
    d = _base(scored_modalities=[], has_transcript=True)
    assert any("tek veri kaynağı" in w for w in d["confidence"]["warnings"])


def test_high_confidence_when_trained_multimodal_clear():
    d = _base(scored_modalities=["audio", "vision"], has_transcript=True)
    assert d["confidence"]["level"] == "yüksek"
    assert d["confidence"]["warnings"] == []


def test_chronic_duration_flagged():
    d = _base(symptom_duration_weeks=30)
    assert any("kronikleşme" in f for f in d["clinical_findings"])
    assert any("uzun belirti süresi" in f for f in d["risk"]["factors"])


def test_short_duration_notes_episode_criterion():
    d = _base(symptom_duration_weeks=1)
    assert any("2 haftadan kısa" in f for f in d["clinical_findings"])


def test_severe_case_bumps_risk_to_at_least_orta():
    d = _base(severity="şiddetli", risk_level="düşük")
    assert d["risk"]["level"] in ("orta", "yüksek")
    assert any("şiddetli" in f for f in d["risk"]["factors"])


def test_doctor_note_indicators_extracted():
    d = _base(doctor_note_masked="Hasta görüşmede umutsuz ve isteksiz görünüyordu.")
    assert any("Klinisyen notunda" in f for f in d["clinical_findings"])


def test_doctor_note_high_risk_bumps_risk():
    d = _base(risk_level="düşük", doctor_note_masked="Hasta intihar düşüncelerini dile getirdi.")
    assert d["risk"]["level"] == "yüksek"


def test_incongruence_finding():
    d = _base(incongruence=0.85)
    assert any("uyumsuzluk" in f for f in d["clinical_findings"])
