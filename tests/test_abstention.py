"""Çekimserlik (abstention) ve kontrol-sınıfı-yok uyarısı testleri.

Kontrol sınıfı içermeyen modelde düşük güvenli tahminlerin "belirsiz"
olarak raporlandığını ve karar motorunun kısıtı uzmana bildirdiğini
doğrular (``score_media`` taklit edilir; ağır bağımlılık gerekmez).
"""

from __future__ import annotations

from psikiyatri_ai.api import session as session_mod
from psikiyatri_ai.decision.engine import generate_decision


def _fake_result(probs: dict[str, float]) -> dict:
    top = max(probs, key=probs.get)
    return {
        "probabilities": probs,
        "predicted_label": top,
        "incongruence": 0.1,
        "modality_contributions": [{"modality": "text", "weight": 1.0}],
        "scored_modalities": ["text"],
        "trained": True,
    }


def _analyze(
    monkeypatch, probs: dict[str, float], transcript: str = "Bir transkript."
) -> dict:
    monkeypatch.setattr(
        session_mod, "score_media", lambda media_paths, transcript=None: _fake_result(probs)
    )
    return session_mod.analyze_session("pid_t", "Dr", transcript, [], {})


def test_low_confidence_without_kontrol_abstains(monkeypatch):
    # 8 sınıf, en yüksek olasılık eşiğin altında -> belirsiz
    probs = {"depresyon": 0.20, "anksiyete": 0.18, "okb": 0.15, "tssb": 0.47 / 5}
    rest = ("panik", "bipolar", "sizofreni", "sosyal_anksiyete")
    probs.update({k: (1 - sum(probs.values())) / 4 for k in rest})
    report = _analyze(monkeypatch, probs)
    assert report["predicted_label"] == "belirsiz"
    assert "belirsiz" in report["method"]
    assert report["risk_level"] == "düşük"


def test_confident_prediction_kept(monkeypatch):
    report = _analyze(monkeypatch, {"depresyon": 0.72, "anksiyete": 0.28})
    assert report["predicted_label"] == "depresyon"
    assert "belirsiz" not in report["method"]


def test_mid_confidence_without_indicators_abstains(monkeypatch):
    # Sözlük belirti bulamıyor + model orta güvenli (< 0.60) -> belirsiz
    report = _analyze(monkeypatch, {"anksiyete": 0.45, "depresyon": 0.55})
    assert report["matched_indicators"] == {}
    assert report["predicted_label"] == "belirsiz"


def test_mid_confidence_with_indicators_kept(monkeypatch):
    # Aynı güven düzeyi ama sözlük belirti buluyor -> model tahmini korunur
    report = _analyze(
        monkeypatch,
        {"depresyon": 0.45, "anksiyete": 0.55},
        transcript="Kendimi çok mutsuz ve değersiz hissediyorum.",
    )
    assert report["matched_indicators"]
    assert report["predicted_label"] == "anksiyete"


def test_kontrol_class_disables_abstention(monkeypatch):
    # kontrol sınıfı varsa düşük olasılık zaten kontrol/bulgu dengesiyle anlatılır
    report = _analyze(monkeypatch, {"kontrol": 0.70, "depresyon": 0.30})
    assert report["predicted_label"] == "kontrol"


def test_engine_warns_when_no_kontrol_class():
    dec = generate_decision(
        probabilities={"depresyon": 0.6, "anksiyete": 0.4},
        predicted_label="depresyon",
        risk_level="düşük",
        high_risk=False,
        incongruence=None,
        scored_modalities=["text"],
        matched_indicators={},
        trained=True,
    )
    assert any("kontrol (sağlıklı) sınıfı içermiyor" in w for w in dec["confidence"]["warnings"])


def test_engine_no_warning_with_kontrol_class():
    dec = generate_decision(
        probabilities={"kontrol": 0.6, "depresyon": 0.4},
        predicted_label="kontrol",
        risk_level="düşük",
        high_risk=False,
        incongruence=None,
        scored_modalities=["text"],
        matched_indicators={},
        trained=True,
    )
    assert not any("sınıfı içermiyor" in w for w in dec["confidence"]["warnings"])
