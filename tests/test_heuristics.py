"""Türkçe metin ön analizi (sözlük tabanlı) testleri."""

from psikiyatri_ai.modalities.text.heuristics import analyze_text_risk


def test_neutral_text_low_risk():
    r = analyze_text_risk("Bugün hava güzeldi, işe gittim ve arkadaşlarımla buluştum.")
    assert r.depression_probability < 0.3
    assert not r.high_risk


def test_depressive_text_higher_risk():
    r = analyze_text_risk(
        "Kendimi çok umutsuz ve değersiz hissediyorum, hiçbir şeyden zevk almıyorum, "
        "sürekli yorgunum ve uyuyamıyorum."
    )
    assert r.depression_probability > 0.5
    assert len(r.matched) >= 2


def test_high_risk_flag():
    r = analyze_text_risk("Artık yaşamak istemiyorum.")
    assert r.high_risk is True
    assert "yüksek risk" in r.matched


def test_case_insensitive_turkish():
    r = analyze_text_risk("UMUTSUZUM ve KARAMSAR bir haldeyim")
    assert r.matched  # büyük harfli Türkçe de eşleşmeli


def test_probability_in_range():
    r = analyze_text_risk("panik korku endişe gergin huzursuz tedirgin")
    assert 0.0 <= r.depression_probability <= 1.0
