"""Sentetik kontrol (sağlıklı) örnek üretimi testleri."""

from __future__ import annotations

from psikiyatri_ai.data.control_data import iter_control_samples
from psikiyatri_ai.data.schema import DiagnosisLabel


def test_generates_requested_count_and_label():
    samples = list(iter_control_samples(50))
    assert len(samples) == 50
    assert all(s.label is DiagnosisLabel.KONTROL for s in samples)
    assert all(s.consent is True for s in samples)


def test_transcript_has_all_24_answers():
    s = next(iter(iter_control_samples(1)))
    assert s.transcript.count("\n") == 23  # 24 satır = 23 satır sonu


def test_deterministic_with_seed():
    a = [s.transcript for s in iter_control_samples(10, seed=7)]
    b = [s.transcript for s in iter_control_samples(10, seed=7)]
    assert a == b


def test_variation_across_samples():
    # Tohumlu rastgele seçim çeşitlilik üretmeli (hepsi aynı olmamalı)
    transcripts = {s.transcript for s in iter_control_samples(30)}
    assert len(transcripts) > 1


def test_pseudonyms_are_anonymous():
    s = next(iter(iter_control_samples(1)))
    assert s.pseudonym.startswith("pid_")
    assert "KONTROL" not in s.pseudonym
