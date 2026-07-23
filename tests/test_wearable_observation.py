"""Betimsel giyilebilir sensör gözlemi testleri."""

from __future__ import annotations

import numpy as np

from psikiyatri_ai.modalities.wearable import observations as wo
from psikiyatri_ai.modalities.wearable.features import NUM_WEARABLE_CHANNELS, WEARABLE_CHANNELS


def test_describe_wearable_from_npz(tmp_path, monkeypatch):
    # 20 örnek x kanal serisi; yüksek nabız + yüksek uyarılma
    series = np.tile([110, 0.3, 0.8, 0.7, 0.4, 0.1, 0.2, 0.3, 0.5], (20, 1)).astype("float32")
    from psikiyatri_ai.modalities.wearable import features as wf

    monkeypatch.setattr(
        wf, "extract_wearable_features",
        lambda p, **k: wf.WearableFeatures(series=series),
    )
    obs = wo.describe_wearable("dummy.npz")
    assert obs is not None
    assert obs.samples == 20
    assert obs.qualitative["nabiz"] == "yüksek"       # 110 bpm
    assert obs.qualitative["uyarilma"] == "yüksek"    # cilt iletkenliği 0.8
    assert obs.dominant_eeg_band in ("delta", "theta", "alpha", "beta")


def test_describe_wearable_empty_returns_none(monkeypatch):
    from psikiyatri_ai.modalities.wearable import features as wf

    monkeypatch.setattr(
        wf, "extract_wearable_features",
        lambda p, **k: wf.WearableFeatures(series=np.zeros((0, NUM_WEARABLE_CHANNELS), "float32")),
    )
    assert wo.describe_wearable("x.npz") is None


def test_channels_intact():
    assert "kalp_atis_hizi" in WEARABLE_CHANNELS and "cilt_iletkenligi" in WEARABLE_CHANNELS


def test_observation_to_dict_none():
    assert wo.observation_to_dict(None) is None
