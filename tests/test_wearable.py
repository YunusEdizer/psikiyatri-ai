"""Giyilebilir modalite testleri: sinyal işleme, çıkarım, kodlayıcı, füzyon."""

import csv

import numpy as np
import pytest
import torch

from psikiyatri_ai.data.batch import ModalityBatch
from psikiyatri_ai.modalities.wearable.features import (
    NUM_WEARABLE_CHANNELS,
    WEARABLE_CHANNELS,
    eeg_band_powers,
    extract_wearable_features,
    tremor_index,
)
from psikiyatri_ai.modalities.wearable.model import WearableEncoder
from psikiyatri_ai.training.model import PsikiyatriModel


def test_eeg_band_powers_detects_alpha():
    fs = 128.0
    t = np.arange(0, 4, 1 / fs)
    signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz -> alpha bandı
    powers = eeg_band_powers(signal, fs)
    assert abs(sum(powers.values()) - 1.0) < 1e-6
    assert powers["alpha"] == max(powers.values())


def test_tremor_index_high_in_band():
    fs = 64.0
    t = np.arange(0, 4, 1 / fs)
    in_band = tremor_index(np.sin(2 * np.pi * 5 * t), fs)   # 5 Hz (3-6 bandı)
    out_band = tremor_index(np.sin(2 * np.pi * 20 * t), fs)  # 20 Hz
    assert in_band > 0.5
    assert in_band > out_band


def test_extract_from_csv(tmp_path):
    path = tmp_path / "wear.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(WEARABLE_CHANNELS)
        for _ in range(30):
            w.writerow([70, 0.5, 2.0, 0.3, 0.1, 0.2, 0.2, 0.3, 0.3])
    wf = extract_wearable_features(path)
    assert wf.series.shape == (30, NUM_WEARABLE_CHANNELS)
    assert "kalp_atis_hizi" in wf.summary


def test_extract_from_npz(tmp_path):
    path = tmp_path / "wear.npz"
    arr = np.random.default_rng(0).standard_normal((25, NUM_WEARABLE_CHANNELS)).astype("float32")
    np.savez(path, wearable=arr)
    wf = extract_wearable_features(path)
    assert wf.series.shape == (25, NUM_WEARABLE_CHANNELS)


def test_empty_csv_rejected(tmp_path):
    """Yalnız başlık içeren (T=0) CSV, downstream GRU'ya ulaşmadan reddedilmeli."""
    path = tmp_path / "empty.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerow(WEARABLE_CHANNELS)
    with pytest.raises(ValueError):
        extract_wearable_features(path)


def test_nan_inf_sanitized(tmp_path):
    """'nan'/'inf' metni float parse edilir; temizlenmezse seriye sızar."""
    path = tmp_path / "bad.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(WEARABLE_CHANNELS)
        w.writerow(["nan", "inf", "-inf", 0.3, 0.1, 0.2, 0.2, 0.3, 0.3])
        w.writerow([70, 0.5, 2.0, 0.3, 0.1, 0.2, 0.2, 0.3, 0.3])
    wf = extract_wearable_features(path)
    assert np.all(np.isfinite(wf.series))
    assert all(np.isfinite(v) for v in wf.summary.values())


def test_feature_extractor_surfaces_wearable(tmp_path):
    """Sentetik npz'de wearable varsa FeatureExtractor onu yüzeye çıkarmalı
    (train._infer_modalities bu mekanizmaya dayanır)."""
    from psikiyatri_ai.data.datamodule import FeatureExtractor, load_manifest
    from psikiyatri_ai.data.synthetic_data import generate_sample_dataset

    manifest = generate_sample_dataset(
        count=2, seed=0, out_dir=tmp_path / "p", data_root=tmp_path, include_wearable=True
    )
    sample = load_manifest(manifest)[0]
    feats = FeatureExtractor(
        data_root=tmp_path, modalities=("vision", "audio", "text", "wearable")
    ).extract(sample)
    assert "wearable" in feats


def test_encoder_output_shape():
    enc = WearableEncoder(embed_dim=128)
    out = enc(torch.randn(4, 60, NUM_WEARABLE_CHANNELS))
    assert out.shape == (4, 128)


def test_wearable_in_fusion_model():
    mods = ("audio", "wearable")
    model = PsikiyatriModel(enabled_modalities=mods)
    batch = ModalityBatch(
        audio_mfcc=torch.randn(3, 50, 40),
        audio_egemaps=torch.randn(3, 88),
        wearable=torch.randn(3, 60, NUM_WEARABLE_CHANNELS),
    )
    out = model(batch)
    assert out.logits.shape == (3, model.cfg.num_classes)
    assert out.attention.shape == (3, 2)  # audio + wearable
