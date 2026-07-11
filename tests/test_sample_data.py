"""Sentetik veri üretimi + manifest üretimi testleri."""

import json

import torch

from psikiyatri_ai.data.datamodule import (
    FeatureExtractor,
    build_dataloaders,
    load_manifest,
)
from psikiyatri_ai.data.manifest_builder import build_manifest_from_raw
from psikiyatri_ai.data.synthetic_data import generate_sample_dataset
from psikiyatri_ai.training.model import PsikiyatriModel
from psikiyatri_ai.training.train import TrainConfig, fit


def test_generate_creates_manifest_and_features(tmp_path):
    out = tmp_path / "processed"
    manifest = generate_sample_dataset(count=8, seed=0, out_dir=out, data_root=tmp_path)
    assert manifest.exists()

    samples = load_manifest(manifest)
    assert len(samples) == 8
    # Sınıflar dengeli
    labels = {s.label.value for s in samples}
    assert labels == {"kontrol", "depresyon"}
    # Öznitelik dosyaları yazıldı
    assert (out / "features" / "00000.npz").exists()


def test_precomputed_feature_extraction(tmp_path):
    out = tmp_path / "processed"
    manifest = generate_sample_dataset(count=4, seed=1, out_dir=out, data_root=tmp_path)
    samples = load_manifest(manifest)

    extractor = FeatureExtractor(data_root=tmp_path, modalities=("vision", "audio"))
    feats = extractor.extract(samples[0])
    assert feats["vision"].shape == (8, 468, 3)
    assert feats["audio_mfcc"].shape == (50, 40)
    assert feats["audio_egemaps"].shape == (88,)


def test_end_to_end_learns_on_synthetic(tmp_path):
    """Sinyal içeren sentetik veride model 0.5'in üzerine çıkabilmeli."""
    torch.manual_seed(0)
    out = tmp_path / "processed"
    manifest = generate_sample_dataset(
        count=120, seed=3, out_dir=out, data_root=tmp_path, signal=1.2
    )
    train_loader, val_loader = build_dataloaders(
        manifest,
        modalities=("vision", "audio"),
        batch_size=16,
        val_split=0.25,
        seed=0,
        data_root=tmp_path,
    )
    model = PsikiyatriModel(enabled_modalities=("vision", "audio"))
    cfg = TrainConfig(epochs=8, modalities=("vision", "audio"))
    fit(model, train_loader, val_loader, cfg, torch.device("cpu"), model_dir=tmp_path)

    from psikiyatri_ai.training.evaluate import evaluate

    metrics = evaluate(model, val_loader)
    # Güçlü sinyalde model şansın (0.5) belirgin üzerine çıkmalı
    assert metrics.balanced_accuracy > 0.6


def test_build_manifest_from_raw_anonymizes(tmp_path):
    # Sahte ham seans oluştur
    raw = tmp_path / "raw"
    sess = raw / "S001"
    sess.mkdir(parents=True)
    (sess / "transcript.txt").write_text(
        "Hasta 12345678901 numarali kayit.", encoding="utf-8"
    )
    (sess / "meta.json").write_text(
        json.dumps({"patient_id": "AHMET-123", "label": "depresyon"}), encoding="utf-8"
    )

    out = tmp_path / "manifest.jsonl"
    build_manifest_from_raw(raw, out_path=out, data_root=tmp_path)
    content = out.read_text(encoding="utf-8")

    # Doğrudan tanımlayıcı sızmamalı; TC maskeli; takma-kimlik üretilmiş
    assert "AHMET-123" not in content
    assert "12345678901" not in content
    assert "pid_" in content
    assert "[TCKN]" in content
