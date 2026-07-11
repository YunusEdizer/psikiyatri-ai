"""Çok-sınıflı (çok-hastalık) boru hattı testleri."""

import torch
from torch.utils.data import DataLoader

from psikiyatri_ai.config import ModelSettings
from psikiyatri_ai.data.datamodule import (
    SyntheticDataset,
    collate_features,
    infer_classes,
)
from psikiyatri_ai.data.schema import DiagnosisLabel, Sample
from psikiyatri_ai.training.checkpoint import (
    labels_for_state,
    load_metadata,
    num_classes_from_state,
    save_metadata,
)
from psikiyatri_ai.training.model import PsikiyatriModel


def test_infer_classes_canonical_order():
    samples = [
        Sample(pseudonym="a", session_id="1", label=DiagnosisLabel.BIPOLAR),
        Sample(pseudonym="b", session_id="2", label=DiagnosisLabel.KONTROL),
        Sample(pseudonym="c", session_id="3", label=DiagnosisLabel.DEPRESYON),
    ]
    # Kontrol her zaman ilk; kanonik sıra korunur
    assert infer_classes(samples) == ["kontrol", "depresyon", "bipolar"]


def test_multiclass_model_forward():
    cfg = ModelSettings(num_classes=5)
    model = PsikiyatriModel(cfg, enabled_modalities=("vision", "audio"))
    ds = SyntheticDataset(size=6, num_classes=5, modalities=("vision", "audio"), seed=0)
    loader = DataLoader(ds, batch_size=6, collate_fn=collate_features)
    batch, _ = next(iter(loader))
    out = model(batch)
    assert out.logits.shape == (6, 5)


def test_checkpoint_metadata_roundtrip(tmp_path):
    save_metadata(tmp_path, ["kontrol", "depresyon", "anksiyete"], ["audio", "wearable"])
    meta = load_metadata(tmp_path)
    assert meta["num_classes"] == 3
    assert meta["classes"][0] == "kontrol"
    assert meta["modalities"] == ["audio", "wearable"]


def test_load_metadata_missing_returns_none(tmp_path):
    assert load_metadata(tmp_path) is None


def test_num_classes_derived_from_checkpoint_state():
    # classes.json yoksa servis, num_classes'ı checkpoint başından türetir
    # (boyut uyuşmazlığı ile yükleme çökmesini önler).
    state5 = {"fusion.classifier.3.weight": torch.zeros(5, 512)}
    assert num_classes_from_state(state5, 2) == 5
    assert len(labels_for_state(state5, ["kontrol", "depresyon"])) == 5
    # Varsayılanla aynı sayıdaysa varsayılan etiketler korunur
    state2 = {"fusion.classifier.3.weight": torch.zeros(2, 512)}
    assert labels_for_state(state2, ["kontrol", "depresyon"]) == ["kontrol", "depresyon"]
    # Anahtar yoksa varsayılana düşer
    assert num_classes_from_state({}, 2) == 2


def test_multiclass_learns_on_synthetic(tmp_path):
    """5 sınıflı sentetik veride model şansın (0.2) belirgin üzerine çıkmalı."""
    from psikiyatri_ai.data.datamodule import build_dataloaders
    from psikiyatri_ai.data.synthetic_data import MULTICLASS, generate_sample_dataset
    from psikiyatri_ai.training.evaluate import evaluate
    from psikiyatri_ai.training.train import TrainConfig, fit

    torch.manual_seed(0)
    out = tmp_path / "processed"
    manifest = generate_sample_dataset(
        count=150, seed=3, out_dir=out, data_root=tmp_path,
        signal=1.5, classes=MULTICLASS, include_wearable=True,
    )
    mods = ("vision", "audio", "wearable")
    class_names = tuple(c.value for c in MULTICLASS)
    train_loader, val_loader = build_dataloaders(
        manifest, classes=class_names, modalities=mods,
        batch_size=16, val_split=0.25, seed=0, data_root=tmp_path,
    )
    cfg_model = ModelSettings(num_classes=len(MULTICLASS))
    model = PsikiyatriModel(cfg_model, enabled_modalities=mods)
    cfg = TrainConfig(epochs=10, modalities=mods, classes=class_names)
    fit(model, train_loader, val_loader, cfg, torch.device("cpu"), model_dir=tmp_path)
    metrics = evaluate(model, val_loader)
    assert metrics.balanced_accuracy > 0.4  # şans 0.2
