"""Uçtan uca eğitim döngüsü testleri (sentetik veri, hafif)."""

import torch
from torch.utils.data import DataLoader

from psikiyatri_ai.data.datamodule import SyntheticDataset, collate_features
from psikiyatri_ai.training.evaluate import evaluate
from psikiyatri_ai.training.model import PsikiyatriModel
from psikiyatri_ai.training.train import TrainConfig, fit


def _loaders():
    train_ds = SyntheticDataset(size=24, modalities=("vision", "audio"), seed=1)
    val_ds = SyntheticDataset(size=8, modalities=("vision", "audio"), seed=2)
    return (
        DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_features),
        DataLoader(val_ds, batch_size=8, collate_fn=collate_features),
    )


def test_model_forward_from_batch():
    ds = SyntheticDataset(size=4, modalities=("vision", "audio"), seed=0)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_features)
    batch, _ = next(iter(loader))
    model = PsikiyatriModel(enabled_modalities=("vision", "audio"))
    out = model(batch)
    assert out.logits.shape == (4, model.cfg.num_classes)
    assert out.attention.shape == (4, 2)


def test_fit_runs_and_updates_weights():
    torch.manual_seed(0)
    train_loader, val_loader = _loaders()
    model = PsikiyatriModel(enabled_modalities=("vision", "audio"))

    before = model.fusion.attn.weight.detach().clone()
    cfg = TrainConfig(epochs=2, modalities=("vision", "audio"))
    fit(model, train_loader, val_loader, cfg, torch.device("cpu"))
    after = model.fusion.attn.weight.detach()

    # Ağırlıklar güncellendiyse eğitim gerçekten çalışıyor demektir.
    assert not torch.allclose(before, after)


def test_evaluate_returns_metrics():
    _, val_loader = _loaders()
    model = PsikiyatriModel(enabled_modalities=("vision", "audio"))
    metrics = evaluate(model, val_loader)
    assert 0.0 <= metrics.balanced_accuracy <= 1.0
    assert 0.0 <= metrics.sensitivity <= 1.0
