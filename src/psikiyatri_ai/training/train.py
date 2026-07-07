"""Uçtan uca eğitim döngüsü.

CLI: ``psikiyatri-train`` (bkz. pyproject [project.scripts]).

Gerçek veri için ``PAI_DATA_DIR/processed/manifest.jsonl`` beklenir.
Manifest yoksa, hattın uçtan uca çalıştığını kanıtlayan bir **sentetik
smoke eğitimi** koşulur (dosya/ağır bağımlılık gerektirmez).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import get_settings
from ..data.batch import ModalityBatch
from ..data.datamodule import SyntheticDataset, build_dataloaders, collate_features
from ..logging_conf import logger, setup_logging
from ..utils.metrics import binary_metrics
from .model import PsikiyatriModel


@dataclass
class TrainConfig:
    epochs: int = 20
    learning_rate: float = 3e-4
    batch_size: int = 16
    val_split: float = 0.2
    seed: int = 42
    modalities: tuple[str, ...] = ("vision", "audio", "text")


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: PsikiyatriModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Tek epoch eğitir, ortalama kaybı döndürür."""
    model.train()
    total, n = 0.0, 0
    for batch, labels in loader:
        batch = _batch_to(batch, device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out.logits, labels)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def validate(model: PsikiyatriModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Doğrulama kümesinde ikili metrikleri hesaplar."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch, labels in loader:
        batch = _batch_to(batch, device)
        out = model(batch)
        y_pred.extend(out.logits.argmax(dim=-1).cpu().tolist())
        y_true.extend(labels.tolist())
    if not y_true:
        return {}
    import numpy as np

    return binary_metrics(np.array(y_true), np.array(y_pred)).as_dict()


def _batch_to(batch: ModalityBatch, device: torch.device) -> ModalityBatch:
    return batch.to(device)


def fit(
    model: PsikiyatriModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: TrainConfig,
    device: torch.device,
) -> PsikiyatriModel:
    """Modeli verilen konfigürasyonla eğitir; en iyi doğrulama modelini saklar."""
    model.to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=cfg.learning_rate
    )
    criterion = nn.CrossEntropyLoss()

    settings = get_settings()
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = settings.model_dir / "psikiyatri_model.pt"
    best_metric = -1.0

    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        msg = f"epoch {epoch:>3}/{cfg.epochs} | kayıp={loss:.4f}"
        if val_loader is not None:
            metrics = validate(model, val_loader, device)
            bacc = metrics.get("balanced_accuracy", 0.0)
            msg += f" | denge_doğruluk={bacc:.3f} | duyarlılık={metrics.get('sensitivity', 0):.3f}"
            if bacc > best_metric:
                best_metric = bacc
                torch.save(model.state_dict(), ckpt_path)
                msg += " | ✔ kaydedildi"
        logger.info(msg)

    logger.info("Eğitim tamamlandı. En iyi denge_doğruluk={:.3f}", max(best_metric, 0.0))
    return model


def main() -> None:
    """CLI giriş noktası."""
    setup_logging()
    settings = get_settings()
    cfg = TrainConfig()
    device = _select_device()
    logger.info("Cihaz: {} | Ortam: {}", device, settings.env)

    manifest = settings.processed_data_dir / "manifest.jsonl"
    if manifest.exists():
        logger.info("Manifest bulundu: {}", manifest)
        modalities, use_egemaps = _infer_modalities(manifest)
        logger.info("Algılanan modaliteler: {}", modalities)
        train_loader, val_loader = build_dataloaders(
            manifest,
            batch_size=cfg.batch_size,
            val_split=cfg.val_split,
            seed=cfg.seed,
            modalities=modalities,
        )
        model = PsikiyatriModel(
            settings.model, enabled_modalities=modalities, use_egemaps=use_egemaps
        )
        fit(model, train_loader, val_loader, cfg, device)
    else:
        logger.warning(
            "Manifest yok ({}); SENTETİK smoke eğitimi çalıştırılıyor. "
            "Bu yalnızca hattın uçtan uca çalıştığını doğrular.",
            manifest,
        )
        _synthetic_smoke(device)


def _infer_modalities(manifest) -> tuple[tuple[str, ...], bool]:  # noqa: ANN001
    """Manifest'in ilk örneğinden mevcut modaliteleri ve eGeMAPS varlığını çıkarır."""
    from ..data.datamodule import FeatureExtractor, load_manifest

    samples = load_manifest(manifest)
    if not samples:
        raise ValueError("Manifest boş.")
    feats = FeatureExtractor(modalities=("vision", "audio", "text")).extract(samples[0])
    mods: list[str] = []
    if "vision" in feats:
        mods.append("vision")
    if "audio_mfcc" in feats:
        mods.append("audio")
    if "text_input_ids" in feats:
        mods.append("text")
    return tuple(mods), "audio_egemaps" in feats


def _synthetic_smoke(device: torch.device) -> None:
    """Sentetik veriyle kısa bir eğitim — hat doğrulaması."""
    modalities = ("vision", "audio")  # transformers gerektirmez
    train_ds = SyntheticDataset(size=48, modalities=modalities, seed=1)
    val_ds = SyntheticDataset(size=16, modalities=modalities, seed=2)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_features)
    val_loader = DataLoader(val_ds, batch_size=8, collate_fn=collate_features)

    model = PsikiyatriModel(enabled_modalities=modalities)
    cfg = TrainConfig(epochs=3, modalities=modalities)
    fit(model, train_loader, val_loader, cfg, device)


if __name__ == "__main__":
    main()
