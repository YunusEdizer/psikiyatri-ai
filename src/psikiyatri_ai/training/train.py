"""Uçtan uca eğitim döngüsü.

CLI: ``psikiyatri-train`` (bkz. pyproject [project.scripts]).

Gerçek veri için ``PAI_DATA_DIR/processed/manifest.jsonl`` beklenir.
Manifest yoksa, hattın uçtan uca çalıştığını kanıtlayan bir **sentetik
smoke eğitimi** koşulur (dosya/ağır bağımlılık gerektirmez).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import get_settings
from ..data.batch import ModalityBatch
from ..data.datamodule import (
    DEFAULT_CLASSES,
    SyntheticDataset,
    build_dataloaders,
    collate_features,
)
from ..logging_conf import logger, setup_logging
from ..utils.metrics import compute_metrics
from .checkpoint import save_metadata
from .model import PsikiyatriModel


@dataclass
class TrainConfig:
    epochs: int = 20
    learning_rate: float = 3e-4
    batch_size: int = 16
    val_split: float = 0.2
    seed: int = 42
    modalities: tuple[str, ...] = ("vision", "audio", "text")
    classes: tuple[str, ...] = DEFAULT_CLASSES


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
def validate(
    model: PsikiyatriModel,
    loader: DataLoader,
    device: torch.device,
    *,
    class_names: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Doğrulama kümesinde metrikleri hesaplar (ikili veya çok-sınıflı)."""
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

    return compute_metrics(
        np.array(y_true),
        np.array(y_pred),
        num_classes=model.cfg.num_classes,
        class_names=class_names,
    ).as_dict()


def _batch_to(batch: ModalityBatch, device: torch.device) -> ModalityBatch:
    return batch.to(device)


def fit(
    model: PsikiyatriModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: TrainConfig,
    device: torch.device,
    *,
    model_dir: Path | None = None,
) -> PsikiyatriModel:
    """Modeli verilen konfigürasyonla eğitir; en iyi doğrulama modelini saklar.

    ``model_dir`` verilmezse checkpoint ve sınıf yan-verisi
    ``get_settings().model_dir`` altına yazılır (test için özelleştirilebilir).
    """
    model.to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=cfg.learning_rate
    )
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(model_dir) if model_dir is not None else get_settings().model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "psikiyatri_model.pt"
    # Servisin etiket sırasını eşleştirebilmesi için sınıf/modalite yan-verisi
    save_metadata(out_dir, cfg.classes, cfg.modalities)
    best_metric = -1.0

    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        msg = f"epoch {epoch:>3}/{cfg.epochs} | kayıp={loss:.4f}"
        if val_loader is not None:
            metrics = validate(model, val_loader, device, class_names=cfg.classes)
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
    import argparse

    parser = argparse.ArgumentParser(description="Multimodal psikiyatri modelini eğit.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Eğitim manifest yolu (varsayılan: <data>/processed/manifest.jsonl)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Epoch sayısı")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch boyutu")
    parser.add_argument(
        "--unfreeze-text-layers",
        type=int,
        default=None,
        help="BERTurk son N transformer katmanını ince ayara aç (0 = tam donuk)",
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Öğrenme oranı (ince ayarda düşük tutun)"
    )
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()
    device = _select_device()
    logger.info("Cihaz: {} | Ortam: {}", device, settings.env)

    manifest = args.manifest or (settings.processed_data_dir / "manifest.jsonl")
    if manifest.exists():
        from ..data.datamodule import infer_classes, load_manifest

        logger.info("Manifest bulundu: {}", manifest)
        modalities, use_egemaps = _infer_modalities(manifest)
        classes = tuple(infer_classes(load_manifest(manifest)))
        if len(classes) < 2:
            # En az iki sınıf garanti et; mevcut (varsayılan-dışı olabilen)
            # etiketi ATMA — aksi hâlde LabelEncoder KeyError verir.
            classes = tuple(dict.fromkeys((*DEFAULT_CLASSES, *classes)))
        logger.info("Algılanan modaliteler: {} | sınıflar: {}", modalities, classes)

        cfg = TrainConfig(modalities=modalities, classes=classes)
        if args.epochs is not None:
            cfg.epochs = args.epochs
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        if args.lr is not None:
            cfg.learning_rate = args.lr
        model_update = {"num_classes": len(classes)}
        if args.unfreeze_text_layers is not None:
            model_update["text_unfreeze_layers"] = args.unfreeze_text_layers
            logger.info(
                "Metin ince ayarı: son {} BERTurk katmanı açık", args.unfreeze_text_layers
            )
        model_cfg = settings.model.model_copy(update=model_update)
        train_loader, val_loader = build_dataloaders(
            manifest,
            classes=classes,
            batch_size=cfg.batch_size,
            val_split=cfg.val_split,
            seed=cfg.seed,
            modalities=modalities,
        )
        model = PsikiyatriModel(
            model_cfg, enabled_modalities=modalities, use_egemaps=use_egemaps
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
    feats = FeatureExtractor(
        modalities=("vision", "audio", "text", "wearable")
    ).extract(samples[0])
    mods: list[str] = []
    if "vision" in feats:
        mods.append("vision")
    if "audio_mfcc" in feats:
        mods.append("audio")
    if "wearable" in feats:
        mods.append("wearable")
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
