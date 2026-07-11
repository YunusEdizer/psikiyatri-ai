"""Model değerlendirme yardımcıları."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.batch import ModalityBatch
from ..utils.metrics import ClassificationMetrics, compute_metrics
from .model import PsikiyatriModel


@torch.no_grad()
def evaluate(
    model: PsikiyatriModel,
    loader: DataLoader,
    device: torch.device | str = "cpu",
    *,
    class_names: list[str] | None = None,
) -> ClassificationMetrics:
    """Modeli bir DataLoader üzerinde değerlendirir (ikili/çok-sınıflı metrikler)."""
    model.eval()
    model.to(device)
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch, labels in loader:
        batch_on_device: ModalityBatch = batch.to(device)
        out = model(batch_on_device)
        y_pred.extend(out.logits.argmax(dim=-1).cpu().tolist())
        y_true.extend(labels.tolist())
    return compute_metrics(
        np.array(y_true),
        np.array(y_pred),
        num_classes=model.cfg.num_classes,
        class_names=class_names,
    )


__all__ = ["evaluate"]
