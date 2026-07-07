"""Model değerlendirme yardımcıları."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from ..utils.metrics import ClassificationMetrics, binary_metrics
from .model import PsikiyatriModel

Batch = tuple[dict[str, torch.Tensor], torch.Tensor]


@torch.no_grad()
def evaluate(model: PsikiyatriModel, batches: Iterable[Batch]) -> ClassificationMetrics:
    """Modeli değerlendirir ve ikili sınıflandırma metriklerini döndürür."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for embeddings, labels in batches:
        out = model(embeddings)
        preds = out.logits.argmax(dim=-1)
        y_pred.extend(preds.cpu().tolist())
        y_true.extend(labels.cpu().tolist())
    return binary_metrics(np.array(y_true), np.array(y_pred))


__all__ = ["evaluate"]
