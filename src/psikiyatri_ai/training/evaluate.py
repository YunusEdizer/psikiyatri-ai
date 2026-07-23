"""Model değerlendirme yardımcıları.

``evaluate`` temel metrikleri döndürür (geriye uyumlu). ``evaluate_full``
ise **dürüst** bir klinik değerlendirme raporu üretir: karışıklık matrisi,
sınıf-başına duyarlılık/özgüllük, ve olasılık **kalibrasyonu** (ECE).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.batch import ModalityBatch
from ..utils.metrics import (
    ClassificationMetrics,
    compute_metrics,
    confusion_matrix,
    expected_calibration_error,
)
from .model import PsikiyatriModel


@torch.no_grad()
def _collect(model: PsikiyatriModel, loader: DataLoader, device):  # noqa: ANN001, ANN202
    """DataLoader üzerinde logit, tahmin ve gerçek etiketleri toplar."""
    model.eval()
    model.to(device)
    logits_all: list[torch.Tensor] = []
    y_true: list[int] = []
    for batch, labels in loader:
        batch_on_device: ModalityBatch = batch.to(device)
        out = model(batch_on_device)
        logits_all.append(out.logits.cpu())
        y_true.extend(labels.tolist())
    logits = torch.cat(logits_all) if logits_all else torch.empty(0)
    return logits, np.array(y_true, dtype=int)


@torch.no_grad()
def evaluate(
    model: PsikiyatriModel,
    loader: DataLoader,
    device: torch.device | str = "cpu",
    *,
    class_names: list[str] | None = None,
) -> ClassificationMetrics:
    """Modeli bir DataLoader üzerinde değerlendirir (ikili/çok-sınıflı metrikler)."""
    logits, y_true = _collect(model, loader, device)
    y_pred = logits.argmax(dim=-1).numpy() if logits.numel() else np.array([], dtype=int)
    return compute_metrics(
        y_true, y_pred, num_classes=model.cfg.num_classes, class_names=class_names
    )


@dataclass
class EvaluationReport:
    """Dürüst değerlendirme raporu."""

    metrics: ClassificationMetrics
    confusion: list[list[int]]
    class_names: list[str]
    ece: float
    calibration_bins: list[dict]
    n_samples: int
    temperature: float | None = None  # uygulandıysa kalibrasyon sıcaklığı
    _extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "metrics": self.metrics.as_dict(),
            "per_class": self.metrics.per_class,
            "confusion": self.confusion,
            "class_names": self.class_names,
            "ece": self.ece,
            "temperature": self.temperature,
            "calibration_bins": self.calibration_bins,
        }


@torch.no_grad()
def evaluate_full(
    model: PsikiyatriModel,
    loader: DataLoader,
    device: torch.device | str = "cpu",
    *,
    class_names: list[str] | None = None,
    temperature: float | None = None,
) -> EvaluationReport:
    """Karışıklık matrisi + sınıf-başına metrikler + kalibrasyon (ECE) üretir.

    ``temperature`` verilirse olasılıklar kalibre edilerek ECE hesaplanır
    (tahmin/argmax değişmez, yalnızca güven ölçeklenir).
    """
    num_classes = model.cfg.num_classes
    names = class_names or [str(i) for i in range(num_classes)]
    logits, y_true = _collect(model, loader, device)
    n = int(y_true.shape[0])
    if n == 0:
        empty = compute_metrics(y_true, y_true, num_classes=num_classes, class_names=names)
        return EvaluationReport(empty, [], names, 0.0, [], 0)

    scaled = logits / max(float(temperature), 0.05) if temperature else logits
    probs = torch.softmax(scaled, dim=-1).numpy()
    y_pred = logits.argmax(dim=-1).numpy()

    metrics = compute_metrics(y_true, y_pred, num_classes=num_classes, class_names=names)
    conf = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    top_conf = probs.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    ece, bins = expected_calibration_error(top_conf, correct)

    return EvaluationReport(
        metrics=metrics,
        confusion=conf.tolist(),
        class_names=names,
        ece=ece,
        calibration_bins=bins,
        n_samples=n,
        temperature=temperature,
    )


def format_report(rep: EvaluationReport) -> str:
    """Raporu okunur bir metne çevirir (CLI çıktısı)."""
    m = rep.metrics
    lines = [
        f"Örnek sayısı        : {rep.n_samples}",
        f"Doğruluk            : {m.accuracy:.3f}",
        f"Denge doğruluğu     : {m.balanced_accuracy:.3f}",
        f"Makro duyarlılık    : {m.sensitivity:.3f}   Makro özgüllük: {m.specificity:.3f}",
        f"Makro F1            : {m.f1:.3f}",
        f"Kalibrasyon (ECE)   : {rep.ece:.3f}"
        + (f"   (T={rep.temperature:.2f})" if rep.temperature else "  (kalibre edilmemiş)"),
        "",
        "Sınıf-başına (duyarlılık / özgüllük / F1 / destek):",
    ]
    for name, s in m.per_class.items():
        lines.append(
            f"  {name:<18} {s['sensitivity']:.3f} / {s['specificity']:.3f} / "
            f"{s['f1']:.3f} / {int(s['support'])}"
        )
    # Karışıklık matrisi (kısa adlarla)
    lines.append("")
    lines.append("Karışıklık matrisi (satır=gerçek, sütun=tahmin):")
    short = [n[:6] for n in rep.class_names]
    lines.append("           " + " ".join(f"{s:>6}" for s in short))
    for i, row in enumerate(rep.confusion):
        lines.append(f"  {short[i]:>8} " + " ".join(f"{v:>6}" for v in row))
    return "\n".join(lines)


__all__ = ["EvaluationReport", "evaluate", "evaluate_full", "format_report"]
