"""Değerlendirme metrikleri.

Klinik bağlamda tek başına doğruluk (accuracy) yanıltıcıdır (sınıf
dengesizliği + hata maliyetlerinin asimetrisi). Bu yüzden duyarlılık
(sensitivity/recall), özgüllük (specificity) ve dengeli doğruluk da
raporlanır.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    sensitivity: float  # gerçek pozitif oranı (recall)
    specificity: float  # gerçek negatif oranı
    precision: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "precision": self.precision,
            "f1": self.f1,
        }


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    """İkili sınıflandırma metriklerini hesaplar (1 = pozitif/tanı var)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    def _safe_div(a: float, b: float) -> float:
        return a / b if b else 0.0

    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    balanced = (sensitivity + specificity) / 2
    f1 = _safe_div(2 * precision * sensitivity, precision + sensitivity)

    return ClassificationMetrics(
        accuracy=accuracy,
        balanced_accuracy=balanced,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        f1=f1,
    )


__all__ = ["ClassificationMetrics", "binary_metrics"]
