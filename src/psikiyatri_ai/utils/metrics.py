"""Değerlendirme metrikleri.

Klinik bağlamda tek başına doğruluk (accuracy) yanıltıcıdır (sınıf
dengesizliği + hata maliyetlerinin asimetrisi). Bu yüzden duyarlılık
(sensitivity/recall), özgüllük (specificity) ve dengeli doğruluk da
raporlanır.

İki hesaplama yolu vardır:

- ``binary_metrics``       : ikili tasarım (1 = tanı-var/pozitif).
- ``multiclass_metrics``   : çok-sınıflı tasarım (ör. kontrol/depresyon/
  anksiyete/bipolar/şizofreni). Duyarlılık/özgüllük her sınıf için
  bire-karşı-hepsi (one-vs-rest) hesaplanıp makro ortalaması alınır;
  dengeli doğruluk = makro duyarlılık (sklearn ``balanced_accuracy``).

``compute_metrics`` sınıf sayısına göre uygun yolu seçen ortak arayüzdür.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    sensitivity: float  # gerçek pozitif oranı (recall); çok-sınıfta makro
    specificity: float  # gerçek negatif oranı; çok-sınıfta makro
    precision: float
    f1: float
    # Çok-sınıfta her sınıf için ayrıntılı skorlar (ikilide boş kalır)
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "precision": self.precision,
            "f1": self.f1,
        }


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    """İkili sınıflandırma metriklerini hesaplar (1 = pozitif/tanı var)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    # Dengeli doğruluk: yalnızca y_true'da mevcut sınıfların recall ortalaması
    # (sklearn balanced_accuracy semantiği; tek-sınıflı fold'da yarıya inmez).
    present_recalls = []
    if tp + fn > 0:  # pozitif sınıf mevcut -> recall = duyarlılık
        present_recalls.append(sensitivity)
    if tn + fp > 0:  # negatif sınıf mevcut -> recall = özgüllük
        present_recalls.append(specificity)
    balanced = float(np.mean(present_recalls)) if present_recalls else 0.0
    f1 = _safe_div(2 * precision * sensitivity, precision + sensitivity)

    return ClassificationMetrics(
        accuracy=accuracy,
        balanced_accuracy=balanced,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        f1=f1,
    )


def multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int | None = None,
    class_names: Sequence[str] | None = None,
) -> ClassificationMetrics:
    """Çok-sınıflı metrikleri bire-karşı-hepsi + makro ortalama ile hesaplar.

    Args:
        y_true, y_pred: tam sayı sınıf indeksleri.
        num_classes: sınıf sayısı (verilmezse etiketlerden çıkarılır).
        class_names: her indeks için okunur ad (``per_class`` anahtarları).

    Returns:
        ``ClassificationMetrics``. ``sensitivity``/``specificity``/``precision``/
        ``f1`` makro (sınıf başına eşit ağırlıklı) ortalamalardır. Dengeli
        doğruluk = destek gören sınıflar üzerinden makro duyarlılıktır.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if num_classes is None:
        max_idx = int(max(y_true.max(initial=0), y_pred.max(initial=0)))
        num_classes = max_idx + 1
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    total = int(y_true.shape[0])
    correct = int((y_true == y_pred).sum())
    accuracy = _safe_div(correct, total)

    recalls_with_support: list[float] = []
    macro_sens: list[float] = []
    macro_spec: list[float] = []
    macro_prec: list[float] = []
    macro_f1: list[float] = []
    per_class: dict[str, dict[str, float]] = {}

    for c in range(num_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        tn = total - tp - fp - fn
        support = tp + fn

        recall = _safe_div(tp, tp + fn)       # duyarlılık (sınıf c)
        spec = _safe_div(tn, tn + fp)         # özgüllük (sınıf c)
        prec = _safe_div(tp, tp + fp)
        f1 = _safe_div(2 * prec * recall, prec + recall)

        per_class[class_names[c]] = {
            "sensitivity": recall,
            "specificity": spec,
            "precision": prec,
            "f1": f1,
            "support": float(support),
        }
        macro_sens.append(recall)
        macro_spec.append(spec)
        macro_prec.append(prec)
        macro_f1.append(f1)
        if support > 0:
            recalls_with_support.append(recall)

    # Dengeli doğruluk: yalnızca destek gören sınıflar üzerinden makro duyarlılık
    balanced = float(np.mean(recalls_with_support)) if recalls_with_support else 0.0

    return ClassificationMetrics(
        accuracy=accuracy,
        balanced_accuracy=balanced,
        sensitivity=float(np.mean(macro_sens)) if macro_sens else 0.0,
        specificity=float(np.mean(macro_spec)) if macro_spec else 0.0,
        precision=float(np.mean(macro_prec)) if macro_prec else 0.0,
        f1=float(np.mean(macro_f1)) if macro_f1 else 0.0,
        per_class=per_class,
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int,
    class_names: Sequence[str] | None = None,
) -> ClassificationMetrics:
    """Sınıf sayısına göre ikili/çok-sınıflı metrik hesaplamayı seçer.

    ``num_classes == 2`` ise ikili semantik (1 = pozitif) korunur; aksi
    halde makro çok-sınıflı metrikler döner.
    """
    if num_classes <= 2:
        return binary_metrics(y_true, y_pred)
    return multiclass_metrics(
        y_true, y_pred, num_classes=num_classes, class_names=class_names
    )


__all__ = [
    "ClassificationMetrics",
    "binary_metrics",
    "compute_metrics",
    "multiclass_metrics",
]
