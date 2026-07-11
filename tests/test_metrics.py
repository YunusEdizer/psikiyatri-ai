"""Değerlendirme metrikleri testleri (ikili + çok-sınıflı)."""

import numpy as np

from psikiyatri_ai.utils.metrics import (
    binary_metrics,
    compute_metrics,
    multiclass_metrics,
)


def test_binary_perfect():
    m = binary_metrics(np.array([0, 1, 1, 0]), np.array([0, 1, 1, 0]))
    assert m.accuracy == 1.0
    assert m.sensitivity == 1.0
    assert m.specificity == 1.0
    assert m.balanced_accuracy == 1.0


def test_binary_all_wrong_specificity():
    # Tüm pozitifleri kaçır, tüm negatifleri yanlış pozitifle
    m = binary_metrics(np.array([1, 1, 0, 0]), np.array([0, 0, 1, 1]))
    assert m.sensitivity == 0.0
    assert m.specificity == 0.0


def test_binary_balanced_accuracy_single_class_fold():
    # y_true tek sınıf (hep pozitif) — sklearn semantiği: mevcut sınıfın
    # recall'ı = 1.0 (yarıya inmemeli).
    m = binary_metrics(np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]))
    assert m.balanced_accuracy == 1.0


def test_multiclass_perfect_and_per_class():
    y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    m = multiclass_metrics(y, y, num_classes=5, class_names=list("abcde"))
    assert m.accuracy == 1.0
    assert m.balanced_accuracy == 1.0
    assert set(m.per_class) == set("abcde")
    assert all(v["sensitivity"] == 1.0 for v in m.per_class.values())


def test_multiclass_partial():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 2, 1])  # son örnek 2->1 yanlış
    m = multiclass_metrics(y_true, y_pred, num_classes=3)
    assert 0.0 < m.balanced_accuracy < 1.0
    assert m.per_class["2"]["sensitivity"] == 0.5


def test_compute_metrics_dispatch():
    y = np.array([0, 1, 0, 1])
    # num_classes=2 => ikili yol; per_class boş
    assert compute_metrics(y, y, num_classes=2).per_class == {}
    # num_classes=3 => çok-sınıflı; per_class dolu
    assert compute_metrics(y, y, num_classes=3).per_class != {}
