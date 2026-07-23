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


def test_confusion_matrix_shape_and_counts():
    import numpy as np

    from psikiyatri_ai.utils.metrics import confusion_matrix

    cm = confusion_matrix(np.array([0, 1, 2, 1]), np.array([0, 1, 1, 1]), num_classes=3)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1 and cm[1, 1] == 2 and cm[2, 1] == 1
    assert cm.sum() == 4


def test_ece_perfectly_calibrated_is_low():
    import numpy as np

    from psikiyatri_ai.utils.metrics import expected_calibration_error

    # %60 güven, %60 doğru → iyi kalibre → düşük ECE
    conf = np.array([0.6] * 10)
    correct = np.array([1] * 6 + [0] * 4)
    ece, bins = expected_calibration_error(conf, correct)
    assert ece < 0.05
    assert bins and bins[0]["ornek_sayisi"] == 10


def test_ece_overconfident_is_high():
    import numpy as np

    from psikiyatri_ai.utils.metrics import expected_calibration_error

    conf = np.array([0.9] * 10)
    correct = np.array([1] * 5 + [0] * 5)  # %90 güven ama %50 doğru
    ece, _ = expected_calibration_error(conf, correct)
    assert ece > 0.3


def test_fit_temperature_lowers_overconfidence():
    import torch

    from psikiyatri_ai.training.calibration import apply_temperature, fit_temperature

    logits = torch.tensor([[5.0, 0, 0], [0, 5.0, 0], [5.0, 0, 0], [0, 0, 5.0]])
    labels = torch.tensor([0, 2, 0, 2])  # yarısı yanlış → aşırı-güven
    t = fit_temperature(logits, labels)
    assert t > 1.0  # güveni düşürmeli
    # ölçekleme argmax'ı değiştirmez
    assert torch.equal(apply_temperature(logits, t).argmax(-1), logits.argmax(-1))


def test_fit_temperature_insufficient_data_returns_one():
    import torch

    from psikiyatri_ai.training.calibration import fit_temperature

    assert fit_temperature(torch.zeros(1, 3), torch.tensor([0])) == 1.0
