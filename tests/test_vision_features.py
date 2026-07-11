"""Görüntü geometrik özellik (AU-vekili) testleri — mediapipe gerektirmez."""

import numpy as np

from psikiyatri_ai.modalities.vision.preprocess import (
    GEOMETRIC_FEATURE_NAMES,
    NUM_FACE_LANDMARKS,
    compute_geometric_features,
    geometric_feature_vector,
)

# İlgili MediaPipe işaretçi indeksleri (preprocess ile aynı)
_CHEEK_L, _CHEEK_R = 234, 454
_RIGHT_EYE = (33, 160, 158, 133, 153, 144)


def _base_face() -> np.ndarray:
    lm = np.zeros((NUM_FACE_LANDMARKS, 3), dtype=np.float32)
    # Yüz genişliği (normalizasyon) sıfır olmasın
    lm[_CHEEK_L, :2] = (0.0, 0.5)
    lm[_CHEEK_R, :2] = (1.0, 0.5)
    return lm


def _set_eye(lm: np.ndarray, opening: float) -> None:
    p1, p2, p3, p4, p5, p6 = _RIGHT_EYE
    lm[p1, :2] = (0.40, 0.50)  # dış köşe
    lm[p4, :2] = (0.50, 0.50)  # iç köşe
    lm[p2, :2] = (0.42, 0.50 - opening)
    lm[p6, :2] = (0.42, 0.50 + opening)
    lm[p3, :2] = (0.47, 0.50 - opening)
    lm[p5, :2] = (0.47, 0.50 + opening)


def test_vector_length_and_keys():
    feats = compute_geometric_features(_base_face())
    assert set(feats) == set(GEOMETRIC_FEATURE_NAMES)
    assert geometric_feature_vector(_base_face()).shape == (len(GEOMETRIC_FEATURE_NAMES),)


def test_degenerate_returns_zeros():
    feats = compute_geometric_features(np.zeros((NUM_FACE_LANDMARKS, 3), dtype=np.float32))
    assert all(v == 0.0 for v in feats.values())


def test_eye_openness_responds():
    open_face = _base_face()
    _set_eye(open_face, opening=0.05)   # açık göz
    closed_face = _base_face()
    _set_eye(closed_face, opening=0.0)  # kapalı göz (kırpma)

    assert compute_geometric_features(open_face)["goz_acikligi"] > \
        compute_geometric_features(closed_face)["goz_acikligi"]


def test_features_are_finite():
    vec = geometric_feature_vector(_base_face())
    assert np.all(np.isfinite(vec))
