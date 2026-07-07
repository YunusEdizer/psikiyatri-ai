"""Akustik özellik çıkarımı.

Ses tonundaki (frekans, hız, tonlama) psikiyatrik belirteçleri yakalamak
için iki tamamlayıcı öznitelik kümesi:

- **MFCC**: spektral zarf; genel akustik karakter.
- **eGeMAPS**: duygu/afekt hesaplama için standart minimalist parametre
  seti (Eyben ve diğ., 2015) — proje metninde açıkça referans verilmiştir.

Ağır bağımlılıklar (librosa, opensmile) tembel (lazy) yüklenir.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_SAMPLE_RATE = 16_000
N_MFCC = 40


@dataclass
class AcousticFeatures:
    """Bir ses kaydından çıkarılmış akustik öznitelikler."""

    mfcc: np.ndarray  # şekil: (N_MFCC, T)
    egemaps: np.ndarray | None = None  # şekil: (88,) — eGeMAPS fonksiyonelleri
    sample_rate: int = DEFAULT_SAMPLE_RATE


def extract_mfcc(audio_path: Path, *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Ses dosyasından MFCC dizisi çıkarır, şekil (N_MFCC, T)."""
    try:
        import librosa  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Ses modalitesi için 'librosa' gerekli. "
            "Kurulum: pip install 'psikiyatri-ai[audio]'"
        ) from exc

    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).astype(np.float32)


def extract_egemaps(audio_path: Path) -> np.ndarray:
    """eGeMAPSv02 fonksiyonel öznitelik vektörünü (88 boyut) çıkarır."""
    try:
        import opensmile  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "eGeMAPS için 'opensmile' gerekli. "
            "Kurulum: pip install 'psikiyatri-ai[audio]'"
        ) from exc

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    df = smile.process_file(str(audio_path))
    return df.to_numpy().astype(np.float32).ravel()


def extract_acoustic_features(audio_path: Path, *, with_egemaps: bool = True) -> AcousticFeatures:
    """MFCC ve (isteğe bağlı) eGeMAPS özniteliklerini birlikte çıkarır."""
    mfcc = extract_mfcc(audio_path)
    egemaps = extract_egemaps(audio_path) if with_egemaps else None
    return AcousticFeatures(mfcc=mfcc, egemaps=egemaps)


__all__ = [
    "AcousticFeatures",
    "extract_acoustic_features",
    "extract_egemaps",
    "extract_mfcc",
    "N_MFCC",
]
