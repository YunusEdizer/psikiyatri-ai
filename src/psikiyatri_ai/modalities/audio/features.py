"""Akustik özellik çıkarımı.

Ses tonundaki (frekans, hız, tonlama) psikiyatrik belirteçleri yakalamak
için iki tamamlayıcı öznitelik kümesi:

- **MFCC**: spektral zarf; genel akustik karakter.
- **eGeMAPS**: duygu/afekt hesaplama için standart minimalist parametre
  seti (Eyben ve diğ., 2015) — proje metninde açıkça referans verilmiştir.

Ağır bağımlılıklar (librosa, opensmile) tembel (lazy) yüklenir.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_SAMPLE_RATE = 16_000
N_MFCC = 40


@lru_cache(maxsize=1)
def ensure_ffmpeg_on_path() -> str | None:
    """Bir ffmpeg ikili dosyasını (imageio-ffmpeg) PATH'e ekler.

    Tarayıcıdan gelen kayıtlar WebM/Opus'tur; librosa bunları çözmek için
    ffmpeg'e ihtiyaç duyar. Sistemde ffmpeg PATH'te değilse imageio-ffmpeg
    ile gelen ikiliyi bulur, adını ``ffmpeg(.exe)`` olarak sağlar ve dizini
    PATH'e ekler. Bir kez çalışır (lru_cache). Bulunamazsa ``None`` döner.
    """
    from shutil import which

    if which("ffmpeg"):
        return which("ffmpeg")
    try:
        import imageio_ffmpeg  # noqa: PLC0415

        exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # noqa: BLE001 - imageio-ffmpeg yoksa sessiz geç
        return None
    if not exe.exists():
        return None
    # audioread 'ffmpeg' adını arar; imageio ikilisi farklı adlandırılmış olabilir.
    canonical = exe.parent / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not canonical.exists():
        try:
            canonical.write_bytes(exe.read_bytes())
            if os.name != "nt":
                canonical.chmod(0o755)
        except OSError:
            canonical = exe  # kopyalayamazsak en azından dizini PATH'e ekle
    os.environ["PATH"] = str(exe.parent) + os.pathsep + os.environ.get("PATH", "")
    return str(canonical)


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

    ensure_ffmpeg_on_path()
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
    """MFCC ve (isteğe bağlı) eGeMAPS özniteliklerini birlikte çıkarır.

    eGeMAPS çıkarımı yerel bir yardımcıya (opensmile → mediainfo) bağlıdır;
    bu bileşen kurulu değilse **MFCC korunarak** eGeMAPS atlanır (ses
    modalitesi tümden düşmez). Böylece temel akustik hat, opsiyonel afekt
    parametreleri olmadan da çalışır.
    """
    mfcc = extract_mfcc(audio_path)
    egemaps: np.ndarray | None = None
    if with_egemaps:
        try:
            egemaps = extract_egemaps(audio_path)
        except Exception as exc:  # noqa: BLE001 - opsiyonel yerel bağımlılık
            global _egemaps_warned
            if not _egemaps_warned:
                from ...logging_conf import logger  # noqa: PLC0415

                logger.warning(
                    "eGeMAPS çıkarılamadı ({}); MFCC ile devam ediliyor. "
                    "eGeMAPS için 'mediainfo' kurulmalı.",
                    str(exc).splitlines()[0] if str(exc) else exc,
                )
                _egemaps_warned = True
    return AcousticFeatures(mfcc=mfcc, egemaps=egemaps)


_egemaps_warned = False


__all__ = [
    "AcousticFeatures",
    "extract_acoustic_features",
    "extract_egemaps",
    "extract_mfcc",
    "N_MFCC",
]
