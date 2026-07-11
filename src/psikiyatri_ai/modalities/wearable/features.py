"""Giyilebilir sensör öznitelik çıkarımı.

Fizyolojik ve davranışsal sinyalleri sabit kanallı bir zaman serisine
``(T, C)`` dönüştürür. Kanallar (bkz. ``WEARABLE_CHANNELS``):

- kalp atış hızı, HRV (kalp atış değişkenliği)
- cilt iletkenliği (EDA)
- hareket (ivmeölçer büyüklüğü) ve titreme (3–6 Hz güç oranı)
- EEG bant güçleri (delta/theta/alpha/beta)

Ham EEG/ivme sinyallerinden bant gücü ve titreme indeksi hesabı için
FFT tabanlı yardımcılar sağlanır (yalnızca NumPy; ağır bağımlılık yok).
Girdi biçimleri: ``.npz`` (``wearable`` anahtarı, ``(T, C)``) veya
başlıklı ``.csv`` (kanal adları sütun başlıkları).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Sabit kanal sırası — encoder girdi boyutu buna bağlıdır.
WEARABLE_CHANNELS: tuple[str, ...] = (
    "kalp_atis_hizi",
    "hrv",
    "cilt_iletkenligi",
    "hareket",
    "titreme",
    "eeg_delta",
    "eeg_theta",
    "eeg_alpha",
    "eeg_beta",
)
NUM_WEARABLE_CHANNELS = len(WEARABLE_CHANNELS)

# EEG frekans bantları (Hz)
_EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
# Fizyolojik titreme bandı (ör. Parkinson dinlenme titremesi 3–6 Hz)
_TREMOR_BAND = (3.0, 6.0)


@dataclass
class WearableFeatures:
    """Bir seansa ait giyilebilir sensör öznitelikleri."""

    series: np.ndarray  # şekil: (T, NUM_WEARABLE_CHANNELS)
    summary: dict[str, float] = field(default_factory=dict)  # kanal ortalamaları
    sample_rate: float = 1.0


def _band_power(signal: np.ndarray, fs: float, low: float, high: float) -> float:
    """Bir frekans bandındaki güç (tek taraflı periodogram ile)."""
    signal = np.asarray(signal, dtype=np.float64)
    n = signal.shape[0]
    if n < 2:
        return 0.0
    signal = signal - signal.mean()
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(signal)) ** 2
    mask = (freqs >= low) & (freqs < high)
    return float(power[mask].sum())


def eeg_band_powers(signal: np.ndarray, fs: float) -> dict[str, float]:
    """Ham EEG sinyalinden göreli bant güçlerini (delta/theta/alpha/beta) döner."""
    powers = {name: _band_power(signal, fs, lo, hi) for name, (lo, hi) in _EEG_BANDS.items()}
    total = sum(powers.values())
    if total <= 0:
        return dict.fromkeys(_EEG_BANDS, 0.0)
    return {name: p / total for name, p in powers.items()}


def tremor_index(accel: np.ndarray, fs: float) -> float:
    """İvmeölçer sinyalinde titreme bandı (3–6 Hz) gücünün toplam güce oranı.

    Yüksek değer, motor titremeyi (ör. Parkinson) düşündürür.
    """
    accel = np.asarray(accel, dtype=np.float64)
    if accel.shape[0] < 2:
        return 0.0
    tremor = _band_power(accel, fs, *_TREMOR_BAND)
    total = _band_power(accel, fs, 0.0, fs / 2.0)
    return float(tremor / total) if total > 0 else 0.0


def _summarize(series: np.ndarray) -> dict[str, float]:
    means = series.mean(axis=0) if series.shape[0] else np.zeros(NUM_WEARABLE_CHANNELS)
    return {ch: float(means[i]) for i, ch in enumerate(WEARABLE_CHANNELS)}


def _read_csv(path: Path) -> np.ndarray:
    """Başlıklı CSV'yi kanal sırasına göre ``(T, C)`` diziye çevirir.

    Eksik kanal sütunları sıfırla doldurulur; bilinmeyen sütunlar yok
    sayılır.
    """
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        return np.zeros((0, NUM_WEARABLE_CHANNELS), dtype=np.float32)
    out = np.zeros((len(rows), NUM_WEARABLE_CHANNELS), dtype=np.float32)
    for i, row in enumerate(rows):
        for c, ch in enumerate(WEARABLE_CHANNELS):
            val = row.get(ch)
            if val not in (None, ""):
                try:
                    out[i, c] = float(str(val))
                except ValueError:
                    out[i, c] = 0.0
    return out


def extract_wearable_features(path: Path, *, sample_rate: float = 1.0) -> WearableFeatures:
    """Bir giyilebilir kayıt dosyasından (.npz/.csv) öznitelik dizisi çıkarır."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Giyilebilir veri dosyası bulunamadı: {path}")

    if path.suffix == ".npz":
        data = np.load(path)
        key = "wearable" if "wearable" in data.files else data.files[0]
        series = np.asarray(data[key], dtype=np.float32)
    elif path.suffix == ".csv":
        series = _read_csv(path)
    else:
        raise ValueError(f"Desteklenmeyen giyilebilir veri biçimi: {path.suffix}")

    if series.ndim != 2 or series.shape[1] != NUM_WEARABLE_CHANNELS or series.shape[0] == 0:
        raise ValueError(
            f"Giyilebilir dizi şekli (T>0, {NUM_WEARABLE_CHANNELS}) olmalı, "
            f"bulunan: {series.shape}"
        )
    # NaN/Inf (bozuk hücre veya 'nan'/'inf' metni) downstream kodlayıcıyı bozar.
    series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return WearableFeatures(
        series=series, summary=_summarize(series), sample_rate=sample_rate
    )


__all__ = [
    "NUM_WEARABLE_CHANNELS",
    "WEARABLE_CHANNELS",
    "WearableFeatures",
    "eeg_band_powers",
    "extract_wearable_features",
    "tremor_index",
]
