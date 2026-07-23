"""Betimsel (descriptive) giyilebilir sensör gözlemleri.

Kaydedilen fizyolojik seriden **doğrudan yorumlanabilir** ölçütler çıkarır:
nabız, kalp hızı değişkenliği (HRV), cilt iletkenliği (otonomik uyarılma),
hareket/titreme ve baskın EEG bandı. Ses/yüz gözlemleriyle aynı ilke:
tanı füzyonuna KATILMAZ (giyilebilir kodlayıcı henüz eğitilmedi), klinisyene
gözlemsel destek sunar (proje tanımı: "kalp atış hızı, cilt iletkenliği,
uyku düzeni ile ... anksiyete/depresyon takibi").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .features import WEARABLE_CHANNELS


@dataclass
class WearableObservation:
    """Betimsel giyilebilir gözlem (tanı değil)."""

    samples: int
    heart_rate: float | None
    hrv: float | None
    skin_conductance: float | None
    movement: float | None
    tremor: float | None
    dominant_eeg_band: str | None
    qualitative: dict[str, str]


def describe_wearable(path: Path, *, sample_rate: float = 1.0) -> WearableObservation | None:
    """Giyilebilir kaydından betimsel gözlemleri çıkarır.

    Okuma/çözme başarısızsa ya da seri boşsa ``None`` döner (akışı kesmez).
    """
    try:
        from .features import extract_wearable_features  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        return None
    try:
        series = extract_wearable_features(path, sample_rate=sample_rate).series
    except Exception:  # noqa: BLE001 - okunamayan/bozuk kayıt
        return None
    if series.ndim != 2 or series.shape[0] == 0:
        return None

    idx = {ch: i for i, ch in enumerate(WEARABLE_CHANNELS)}
    means = series.mean(axis=0)

    def m(ch: str) -> float | None:
        return round(float(means[idx[ch]]), 3) if ch in idx else None

    hr = m("kalp_atis_hizi")
    hrv = m("hrv")
    scl = m("cilt_iletkenligi")
    mov = m("hareket")
    trem = m("titreme")

    # Baskın EEG bandı (delta/theta/alpha/beta ortalamalarının en büyüğü).
    bands = {b: float(means[idx[f"eeg_{b}"]]) for b in ("delta", "theta", "alpha", "beta")
             if f"eeg_{b}" in idx}
    dominant = max(bands, key=lambda k: bands[k]) if bands else None

    return WearableObservation(
        samples=int(series.shape[0]),
        heart_rate=hr, hrv=hrv, skin_conductance=scl, movement=mov, tremor=trem,
        dominant_eeg_band=dominant,
        qualitative=_qualitative(hr, scl, mov, trem, dominant),
    )


def _qualitative(
    hr: float | None, scl: float | None, mov: float | None,
    trem: float | None, band: str | None,
) -> dict[str, str]:
    """Ham ölçütleri nötr, betimsel Türkçe etiketlere çevirir (tanı değil)."""
    q: dict[str, str] = {}
    if hr is not None:
        q["nabiz"] = "düşük" if hr < 60 else "yüksek" if hr > 100 else "normal"
    if scl is not None:  # cilt iletkenliği ≈ otonomik uyarılma
        q["uyarilma"] = "düşük" if scl < 0.2 else "yüksek" if scl > 0.6 else "orta"
    if mov is not None:
        q["hareketlilik"] = "az" if mov < 0.2 else "yüksek" if mov > 0.6 else "orta"
    if trem is not None:
        q["titreme"] = "belirgin" if trem > 0.3 else "hafif"
    if band:
        # Kaba yorum: beta uyanıklık/uyarılma, alpha gevşeme (bilgilendirici).
        q["eeg_baskin"] = band
    return q


def observation_to_dict(obs: WearableObservation | None) -> dict | None:
    return asdict(obs) if obs is not None else None


__all__ = ["WearableObservation", "describe_wearable", "observation_to_dict"]
