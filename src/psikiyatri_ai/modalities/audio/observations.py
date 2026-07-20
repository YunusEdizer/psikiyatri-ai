"""Betimsel (descriptive) akustik gözlemler.

Kaydedilen sesten **doğrudan yorumlanabilir** akustik ölçütler çıkarır:
temel frekans (perde), perde değişkenliği, ses şiddeti (enerji) ve
konuşma etkinliği (seslilik oranı). Bu ölçütler klinisyene sunulan
gözlemsel sinyallerdir; eğitilmiş bir sınıflandırıcı GEREKTİRMEZ ve tanı
üretmez.

Ayrım önemlidir: proje modelinin ses *kodlayıcısı* henüz etiketli ses
verisiyle eğitilmediğinden ses, tanı füzyonuna katılmaz (katılırsa
gürültü ekler). Buna karşılık aşağıdaki betimsel ölçütler ham sinyalden
deterministik biçimde hesaplanır ve gözlemsel olarak güvenilirdir
(ör. "ses tonu: frekans, hız, tonlama" — proje tanımı).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .features import DEFAULT_SAMPLE_RATE, ensure_ffmpeg_on_path


@dataclass
class AcousticObservation:
    """Betimsel akustik gözlem (tanı değil)."""

    duration_s: float
    pitch_hz: float | None  # ortalama F0 (seslendirilmiş çerçeveler)
    pitch_variability_hz: float | None  # F0 std — monoton(düşük)/dalgalı(yüksek)
    loudness_rms: float  # ortalama RMS enerji (0–1 ölçeğine yakın)
    voiced_ratio: float  # seslendirilmiş çerçeve oranı (konuşma etkinliği)
    qualitative: dict[str, str]  # Türkçe nitel etiketler (perde/enerji/akıcılık)


def describe_audio(
    audio_path: Path, *, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> AcousticObservation | None:
    """Ses kaydından betimsel akustik gözlemleri çıkarır.

    Çözümleme/okuma başarısız olursa ``None`` döner (rapor akışını kesmez).
    """
    try:
        import librosa  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        return None

    ensure_ffmpeg_on_path()
    try:
        y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    except Exception:  # noqa: BLE001 - okunamayan/bozuk medya
        return None
    if y.size == 0:
        return None

    duration = float(len(y) / sr)
    rms = float(np.mean(librosa.feature.rms(y=y)))

    pitch_hz: float | None = None
    pitch_var: float | None = None
    voiced_ratio = 0.0
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=65.0, fmax=400.0, sr=sr, frame_length=2048
        )
        voiced = f0[np.isfinite(f0)]
        if voiced_flag is not None and voiced_flag.size:
            voiced_ratio = float(np.mean(voiced_flag.astype(float)))
        if voiced.size:
            pitch_hz = float(np.mean(voiced))
            pitch_var = float(np.std(voiced))
    except Exception:  # noqa: BLE001 - pyin bazı sinyallerde düşebilir
        pass

    return AcousticObservation(
        duration_s=round(duration, 1),
        pitch_hz=round(pitch_hz, 1) if pitch_hz is not None else None,
        pitch_variability_hz=round(pitch_var, 1) if pitch_var is not None else None,
        loudness_rms=round(rms, 4),
        voiced_ratio=round(voiced_ratio, 3),
        qualitative=_qualitative(pitch_hz, pitch_var, rms, voiced_ratio),
    )


def _qualitative(
    pitch: float | None, pitch_var: float | None, rms: float, voiced: float
) -> dict[str, str]:
    """Ham ölçütleri nötr, betimsel Türkçe etiketlere çevirir (tanı değil).

    Eşikler kaba betimsel gruplamadır; klinik ölçüt değildir.
    """
    q: dict[str, str] = {}
    if pitch is not None:
        q["perde"] = "düşük" if pitch < 120 else "yüksek" if pitch > 220 else "orta"
    if pitch_var is not None:
        # Düşük değişkenlik = monoton tonlama (düz duygulanım göstergesi olabilir).
        q["tonlama"] = "monoton" if pitch_var < 20 else "dalgalı" if pitch_var > 55 else "orta"
    q["ses_siddeti"] = "kısık" if rms < 0.02 else "yüksek" if rms > 0.15 else "orta"
    q["konusma_etkinligi"] = (
        "az" if voiced < 0.25 else "yoğun" if voiced > 0.6 else "orta"
    )
    return q


def observation_to_dict(obs: AcousticObservation | None) -> dict | None:
    return asdict(obs) if obs is not None else None


__all__ = ["AcousticObservation", "describe_audio", "observation_to_dict"]
