"""Sentetik örnek veri üretimi (geliştirme/demo).

Gerçek klinik veri hazır olmadan tüm eğitim hattını uçtan uca
çalıştırabilmek için önceden çıkarılmış öznitelikler (.npz) + manifest
üretir. Sınıflara **öğrenilebilir bir sinyal** enjekte edilir; böylece
model eğitildiğinde denge-doğruluğu şansın belirgin üzerine çıkar ve
hattın gerçekten öğrendiği kanıtlanır.

İkili varsayılan (kontrol/depresyon) dışında, ``classes`` ile çok-
hastalık (anksiyete/bipolar/şizofreni) ve ``include_wearable`` ile
giyilebilir modalite de üretilebilir. Her sınıfa ayırt edici bir imza
verilir.

Not: Üretilen veri tamamen yapaydır, hiçbir gerçek hastayı temsil etmez.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..modalities.wearable.features import NUM_WEARABLE_CHANNELS
from .schema import DiagnosisLabel, Sample

# İkili demo: kontrol (0) vs. depresyon (1)
CLASSES: tuple[DiagnosisLabel, ...] = (
    DiagnosisLabel.KONTROL,
    DiagnosisLabel.DEPRESYON,
)

# Çok-hastalık demo için genişletilmiş sınıf kümesi (kontrol = 0)
MULTICLASS: tuple[DiagnosisLabel, ...] = (
    DiagnosisLabel.KONTROL,
    DiagnosisLabel.DEPRESYON,
    DiagnosisLabel.ANKSIYETE,
    DiagnosisLabel.BIPOLAR,
    DiagnosisLabel.SIZOFRENI,
)


def generate_sample_dataset(
    count: int = 200,
    *,
    seed: int = 42,
    out_dir: Path | None = None,
    data_root: Path | None = None,
    signal: float = 0.6,
    t_vision: int = 8,
    t_audio: int = 50,
    classes: Sequence[DiagnosisLabel] = CLASSES,
    include_wearable: bool = False,
    t_wearable: int = 60,
) -> Path:
    """Sentetik öznitelik dosyaları (.npz) + manifest üretir.

    Args:
        count: Üretilecek örnek sayısı (sınıflar dengeli döngüyle atanır).
        seed: Tekrarüretilebilirlik için tohum.
        out_dir: Çıktı dizini (varsayılan: ``<data>/processed``).
        data_root: Manifest'teki göreli yolların tabanı (varsayılan: ``<data>``).
        signal: Sınıflar arası ortalama kayması (0 = ayırt edilemez, saf gürültü).
        classes: Üretilecek tanı sınıfları (ilk sınıf kontrol olmalıdır).
        include_wearable: Giyilebilir sensör kanalı da üretilsin mi.

    Returns:
        Yazılan manifest dosyasının yolu.
    """
    settings = get_settings()
    out_dir = Path(out_dir) if out_dir else settings.processed_data_dir
    data_root = Path(data_root) if data_root else settings.data_dir
    n_classes = len(classes)

    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    lines: list[str] = []
    for i in range(count):
        label_idx = i % n_classes  # dengeli sınıflar
        shift = signal if label_idx != 0 else 0.0  # kontrol = kaymasız

        vision = rng.standard_normal((t_vision, 468, 3)).astype("float32")
        mfcc = (rng.standard_normal((t_audio, 40)) + shift * 0.3).astype("float32")
        egemaps = (rng.standard_normal(88) + shift).astype("float32")
        # Çok-sınıfta her tanıya ayırt edici bir segment imzası ver.
        if n_classes > 2 and label_idx != 0:
            seg = slice((label_idx - 1) * 15, (label_idx - 1) * 15 + 15)
            egemaps[seg] += signal * 1.5

        arrays = {"vision": vision, "audio_mfcc": mfcc, "audio_egemaps": egemaps}
        if include_wearable:
            wear = rng.standard_normal((t_wearable, NUM_WEARABLE_CHANNELS)).astype("float32")
            if label_idx != 0:  # tanıya özel kanal kayması
                wear[:, (label_idx - 1) % NUM_WEARABLE_CHANNELS] += signal
            arrays["wearable"] = wear

        fpath = feat_dir / f"{i:05d}.npz"
        np.savez_compressed(fpath, **arrays)  # type: ignore[arg-type]

        try:
            rel = fpath.relative_to(data_root)
        except ValueError:
            rel = fpath

        sample = Sample(
            pseudonym=f"pid_{i:016d}",
            session_id=f"S{i:05d}",
            features_path=rel,
            consent=True,
            label=classes[label_idx],
        )
        lines.append(sample.model_dump_json(exclude_none=True))

    manifest = out_dir / "manifest.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


__all__ = ["CLASSES", "MULTICLASS", "generate_sample_dataset"]
