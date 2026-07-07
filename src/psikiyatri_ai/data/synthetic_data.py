"""Sentetik örnek veri üretimi (geliştirme/demo).

Gerçek klinik veri hazır olmadan tüm eğitim hattını uçtan uca
çalıştırabilmek için önceden çıkarılmış öznitelikler (.npz) + manifest
üretir. Sınıflara **öğrenilebilir bir sinyal** enjekte edilir; böylece
model eğitildiğinde denge-doğruluğu 0.5'in belirgin üzerine çıkar ve
hattın gerçekten öğrendiği kanıtlanır.

Not: Üretilen veri tamamen yapaydır, hiçbir gerçek hastayı temsil etmez.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import get_settings
from .schema import DiagnosisLabel, Sample

# İkili demo: kontrol (0) vs. depresyon (1)
CLASSES: tuple[DiagnosisLabel, DiagnosisLabel] = (
    DiagnosisLabel.KONTROL,
    DiagnosisLabel.DEPRESYON,
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
) -> Path:
    """Sentetik öznitelik dosyaları (.npz) + manifest üretir.

    Args:
        count: Üretilecek örnek sayısı (sınıflar dengeli).
        seed: Tekrarüretilebilirlik için tohum.
        out_dir: Çıktı dizini (varsayılan: ``<data>/processed``).
        data_root: Manifest'teki göreli yolların tabanı (varsayılan: ``<data>``).
        signal: Sınıflar arası ortalama kayması (0 = ayırt edilemez, saf gürültü).

    Returns:
        Yazılan manifest dosyasının yolu.
    """
    settings = get_settings()
    out_dir = Path(out_dir) if out_dir else settings.processed_data_dir
    data_root = Path(data_root) if data_root else settings.data_dir

    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    lines: list[str] = []
    for i in range(count):
        label_idx = i % 2  # dengeli sınıflar
        shift = signal if label_idx == 1 else 0.0

        vision = rng.standard_normal((t_vision, 468, 3)).astype("float32")
        mfcc = (rng.standard_normal((t_audio, 40)) + shift * 0.3).astype("float32")
        egemaps = (rng.standard_normal(88) + shift).astype("float32")

        fpath = feat_dir / f"{i:05d}.npz"
        np.savez_compressed(fpath, vision=vision, audio_mfcc=mfcc, audio_egemaps=egemaps)

        try:
            rel = fpath.relative_to(data_root)
        except ValueError:
            rel = fpath

        sample = Sample(
            pseudonym=f"pid_{i:016d}",
            session_id=f"S{i:05d}",
            features_path=rel,
            label=CLASSES[label_idx],
        )
        lines.append(sample.model_dump_json(exclude_none=True))

    manifest = out_dir / "manifest.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


__all__ = ["CLASSES", "generate_sample_dataset"]
