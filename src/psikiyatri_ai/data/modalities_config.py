"""Veri hattı yapılandırması (öznitelik çıkarımı parametreleri).

Model hiperparametreleri için ``config.ModelSettings``; bu dosya ise
ham veri -> öznitelik dönüşümünün parametrelerini tutar. Varsayılanlar
``config/config.yaml`` ile tutarlıdır.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..modalities.text.model import DEFAULT_MODEL_NAME


@dataclass
class DataConfig:
    # Görüntü
    frame_sample_fps: float = 2.0

    # Ses
    audio_sample_rate: int = 16_000
    use_egemaps: bool = True

    # Metin
    text_model_name: str = DEFAULT_MODEL_NAME
    text_max_length: int = 256

    # Giyilebilir sensör
    wearable_sample_rate: float = 1.0  # örnekleme frekansı (Hz)


__all__ = ["DataConfig"]
