"""Eğitim/çıkarım sırasında dolaşan tensör yapıları.

``SampleFeatures`` tek bir örneğin çıkarılmış (ham değil) özniteliklerini,
``ModalityBatch`` ise bunların toplu (batch) ve doldurulmuş (padded)
halini tutar. Bu yapılar torch'a bağlıdır; pydantic tabanlı veri
sözleşmeleri için bkz. ``data/schema.py``.

Ayrı bir modülde tutulur ki hem ``data/datamodule.py`` hem de
``training/model.py`` döngüsel import olmadan kullanabilsin.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SampleFeatures:
    """Tek bir seansa ait çıkarılmış öznitelikler (+ etiket).

    Kullanılmayan modaliteler ``None`` bırakılır. Eğitimde bir batch
    içindeki tüm örneklerin aynı modalite kümesine sahip olması beklenir
    (kontrollü veri toplama). Kısmi modalite yalnızca çıkarım tarafında
    (füzyonun sözlük arayüzü) desteklenir.
    """

    label: int
    vision: torch.Tensor | None = None          # (T, 468, 3)
    audio_mfcc: torch.Tensor | None = None       # (T, 40)
    audio_egemaps: torch.Tensor | None = None    # (88,)
    text_input_ids: torch.Tensor | None = None   # (L,)
    text_attention_mask: torch.Tensor | None = None  # (L,)


@dataclass
class ModalityBatch:
    """Doldurulmuş, toplu modalite tensörleri."""

    vision: torch.Tensor | None = None           # (B, T, 468, 3)
    audio_mfcc: torch.Tensor | None = None        # (B, T, 40)
    audio_egemaps: torch.Tensor | None = None     # (B, 88)
    text_input_ids: torch.Tensor | None = None    # (B, L)
    text_attention_mask: torch.Tensor | None = None  # (B, L)

    def to(self, device: torch.device | str) -> ModalityBatch:
        """Tüm mevcut tensörleri hedef cihaza taşır."""
        return ModalityBatch(
            **{
                k: (v.to(device) if v is not None else None)
                for k, v in self.__dict__.items()
            }
        )

    @property
    def present_modalities(self) -> list[str]:
        mods: list[str] = []
        if self.vision is not None:
            mods.append("vision")
        if self.audio_mfcc is not None:
            mods.append("audio")
        if self.text_input_ids is not None:
            mods.append("text")
        return mods


__all__ = ["ModalityBatch", "SampleFeatures"]
