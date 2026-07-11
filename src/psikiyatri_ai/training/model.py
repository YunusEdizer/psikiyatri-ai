"""Uçtan uca model: modalite kodlayıcıları + füzyon.

``PsikiyatriModel`` etkin modalite kodlayıcılarını ve füzyon başını tek
çatıda toplar. İki kullanım yolu vardır:

- ``forward(batch)`` : ham öznitelikler -> kodlayıcı -> füzyon (eğitim;
  gradyan uçtan uca akar).
- ``fuse(embeddings)`` : önceden hesaplanmış gömmeler -> füzyon (servis;
  kodlayıcılar olmadan da çalışır — bkz. ``api/main.py``).

``enabled_modalities`` ile hangi kodlayıcıların oluşturulacağı seçilir.
Metin kodlayıcısı ``transformers`` gerektirdiğinden, o bağımlılık yoksa
sadece ``("vision", "audio")`` ile çalışılabilir.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from ..config import ModelSettings
from ..data.batch import ModalityBatch
from ..fusion.multimodal import FusionOutput, MultimodalFusion
from ..modalities.audio.model import AudioEncoder
from ..modalities.vision.model import VisionEncoder
from ..modalities.wearable.model import WearableEncoder

ALL_MODALITIES = ("vision", "audio", "text", "wearable")


class PsikiyatriModel(nn.Module):
    """Multimodal karar destek modeli (kodlayıcılar + füzyon)."""

    def __init__(
        self,
        cfg: ModelSettings | None = None,
        enabled_modalities: Sequence[str] = ALL_MODALITIES,
        *,
        use_egemaps: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg or ModelSettings()
        self.enabled = tuple(enabled_modalities)
        self.encoders = nn.ModuleDict()

        if "vision" in self.enabled:
            self.encoders["vision"] = VisionEncoder(
                embed_dim=self.cfg.vision_embed_dim, dropout=self.cfg.dropout
            )
        if "audio" in self.enabled:
            self.encoders["audio"] = AudioEncoder(
                embed_dim=self.cfg.audio_embed_dim,
                use_egemaps=use_egemaps,
                dropout=self.cfg.dropout,
            )
        if "wearable" in self.enabled:
            self.encoders["wearable"] = WearableEncoder(
                embed_dim=self.cfg.wearable_embed_dim, dropout=self.cfg.dropout
            )
        if "text" in self.enabled:
            # Ağır bağımlılık (transformers) yalnızca burada yüklenir.
            from ..modalities.text.model import TextEncoder

            self.encoders["text"] = TextEncoder(embed_dim=self.cfg.text_embed_dim)

        self.fusion = MultimodalFusion(self.cfg)

    def encode(self, batch: ModalityBatch) -> dict[str, torch.Tensor]:
        """Mevcut ve etkin modaliteler için gömme üretir."""
        emb: dict[str, torch.Tensor] = {}
        if "vision" in self.encoders and batch.vision is not None:
            emb["vision"] = self.encoders["vision"](batch.vision)
        if "audio" in self.encoders and batch.audio_mfcc is not None:
            emb["audio"] = self.encoders["audio"](batch.audio_mfcc, batch.audio_egemaps)
        if "wearable" in self.encoders and batch.wearable is not None:
            emb["wearable"] = self.encoders["wearable"](batch.wearable)
        if "text" in self.encoders and batch.text_input_ids is not None:
            emb["text"] = self.encoders["text"](
                batch.text_input_ids, batch.text_attention_mask
            )
        if not emb:
            raise ValueError("Batch hiçbir etkin modalite içermiyor.")
        return emb

    def forward(self, batch: ModalityBatch) -> FusionOutput:
        return self.fusion(self.encode(batch))

    def fuse(self, embeddings: dict[str, torch.Tensor]) -> FusionOutput:
        """Önceden hesaplanmış gömmelerden doğrudan füzyon (servis yolu)."""
        return self.fusion(embeddings)

    @torch.no_grad()
    def predict_proba(self, batch: ModalityBatch) -> torch.Tensor:
        """Sınıf olasılıkları (B, num_classes)."""
        self.eval()
        return torch.softmax(self.forward(batch).logits, dim=-1)


__all__ = ["ALL_MODALITIES", "PsikiyatriModel"]
