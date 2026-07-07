"""Uçtan uca model: modalite kodlayıcıları + füzyon.

Bu sınıf, üç modalite kodlayıcısını ve füzyon başını tek bir çatı altında
toplar. Kodlayıcılar tembel oluşturulur (opsiyonel bağımlılıklar), böylece
yalnızca kullanılan modaliteler için ağır kütüphaneler gerekir.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelSettings
from ..fusion.multimodal import FusionOutput, MultimodalFusion


class PsikiyatriModel(nn.Module):
    """Multimodal karar destek modeli.

    ``forward`` doğrudan önceden hesaplanmış modalite gömmelerini kabul
    eder; bu, kodlayıcı çıktısının önbelleğe alınmasını ve modalitelerin
    ayrı ayrı ölçeklenmesini kolaylaştırır.
    """

    def __init__(self, cfg: ModelSettings | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelSettings()
        self.fusion = MultimodalFusion(self.cfg)

    def forward(self, embeddings: dict[str, torch.Tensor]) -> FusionOutput:
        return self.fusion(embeddings)

    @torch.no_grad()
    def predict_proba(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sınıf olasılıklarını döndürür (B, num_classes)."""
        self.eval()
        out = self.fusion(embeddings)
        return torch.softmax(out.logits, dim=-1)


__all__ = ["PsikiyatriModel"]
