"""Multimodal füzyon modeli.

Her modalite kodlayıcısından gelen gömmeleri ortak bir uzaya izdüşürür,
dikkat (attention) tabanlı olarak birleştirir, uyumsuzluk skorunu ek
öznitelik olarak ekler ve sınıflandırma başına verir.

Tasarım ilkeleri:
- **Eksik modaliteye dayanıklılık**: bir seansta yalnızca ses+metin
  olabilir; model mevcut modalitelerle çalışabilmelidir.
- **Açıklanabilirlik**: dikkat ağırlıkları ve uyumsuzluk skoru çıktıya
  dahil edilerek uzmana gerekçe sunulur.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..config import ModelSettings
from .incongruence import pairwise_incongruence

MODALITIES = ("vision", "audio", "text")


@dataclass
class FusionOutput:
    """Füzyon modeli çıktısı."""

    logits: torch.Tensor  # (B, num_classes)
    incongruence: torch.Tensor  # (B,)
    attention: torch.Tensor  # (B, n_modalite) — modalite katkı ağırlıkları


class MultimodalFusion(nn.Module):
    """Gömme sözlüğü -> tanı logit'leri + açıklanabilirlik sinyalleri."""

    def __init__(self, cfg: ModelSettings | None = None) -> None:
        super().__init__()
        cfg = cfg or ModelSettings()
        d = cfg.fusion_dim

        # Her modaliteyi ortak uzaya izdüşür
        self.proj = nn.ModuleDict(
            {
                "vision": nn.Linear(cfg.vision_embed_dim, d),
                "audio": nn.Linear(cfg.audio_embed_dim, d),
                "text": nn.Linear(cfg.text_embed_dim, d),
            }
        )
        # Modalite başına skaler dikkat skoru
        self.attn = nn.Linear(d, 1)
        # +1: uyumsuzluk skoru ek öznitelik
        self.classifier = nn.Sequential(
            nn.Linear(d + 1, d),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, cfg.num_classes),
        )

    def forward(self, embeddings: dict[str, torch.Tensor]) -> FusionOutput:
        """
        Args:
            embeddings: modalite adı -> (B, embed_dim). Alt küme olabilir
                (eksik modalite desteklenir). En az bir modalite gerekli.

        Returns:
            FusionOutput.
        """
        if not embeddings:
            raise ValueError("En az bir modalite gömmesi gerekli.")

        projected = {m: torch.relu(self.proj[m](e)) for m, e in embeddings.items()}
        keys = [m for m in MODALITIES if m in projected]  # sabit sıra
        stacked = torch.stack([projected[m] for m in keys], dim=1)  # (B, M, d)

        # Dikkat ağırlıkları ile modaliteleri topla
        scores = self.attn(stacked).squeeze(-1)  # (B, M)
        weights = torch.softmax(scores, dim=1)  # (B, M)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # (B, d)

        incong = pairwise_incongruence(projected)  # (B,)
        feat = torch.cat([fused, incong.unsqueeze(-1)], dim=-1)
        logits = self.classifier(feat)

        return FusionOutput(logits=logits, incongruence=incong, attention=weights)


__all__ = ["FusionOutput", "MultimodalFusion", "MODALITIES"]
