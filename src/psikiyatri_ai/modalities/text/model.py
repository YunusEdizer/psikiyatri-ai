"""Metin modalitesi kodlayıcısı (Türkçe BERT / BERTurk).

Konuşma içeriğini önceden eğitilmiş bir Türkçe dil modeliyle kodlar ve
[CLS] temsilini gömme olarak döndürür. Transformers bağımlılığı tembel
yüklenir (`[text]` ekstrası).
"""

from __future__ import annotations

import torch
from torch import nn

DEFAULT_MODEL_NAME = "dbmdz/bert-base-turkish-cased"


class TextEncoder(nn.Module):
    """Türkçe transformer tabanlı metin kodlayıcı."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        embed_dim: int = 768,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Metin modalitesi için 'transformers' gerekli. "
                "Kurulum: pip install 'psikiyatri-ai[text]'"
            ) from exc

        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.proj = nn.Identity() if hidden == embed_dim else nn.Linear(hidden, embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) tokenize edilmiş metin.
            attention_mask: (B, L).

        Returns:
            (B, embed_dim) gömme ([CLS] temsili).
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # [CLS]
        return self.proj(cls)


__all__ = ["TextEncoder", "DEFAULT_MODEL_NAME"]
