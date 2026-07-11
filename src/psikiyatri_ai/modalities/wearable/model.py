"""Giyilebilir modalite kodlayıcısı.

Sabit kanallı fizyolojik zaman serisini ``(B, T, C)`` çift yönlü bir GRU
ile kodlar ve sabit boyutlu bir gömme üretir. Ses kodlayıcısıyla (bkz.
``modalities/audio/model.py``) aynı tasarım desenini izler.
"""

from __future__ import annotations

import torch
from torch import nn

from .features import NUM_WEARABLE_CHANNELS


class WearableEncoder(nn.Module):
    """Giyilebilir sensör serisi -> gömme."""

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_size: int = 64,
        num_channels: int = NUM_WEARABLE_CHANNELS,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=num_channels,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        """
        Args:
            series: (B, T, C) fizyolojik zaman serisi.

        Returns:
            (B, embed_dim) gömme.
        """
        _, h_n = self.gru(series)
        pooled = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # iki yön son gizli durum
        return self.head(pooled)


__all__ = ["WearableEncoder"]
