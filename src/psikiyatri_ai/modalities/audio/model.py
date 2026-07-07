"""Ses modalitesi kodlayıcısı.

MFCC dizisini (zamansal) bir GRU ile kodlar; gerekirse eGeMAPS
fonksiyonellerini birleştirerek gömme üretir.
"""

from __future__ import annotations

import torch
from torch import nn

from .features import N_MFCC


class AudioEncoder(nn.Module):
    """MFCC (+ opsiyonel eGeMAPS) -> gömme."""

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_size: int = 128,
        egemaps_dim: int = 88,
        use_egemaps: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.use_egemaps = use_egemaps
        self.gru = nn.GRU(
            input_size=N_MFCC,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        fusion_in = hidden_size * 2 + (egemaps_dim if use_egemaps else 0)
        self.head = nn.Sequential(
            nn.Linear(fusion_in, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, mfcc: torch.Tensor, egemaps: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            mfcc: (B, T, N_MFCC).
            egemaps: (B, egemaps_dim) veya None.

        Returns:
            (B, embed_dim) gömme.
        """
        _, h_n = self.gru(mfcc)
        # İki yönün son gizli durumlarını birleştir
        pooled = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        if self.use_egemaps and egemaps is not None:
            pooled = torch.cat([pooled, egemaps], dim=-1)
        return self.head(pooled)


__all__ = ["AudioEncoder"]
