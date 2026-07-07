"""Görüntü modalitesi kodlayıcısı (encoder).

Kare dizisi işaretçilerini zamansal olarak toplayıp sabit boyutlu bir
gömme vektörü üretir. İskelet: işaretçi dizisi -> zamansal ortalama +
MLP. İleride buraya CNN/Transformer tabanlı bir omurga (ör. mikro-ifade
için 3D-CNN) takılabilir.
"""

from __future__ import annotations

import torch
from torch import nn

from .preprocess import NUM_FACE_LANDMARKS


class VisionEncoder(nn.Module):
    """Yüz işaretçisi dizisini gömmeye çeviren kodlayıcı."""

    def __init__(self, embed_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        in_features = NUM_FACE_LANDMARKS * 3  # (x, y, z)
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
        )

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            landmarks: (B, T, NUM_FACE_LANDMARKS, 3) — B örnek, T kare.

        Returns:
            (B, embed_dim) gömme.
        """
        b, t, n, c = landmarks.shape
        # Zaman ekseninde ortalama al (basit zamansal toplama)
        pooled = landmarks.mean(dim=1).reshape(b, n * c)
        return self.net(pooled)


__all__ = ["VisionEncoder"]
