"""Eğitim döngüsü giriş noktası.

Şu an minimal ve bağımlılıktan arınmış bir iskelettir: gerçek veri
modülü (``data.datamodule``) ve kodlayıcı çıktıları hazır olduğunda
buradaki ``train_one_epoch`` gerçek DataLoader ile beslenecektir.

CLI: ``psikiyatri-train`` (bkz. pyproject [project.scripts]).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from ..config import get_settings
from ..logging_conf import logger, setup_logging
from .model import PsikiyatriModel

Batch = tuple[dict[str, torch.Tensor], torch.Tensor]


def train_one_epoch(
    model: PsikiyatriModel,
    batches: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Tek bir epoch eğitir, ortalama kaybı döndürür."""
    model.train()
    total, n = 0.0, 0
    for embeddings, labels in batches:
        optimizer.zero_grad()
        out = model(embeddings)
        loss = criterion(out.logits, labels)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def main() -> None:
    """CLI giriş noktası — iskelet.

    Gerçek eğitim verisi bağlanana kadar yapılandırmayı doğrular ve
    modelin oluşturulabildiğini gösterir.
    """
    setup_logging()
    settings = get_settings()
    logger.info("Ortam: {}", settings.env)
    logger.info("Model dizini: {}", settings.model_dir)

    model = PsikiyatriModel(settings.model)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model oluşturuldu. Parametre sayısı: {:,}", n_params)
    logger.warning(
        "Veri modülü henüz bağlı değil; eğitim döngüsü DataLoader "
        "hazır olduğunda etkinleştirilecek (bkz. data/datamodule.py)."
    )


if __name__ == "__main__":
    main()
