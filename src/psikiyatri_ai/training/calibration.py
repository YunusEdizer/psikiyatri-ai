"""Model kalibrasyonu — sıcaklık ölçekleme (temperature scaling).

Sinir ağları çoğunlukla **aşırı-güvenlidir**: %90 dediğinde gerçek doğruluk
%70 olabilir. Klinik karar desteğinde olasılığın anlamlı olması önemlidir;
bu nedenle doğrulama kümesinde tek bir skaler sıcaklık ``T`` öğrenilir ve
logit'ler ``T``'ye bölünerek olasılıklar yeniden ölçeklenir (Guo ve diğ.,
2017). ``T > 1`` güveni düşürür, ``T < 1`` artırır. Sınıf sıralamasını (ve
dolayısıyla doğruluğu) DEĞİŞTİRMEZ; yalnızca olasılıkları kalibre eder.
"""

from __future__ import annotations

import torch


def fit_temperature(
    logits: torch.Tensor, labels: torch.Tensor, *, max_iter: int = 100
) -> float:
    """Doğrulama logit'lerinde NLL'yi en aza indiren sıcaklığı öğrenir.

    Args:
        logits: (N, C) ölçeklenmemiş model çıktıları.
        labels: (N,) doğru sınıf indeksleri.

    Returns:
        Öğrenilen sıcaklık ``T`` (>= 0.05'e kırpılır). Yetersiz veride 1.0.
    """
    logits = logits.detach().float()
    labels = labels.detach().long()
    if logits.ndim != 2 or logits.shape[0] < 2:
        return 1.0

    temperature = torch.ones(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def _closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = nll(logits / temperature.clamp(min=0.05), labels)
        loss.backward()
        return loss

    optimizer.step(_closure)  # type: ignore[arg-type]
    return float(temperature.detach().clamp(min=0.05).item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Logit'leri sıcaklıkla ölçekler (kalibre olasılıklar için softmax öncesi)."""
    t = max(float(temperature), 0.05)
    return logits / t


__all__ = ["apply_temperature", "fit_temperature"]
