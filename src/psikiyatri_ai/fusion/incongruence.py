"""Modaliteler arası uyumsuzluk (incongruence) ölçütü.

Projenin ayırt edici hipotezi: ses tonu, yüz ifadesi ve konuşma içeriği
arasındaki *uyumsuzluk* (ör. üzgün içerik ama nötr ses tonu) bastırılmış
duyguların ve psikiyatrik belirtilerin bir göstergesi olabilir.

Uyumsuzluğu, modalite gömmeleri ortak bir uzaya izdüşürüldükten sonra
ikili kosinüs mesafeleri üzerinden ölçeriz. Bu skor hem füzyon modeline
ek bir öznitelik olarak verilebilir hem de açıklanabilirlik için uzmana
sunulabilir.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812  (torch yaygın kısaltması)


def pairwise_incongruence(embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mevcut modaliteler arasında ortalama ikili uyumsuzluk skoru.

    Args:
        embeddings: modalite adı -> (B, D) gömme. Ortak uzayda ve aynı
            D boyutunda olmalıdır (bkz. ``MultimodalFusion`` izdüşümleri).

    Returns:
        (B,) aralığı [0, 1] olan uyumsuzluk skoru. 0 = tam uyum,
        1 = tam zıtlık. Tek modalite varsa sıfır döner.
    """
    keys = sorted(embeddings)
    if len(keys) < 2:
        any_emb = next(iter(embeddings.values()))
        return torch.zeros(any_emb.shape[0], device=any_emb.device)

    distances: list[torch.Tensor] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            cos = F.cosine_similarity(embeddings[keys[i]], embeddings[keys[j]], dim=-1)
            distances.append((1.0 - cos) / 2.0)  # [-1,1] -> [0,1]
    return torch.stack(distances, dim=0).mean(dim=0)


__all__ = ["pairwise_incongruence"]
