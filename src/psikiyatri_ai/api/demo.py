"""Demo çıkarım motoru — görsel arayüz için.

Ham medya/kodlayıcı bağımlılıkları olmadan, eğitilmiş tam modeli
(görüntü + ses kodlayıcıları + füzyon) checkpoint'ten yükler ve sentetik
bir seans örneği üretip karar destek çıktısı döndürür.

> Bu tamamen bir DEMO'dur: üretilen veri yapaydır, hiçbir gerçek hastayı
> temsil etmez ve çıktı teşhis değildir.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from ..config import get_settings
from ..data.batch import ModalityBatch
from ..logging_conf import logger

DEMO_MODALITIES = ("vision", "audio")


@lru_cache(maxsize=1)
def _load_model():
    """Tam modeli (görüntü+ses) checkpoint'ten yükler (bir kez, önbellekli)."""
    from ..training.model import PsikiyatriModel

    settings = get_settings()
    model = PsikiyatriModel(
        settings.model, enabled_modalities=DEMO_MODALITIES, use_egemaps=True
    )
    ckpt = settings.model_dir / "psikiyatri_model.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        logger.info("Demo modeli checkpoint'ten yüklendi: {}", ckpt)
    else:
        logger.warning(
            "Demo için checkpoint yok ({}); rastgele model kullanılıyor. "
            "Önce 'psikiyatri-train' çalıştırın.",
            ckpt,
        )
    model.eval()
    return model


def _make_batch(scenario: str, signal: float, seed: int) -> ModalityBatch:
    """Senaryoya göre sentetik bir seans örneği (B=1) üretir.

    Sinyal, eğitimdeki ile aynı biçimde ses özniteliklerine enjekte edilir
    (bkz. data/synthetic_data.py): 'depresyon' senaryosunda kayma uygulanır,
    'kontrol'de uygulanmaz.
    """
    rng = np.random.default_rng(seed)
    if scenario == "depresyon":
        shift = signal
    elif scenario == "kontrol":
        shift = 0.0
    else:  # rastgele
        shift = float(rng.choice([0.0, signal]))

    vision = rng.standard_normal((1, 8, 468, 3)).astype("float32")
    mfcc = (rng.standard_normal((1, 50, 40)) + shift * 0.3).astype("float32")
    egemaps = (rng.standard_normal((1, 88)) + shift).astype("float32")

    return ModalityBatch(
        vision=torch.from_numpy(vision),
        audio_mfcc=torch.from_numpy(mfcc),
        audio_egemaps=torch.from_numpy(egemaps),
    )


def run_demo(
    scenario: str,
    signal: float,
    seed: int,
    modalities: list[str],
) -> dict:
    """Demo örneğini modelden geçirip yapılandırılmış sonuç döndürür."""
    if not modalities:
        raise ValueError("En az bir modalite seçilmelidir.")

    model = _load_model()
    batch = _make_batch(scenario, signal, seed)

    # Kapatılan modaliteleri maskele
    if "vision" not in modalities:
        batch.vision = None
    if "audio" not in modalities:
        batch.audio_mfcc = None
        batch.audio_egemaps = None

    with torch.no_grad():
        out = model(batch)
    probs = torch.softmax(out.logits, dim=-1).squeeze(0)

    labels = ["kontrol", "depresyon"]
    present = batch.present_modalities
    return {
        "probabilities": {labels[i]: float(probs[i]) for i in range(len(labels))},
        "predicted_label": labels[int(probs.argmax())],
        "incongruence": float(out.incongruence[0]),
        "modality_contributions": [
            {"modality": m, "weight": float(out.attention[0, i])}
            for i, m in enumerate(present)
        ],
    }


__all__ = ["DEMO_MODALITIES", "run_demo"]
