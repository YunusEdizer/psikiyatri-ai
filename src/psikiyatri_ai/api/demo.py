"""Demo çıkarım motoru — görsel arayüz için.

Ham medya/kodlayıcı bağımlılıkları olmadan, eğitilmiş modeli (görüntü +
ses + isteğe bağlı giyilebilir kodlayıcıları + füzyon) checkpoint'ten
yükler; senaryoya uygun sentetik bir seans üretip karar destek çıktısı
döndürür. Sınıf sayısı ve etiketler eğitim yan-verisinden (classes.json)
okunur — ikili veya çok-sınıflı (kontrol/depresyon/anksiyete/bipolar/
şizofreni) modeli aynı kod servis eder.

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
from ..modalities.wearable.features import NUM_WEARABLE_CHANNELS
from ..training.checkpoint import labels_for_state, load_metadata

# Demo'da servis edilebilir modaliteler (metin hariç: transformers gerektirir).
DEMO_CAPABLE = ("vision", "audio", "wearable")
_DEFAULT_LABELS = ["kontrol", "depresyon"]


@lru_cache(maxsize=1)
def _load() -> tuple:
    """Modeli + sınıf/modalite yan-verisini yükler (bir kez, önbellekli)."""
    from ..training.model import PsikiyatriModel

    settings = get_settings()
    ckpt = settings.model_dir / "psikiyatri_model.pt"
    state = torch.load(ckpt, map_location="cpu") if ckpt.exists() else None
    meta = load_metadata(settings.model_dir)
    if meta:
        classes = list(meta["classes"])
        enabled = tuple(m for m in meta["modalities"] if m in DEMO_CAPABLE) or ("vision", "audio")
    elif state is not None:
        # Yan-veri yoksa sınıf sayısını checkpoint'ten türet (çökmeyi önle).
        classes = labels_for_state(state, _DEFAULT_LABELS)
        enabled = ("vision", "audio")
    else:
        classes = list(_DEFAULT_LABELS)
        enabled = ("vision", "audio")

    model_cfg = settings.model.model_copy(update={"num_classes": len(classes)})
    model = PsikiyatriModel(model_cfg, enabled_modalities=enabled, use_egemaps=True)
    if state is not None:
        model.load_state_dict(state, strict=False)
        logger.info("Demo modeli checkpoint'ten yüklendi: {} | sınıflar: {}", ckpt, classes)
    else:
        logger.warning(
            "Demo için checkpoint yok ({}); rastgele model kullanılıyor. "
            "Önce 'psikiyatri-train' çalıştırın.",
            ckpt,
        )
    model.eval()
    return model, classes, enabled


def _scenario_to_index(scenario: str, classes: list[str], rng: np.random.Generator) -> int:
    if scenario in classes:
        return classes.index(scenario)
    if scenario == "rastgele":
        return int(rng.integers(0, len(classes)))
    return 0  # bilinmeyen -> kontrol


def _make_batch(
    scenario: str, signal: float, seed: int, classes: list[str], enabled: tuple[str, ...]
) -> ModalityBatch:
    """Senaryoya göre sentetik bir seans örneği (B=1) üretir.

    Sinyal, eğitimdeki ile aynı biçimde enjekte edilir (bkz.
    data/synthetic_data.py): kontrolde kayma yok; diğer tanılarda ses/
    giyilebilir kanallara sınıfa özel kayma uygulanır.
    """
    rng = np.random.default_rng(seed)
    n = len(classes)
    idx = _scenario_to_index(scenario, classes, rng)
    shift = 0.0 if idx == 0 else signal

    fields: dict[str, torch.Tensor] = {}
    if "vision" in enabled:
        fields["vision"] = torch.from_numpy(
            rng.standard_normal((1, 8, 468, 3)).astype("float32")
        )
    if "audio" in enabled:
        mfcc = (rng.standard_normal((1, 50, 40)) + shift * 0.3).astype("float32")
        egemaps = (rng.standard_normal((1, 88)) + shift).astype("float32")
        if n > 2 and idx != 0:
            seg = slice((idx - 1) * 15, (idx - 1) * 15 + 15)
            egemaps[0, seg] += signal * 1.5
        fields["audio_mfcc"] = torch.from_numpy(mfcc)
        fields["audio_egemaps"] = torch.from_numpy(egemaps)
    if "wearable" in enabled:
        wear = rng.standard_normal((1, 60, NUM_WEARABLE_CHANNELS)).astype("float32")
        if idx != 0:
            wear[0, :, (idx - 1) % NUM_WEARABLE_CHANNELS] += signal
        fields["wearable"] = torch.from_numpy(wear)
    return ModalityBatch(**fields)


def run_demo(
    scenario: str,
    signal: float,
    seed: int,
    modalities: list[str],
) -> dict:
    """Demo örneğini modelden geçirip yapılandırılmış sonuç döndürür."""
    if not modalities:
        raise ValueError("En az bir modalite seçilmelidir.")

    model, classes, enabled = _load()
    requested = [m for m in modalities if m in enabled]
    if not requested:
        requested = [enabled[0]]

    batch = _make_batch(scenario, signal, seed, classes, enabled)
    # İstenmeyen modaliteleri maskele
    if "vision" not in requested:
        batch.vision = None
    if "audio" not in requested:
        batch.audio_mfcc = None
        batch.audio_egemaps = None
    if "wearable" not in requested:
        batch.wearable = None

    with torch.no_grad():
        out = model(batch)
    probs = torch.softmax(out.logits, dim=-1).squeeze(0)

    present = batch.present_modalities
    return {
        "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))},
        "predicted_label": classes[int(probs.argmax())],
        "incongruence": float(out.incongruence[0]),
        "modality_contributions": [
            {"modality": m, "weight": float(out.attention[0, i])}
            for i, m in enumerate(present)
        ],
    }


__all__ = ["DEMO_CAPABLE", "run_demo"]
