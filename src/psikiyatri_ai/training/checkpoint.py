"""Model checkpoint yan-verisi (sınıf adları / modaliteler).

Ağırlık dosyası (``.pt``) sınıf adlarını ve modalite kümesini saklamaz.
Servis tarafının (API/demo) etiket sırasını ve model yapılandırmasını
eğitimle eşleştirebilmesi için checkpoint'in yanına küçük bir
``classes.json`` yazılır. Böylece ikili/çok-sınıflı model aynı kodla
servis edilir.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

METADATA_NAME = "classes.json"


def save_metadata(
    model_dir: Path,
    classes: Sequence[str],
    modalities: Sequence[str],
    *,
    temperature: float | None = None,
    card: dict | None = None,
) -> Path:
    """Sınıf/modalite yan-verisini ``<model_dir>/classes.json`` içine yazar.

    Ek olarak (varsa) kalibrasyon ``temperature`` ve bir **model kartı**
    (``card``: git commit, veri hash'i, eğitim config'i, metrikler, tarih)
    saklanır — böylece hangi model sürümünün hangi çıktıyı ürettiği
    izlenebilir olur.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "classes": list(classes),
        "num_classes": len(classes),
        "modalities": list(modalities),
    }
    if temperature is not None:
        meta["temperature"] = round(float(temperature), 4)
    if card:
        meta["card"] = card
    path = model_dir / METADATA_NAME
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_model_card(
    *, classes: Sequence[str], modalities: Sequence[str], metrics: dict, extra: dict | None = None
) -> dict:
    """İzlenebilirlik için model kartı üretir (git commit + veri + metrikler).

    ``Date.now`` yerine sistem saati kullanılabildiğinden burada zaman
    damgası çağıran tarafça verilebilir (``extra['egitim_tarihi']``).
    """
    import subprocess  # noqa: PLC0415

    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=5, check=True
            ).stdout.strip()
        except Exception:  # noqa: BLE001 - git yoksa/hata → None
            return None

    card = {
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "classes": list(classes),
        "modalities": list(modalities),
        "metrics": metrics,
    }
    if extra:
        card.update(extra)
    return {k: v for k, v in card.items() if v is not None}


def load_metadata(model_dir: Path) -> dict | None:
    """Yan-veriyi okur; yoksa ``None`` döner (servis ikili varsayılana düşer)."""
    path = Path(model_dir) / METADATA_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# Füzyon sınıflandırıcısının son katman anahtarı (num_classes = ilk boyut).
_CLASSIFIER_HEAD_KEY = "fusion.classifier.3.weight"


def num_classes_from_state(state: dict, default: int) -> int:
    """Checkpoint durumundan sınıf sayısını (sınıflandırıcı başı) çıkarır.

    ``classes.json`` yan-verisi olmayan bir checkpoint yüklenirken, servis
    modelinin doğru ``num_classes`` ile kurulup boyut uyuşmazlığından
    (yükleme çökmesi) kaçınması için kullanılır.
    """
    weight = state.get(_CLASSIFIER_HEAD_KEY)
    if weight is None:
        return default
    return int(weight.shape[0])


def labels_for_state(state: dict, default_labels: list[str]) -> list[str]:
    """Yan-veri yokken checkpoint'e uygun etiket listesi türetir.

    Sınıf sayısı varsayılanla aynıysa varsayılan etiketleri, farklıysa
    genel ``sinif_i`` etiketlerini döndürür (çökme yerine zarif geri düşüş).
    """
    n = num_classes_from_state(state, len(default_labels))
    if n == len(default_labels):
        return list(default_labels)
    return [f"sinif_{i}" for i in range(n)]


__all__ = [
    "METADATA_NAME",
    "build_model_card",
    "labels_for_state",
    "load_metadata",
    "num_classes_from_state",
    "save_metadata",
]
