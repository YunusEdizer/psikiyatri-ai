"""Eğitilmiş modeli bir manifest üzerinde değerlendirir (dürüst rapor).

Karışıklık matrisi, sınıf-başına duyarlılık/özgüllük ve olasılık
kalibrasyonu (ECE) üretir. Metrikler tek başına doğruluktan çok daha
bilgilendiricidir; özellikle sentetik/şablonlu veride yüksek doğruluğun
yanıltıcı olabileceğini gözler önüne serer.

KULLANIM:
    python scripts/evaluate_model.py --manifest data/processed/nlp_manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    import torch

    from psikiyatri_ai.config import get_settings
    from psikiyatri_ai.data.datamodule import build_dataloaders, infer_classes, load_manifest
    from psikiyatri_ai.logging_conf import setup_logging
    from psikiyatri_ai.training.checkpoint import load_metadata
    from psikiyatri_ai.training.evaluate import evaluate_full, format_report
    from psikiyatri_ai.training.model import PsikiyatriModel
    from psikiyatri_ai.training.train import _infer_modalities

    setup_logging()
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Modeli manifest üzerinde değerlendir.")
    parser.add_argument(
        "--manifest", type=Path,
        default=settings.processed_data_dir / "nlp_manifest.jsonl",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--json-out", type=Path, default=None, help="Raporu JSON yaz")
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Manifest bulunamadı: {args.manifest}")

    modalities, use_egemaps = _infer_modalities(args.manifest)
    classes = tuple(infer_classes(load_manifest(args.manifest)))
    meta = load_metadata(settings.model_dir) or {}
    if meta.get("classes"):
        classes = tuple(meta["classes"])
    temperature = meta.get("temperature")

    _, val_loader = build_dataloaders(
        args.manifest, classes=classes, batch_size=args.batch_size,
        val_split=args.val_split, modalities=modalities,
    )

    ckpt = settings.model_dir / "psikiyatri_model.pt"
    model_cfg = settings.model.model_copy(update={"num_classes": len(classes)})
    model = PsikiyatriModel(model_cfg, enabled_modalities=modalities, use_egemaps=use_egemaps)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    else:
        print("UYARI: eğitilmiş checkpoint yok; rastgele model değerlendiriliyor.")

    rep = evaluate_full(
        model, val_loader, "cpu", class_names=list(classes), temperature=temperature
    )
    print("\n" + "=" * 60)
    print("MODEL DEĞERLENDİRME RAPORU (doğrulama kümesi)")
    print("=" * 60)
    print(format_report(rep))
    if meta.get("card"):
        print("\nModel kartı:", json.dumps(meta["card"], ensure_ascii=False))
    print(
        "\nNOT: Yüksek metrikler sentetik/şablonlu veride yanıltıcı olabilir; "
        "gerçek klinik genelleme yalnızca gerçek veriyle doğrulanır."
    )
    if args.json_out:
        args.json_out.write_text(
            json.dumps(rep.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nJSON rapor: {args.json_out}")


if __name__ == "__main__":
    main()
