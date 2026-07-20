"""Psikiyatrik NLP Excel veri setinden anonim manifest üretici (CLI).

KULLANIM:
    python scripts/ingest_nlp_dataset.py \
        --xlsx data/external/psychiatric_nlp_dataset_20260627.xlsx

Ham Excel ``data/external/`` altında tutulur (git'e girmez). Üretilen
``data/processed/nlp_manifest.jsonl`` doğrudan eğitim hattıyla
kullanılabilir:

    psikiyatri-train --manifest data/processed/nlp_manifest.jsonl

Ayrıntı: docs/veri-formati.md ve psikiyatri_ai.data.nlp_dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from psikiyatri_ai.config import get_settings
from psikiyatri_ai.data.nlp_dataset import build_manifest_from_nlp_dataset
from psikiyatri_ai.logging_conf import setup_logging


def main() -> None:
    settings = get_settings()
    ext = settings.data_dir / "external"
    # En güncel klinik sürümü tercih et; yoksa ilk sürüme düş.
    default_xlsx = ext / "psychiatric_nlp_dataset_klinik_20260719.xlsx"
    if not default_xlsx.exists():
        default_xlsx = ext / "psychiatric_nlp_dataset_20260627.xlsx"

    parser = argparse.ArgumentParser(
        description="Psikiyatrik NLP Excel veri setinden anonim manifest üret."
    )
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=default_xlsx,
        help=f"Excel veri seti yolu (varsayılan: {default_xlsx})",
    )
    parser.add_argument("--out", type=Path, default=None, help="Manifest çıktı yolu")
    parser.add_argument(
        "--limit", type=int, default=None, help="En fazla N örnek yaz (deneme için)"
    )
    args = parser.parse_args()

    setup_logging()
    out_path, stats = build_manifest_from_nlp_dataset(
        args.xlsx, out_path=args.out, limit=args.limit
    )

    print(f"\nManifest: {out_path}")
    print(f"Yazılan örnek : {stats.written}")
    print(f"Toplam satır  : {stats.total}")
    skipped = (
        stats.skipped_no_label
        + stats.skipped_unknown_label
        + stats.skipped_empty_text
    )
    print(
        f"Atlanan       : {skipped} "
        f"(etiketsiz={stats.skipped_no_label}, "
        f"bilinmeyen={stats.skipped_unknown_label}, "
        f"boş metin={stats.skipped_empty_text})"
    )
    print("Etiket dağılımı:")
    for label, count in stats.label_counts.most_common():
        print(f"  {count:6d}  {label}")


if __name__ == "__main__":
    main()
