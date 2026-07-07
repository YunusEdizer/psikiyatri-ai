"""Sentetik örnek veri üretici (CLI).

Gerçek klinik veri olmadan eğitim hattını uçtan uca denemek için
önceden çıkarılmış öznitelik (.npz) + manifest üretir.

KULLANIM:
    python scripts/generate_sample_data.py --count 300 --seed 42
    psikiyatri-train        # üretilen manifest ile eğitim
"""

from __future__ import annotations

import argparse
from pathlib import Path

from psikiyatri_ai.data.synthetic_data import generate_sample_dataset
from psikiyatri_ai.logging_conf import logger, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentetik örnek veri üret.")
    parser.add_argument("--count", type=int, default=300, help="Örnek sayısı")
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik tohumu")
    parser.add_argument("--signal", type=float, default=0.6, help="Sınıflar arası sinyal")
    parser.add_argument("--out", type=Path, default=None, help="Çıktı dizini")
    args = parser.parse_args()

    setup_logging()
    manifest = generate_sample_dataset(
        count=args.count, seed=args.seed, signal=args.signal, out_dir=args.out
    )
    logger.info("Sentetik veri üretildi: {} ({} örnek)", manifest, args.count)


if __name__ == "__main__":
    main()
