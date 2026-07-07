"""Ham seans klasörlerinden anonim manifest üretici (CLI).

KULLANIM:
    python scripts/build_manifest.py --raw-root data/raw/sessions

Beklenen ham yapı için bkz. docs/veri-formati.md ve
psikiyatri_ai.data.manifest_builder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from psikiyatri_ai.data.manifest_builder import build_manifest_from_raw
from psikiyatri_ai.logging_conf import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Ham seanslardan anonim manifest üret.")
    parser.add_argument("--raw-root", type=Path, required=True, help="Ham seanslar kök dizini")
    parser.add_argument("--out", type=Path, default=None, help="Manifest çıktı yolu")
    args = parser.parse_args()

    setup_logging()
    build_manifest_from_raw(args.raw_root, out_path=args.out)


if __name__ == "__main__":
    main()
