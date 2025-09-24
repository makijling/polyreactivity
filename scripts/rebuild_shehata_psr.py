"""Rebuild the Shehata et al. (2019) PSR evaluation dataset.

This script consumes the protective supplementary Excel file (\*.xlsx) that
contains the 398-entry PSR table with qualitative High/Low labels and optional
score columns. Users must download the file manually due to Cloudflare gating
on the Cell website and place it in ``data/raw/shehata/`` before running:

```
python scripts/rebuild_shehata_psr.py \
    --input data/raw/shehata/mmc1.xlsx \
    --output data/processed/shehata_full.csv
```

An audit JSON with summary statistics is written alongside the CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from polyreact.data_loaders import shehata

DEFAULT_INPUT = Path("data/raw/shehata/mmc1.xlsx")
DEFAULT_OUTPUT = Path("data/processed/shehata_full.csv")


def _build_audit(frame: pd.DataFrame, input_path: Path) -> dict[str, object]:
    score_columns = [column for column in frame.columns if column.startswith("psr_")]
    summary = {
        "input_path": str(input_path),
        "rows": int(len(frame)),
        "positives": int(frame["label"].sum()),
        "negatives": int((frame["label"] == 0).sum()),
        "positive_rate": float(frame["label"].mean()) if len(frame) else 0.0,
        "score_columns": score_columns,
    }
    return summary


def rebuild(input_path: Path, output_path: Path, *, heavy_only: bool = True) -> None:
    frame = shehata.load_dataframe(str(input_path), heavy_only=heavy_only)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    audit = _build_audit(frame, input_path)
    audit_path = output_path.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(f"Wrote {len(frame)} rows to {output_path}")
    print(f"Positives: {audit['positives']} | Negatives: {audit['negatives']}")
    if audit["score_columns"]:
        print("PSR score columns detected:", ", ".join(audit["score_columns"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild Shehata PSR dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the PSR Excel file (download manually).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination CSV path (default: data/processed/shehata_full.csv)",
    )
    parser.add_argument(
        "--include-light",
        action="store_true",
        help="Preserve light-chain sequences instead of trimming to heavy-only.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file {args.input} not found. Download mmc1.xlsx from Cell's "
            "supplemental materials and place it at this path."
        )

    rebuild(args.input, args.output, heavy_only=not args.include_light)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
