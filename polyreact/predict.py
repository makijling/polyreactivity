"""Command-line interface for polyreactivity predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .api import predict_batch
from .config import load_config
from .utils.io import read_table, write_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Polyreactivity prediction CLI")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV or JSONL file with sequences.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write predictions CSV.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--backend",
        choices=["plm", "descriptors", "concat"],
        help="Override feature backend from config.",
    )
    parser.add_argument(
        "--plm-model",
        help="Override PLM model name.",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to trained model artifact (joblib).",
    )
    parser.add_argument(
        "--heavy-only",
        dest="heavy_only",
        action="store_true",
        default=True,
        help="Use only heavy chains (default).",
    )
    parser.add_argument(
        "--paired",
        dest="heavy_only",
        action="store_false",
        help="Use paired heavy/light chains if available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for model inference (PLM backend).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        help="Computation device override.",
    )
    parser.add_argument(
        "--cache-dir",
        help="Cache directory for embeddings.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    df = read_table(args.input)

    if "heavy_seq" not in df.columns and "heavy" not in df.columns:
        parser.error("Input file must contain a 'heavy_seq' column (or 'heavy').")
    if df.get("heavy_seq", df.get("heavy", "")).fillna("").str.len().eq(0).all():
        parser.error("At least one non-empty heavy sequence is required.")

    predictions = predict_batch(
        df.to_dict("records"),
        config=config,
        backend=args.backend,
        plm_model=args.plm_model,
        weights=args.weights,
        heavy_only=args.heavy_only,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    write_table(predictions, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
