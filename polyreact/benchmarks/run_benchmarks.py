"""Run end-to-end benchmarks for the polyreactivity model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .. import train as train_cli

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = PROJECT_ROOT / "tests" / "fixtures" / "boughter.csv"
DEFAULT_EVAL = [
    PROJECT_ROOT / "tests" / "fixtures" / "jain.csv",
    PROJECT_ROOT / "tests" / "fixtures" / "shehata.csv",
    PROJECT_ROOT / "tests" / "fixtures" / "harvey.csv",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run polyreactivity benchmarks")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--train",
        default=str(DEFAULT_TRAIN),
        help="Training dataset CSV (defaults to bundled fixture).",
    )
    parser.add_argument(
        "--eval",
        nargs="+",
        default=[str(path) for path in DEFAULT_EVAL],
        help="Evaluation dataset CSV paths (>=1).",
    )
    parser.add_argument(
        "--report-dir",
        default="artifacts",
        help="Directory to write metrics, predictions, and plots.",
    )
    parser.add_argument(
        "--model-path",
        default="artifacts/model.joblib",
        help="Destination for the trained model artifact.",
    )
    parser.add_argument(
        "--backend",
        choices=["descriptors", "plm", "concat"],
        help="Override feature backend during training.",
    )
    parser.add_argument("--plm-model", help="Optional PLM model override.")
    parser.add_argument("--cache-dir", help="Embedding cache directory override.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        help="Device override for embeddings.",
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Use paired heavy/light chains when available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for PLM embedding batches.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if len(args.eval) < 1:
        parser.error("Provide at least one evaluation dataset via --eval.")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_args: list[str] = [
        "--config",
        args.config,
        "--train",
        args.train,
        "--save-to",
        str(Path(args.model_path)),
        "--report-to",
        str(report_dir),
        "--batch-size",
        str(args.batch_size),
    ]

    train_args.extend(["--eval", *args.eval])

    if args.backend:
        train_args.extend(["--backend", args.backend])
    if args.plm_model:
        train_args.extend(["--plm-model", args.plm_model])
    if args.cache_dir:
        train_args.extend(["--cache-dir", args.cache_dir])
    if args.device:
        train_args.extend(["--device", args.device])
    if args.paired:
        train_args.append("--paired")

    return train_cli.main(train_args)


if __name__ == "__main__":
    raise SystemExit(main())
