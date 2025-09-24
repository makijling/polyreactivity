"""I/O helpers for reading and writing artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def read_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read CSV or JSONL files into a DataFrame."""

    path = Path(path)
    if not path.exists():
        msg = f"Input file does not exist: {path}"
        raise FileNotFoundError(msg)
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=True, **kwargs)
    if suffix in {".csv", ""}:
        return pd.read_csv(path, **kwargs)
    msg = f"Unsupported file extension: {suffix}"
    raise ValueError(msg)


def write_table(frame: pd.DataFrame, path: str | Path, *, index: bool = False, **kwargs: Any) -> None:
    """Persist a DataFrame as CSV or JSONL, creating directories as needed."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        frame.to_json(path, orient="records", lines=True, **kwargs)
    elif suffix in {".csv", ""}:
        frame.to_csv(path, index=index, **kwargs)
    else:
        msg = f"Unsupported file extension: {suffix}"
        raise ValueError(msg)
