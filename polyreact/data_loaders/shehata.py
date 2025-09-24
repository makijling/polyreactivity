"""Loader for the Shehata et al. (2019) PSR dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .utils import standardize_frame

SHEHATA_SOURCE = "shehata2019"

_COLUMN_ALIASES = {
    "id": (
        "antibody_id",
        "antibody",
        "antibody name",
        "antibody_name",
        "sequence_name",
        "Antibody Name",
    ),
    "heavy_seq": (
        "heavy",
        "heavy_chain",
        "heavy aa",
        "heavy_sequence",
        "vh",
        "vh_sequence",
        "heavy chain aa",
        "Heavy Chain AA",
    ),
    "light_seq": (
        "light",
        "light_chain",
        "light aa",
        "light_sequence",
        "vl",
        "vl_sequence",
        "light chain aa",
        "Light Chain AA",
    ),
    "label": (
        "polyreactive",
        "binding_class",
        "binding class",
        "psr_class",
        "psr binding",
        "psr classification",
        "Binding class",
        "Binding Class",
    ),
}

_LABEL_MAP = {
    "polyreactive": 1,
    "non-polyreactive": 0,
    "positive": 1,
    "negative": 0,
    "high": 1,
    "low": 0,
    "pos": 1,
    "neg": 0,
    1: 1,
    0: 0,
    1.0: 1,
    0.0: 0,
    "1": 1,
    "0": 0,
}

_PSR_SCORE_ALIASES: tuple[str, ...] = (
    "psr score",
    "psr_score",
    "psr overall score",
    "overall score",
    "psr z",
    "psr_z",
)


def _clean_sequence(sequence: object) -> str:
    if isinstance(sequence, str):
        return "".join(sequence.split()).upper()
    return ""


def _maybe_extract_psr_scores(frame: pd.DataFrame) -> pd.DataFrame:
    scores: dict[str, pd.Series] = {}
    for column in frame.columns:
        lowered = column.strip().lower()
        if any(alias in lowered for alias in _PSR_SCORE_ALIASES):
            key = lowered.replace(" ", "_")
            scores[key] = frame[column]
    if not scores:
        return pd.DataFrame(index=frame.index)
    renamed = {}
    for name, series in scores.items():
        cleaned_name = name
        for prefix in ("psr_", "overall_"):
            if cleaned_name.startswith(prefix):
                cleaned_name = cleaned_name[len(prefix) :]
                break
        cleaned_name = cleaned_name.replace("__", "_")
        cleaned_name = cleaned_name.replace("(", "").replace(")", "")
        cleaned_name = cleaned_name.replace("-", "_")
        renamed[f"psr_{cleaned_name}"] = pd.to_numeric(series, errors="coerce")
    return pd.DataFrame(renamed)


def _pick_source_label(path: Path | None) -> str:
    if path is None:
        return SHEHATA_SOURCE
    stem = path.stem.lower()
    if "curated" in stem or "subset" in stem:
        return f"{SHEHATA_SOURCE}_curated"
    return SHEHATA_SOURCE


def _standardize(
    frame: pd.DataFrame,
    *,
    heavy_only: bool,
    source: str,
) -> pd.DataFrame:
    standardized = standardize_frame(
        frame,
        source=source,
        heavy_only=heavy_only,
        column_aliases=_COLUMN_ALIASES,
        label_map=_LABEL_MAP,
        is_test=True,
    )

    psr_scores = _maybe_extract_psr_scores(frame)

    mask = standardized["heavy_seq"].map(_clean_sequence) != ""
    standardized = standardized.loc[mask].copy()
    standardized.reset_index(drop=True, inplace=True)
    standardized["heavy_seq"] = standardized["heavy_seq"].map(_clean_sequence)
    standardized["light_seq"] = standardized["light_seq"].map(_clean_sequence)

    if not psr_scores.empty:
        psr_scores = psr_scores.loc[mask]
        psr_scores = psr_scores.reset_index(drop=True)
        for column in psr_scores.columns:
            standardized[column] = psr_scores[column].reset_index(drop=True)

    return standardized


def _read_excel(path: Path, *, heavy_only: bool) -> pd.DataFrame:
    excel = pd.ExcelFile(path, engine="openpyxl")
    sheet_candidates: Iterable[str] = excel.sheet_names

    def _score(name: str) -> tuple[int, str]:
        lowered = name.lower()
        priority = 0
        if "psr" in lowered or "polyreactivity" in lowered:
            priority = 2
        elif "sheet" not in lowered:
            priority = 1
        return (-priority, name)

    sheet_name = sorted(sheet_candidates, key=_score)[0]
    raw = excel.parse(sheet_name)
    raw = raw.dropna(how="all")
    return _standardize(raw, heavy_only=heavy_only, source=_pick_source_label(path))


def load_dataframe(path_or_url: str, heavy_only: bool = True) -> pd.DataFrame:
    """Load the Shehata dataset into the canonical format.

    Supports both pre-processed CSV exports and the original Excel supplement
    (*.xls/*.xlsx). Additional PSR score columns are preserved when available.
    """

    lower = path_or_url.lower()
    source_override: str | None = None
    if lower.startswith("http://") or lower.startswith("https://"):
        if lower.endswith((".xls", ".xlsx")):
            raw = pd.read_excel(path_or_url, engine="openpyxl")
            return _standardize(raw, heavy_only=heavy_only, source=SHEHATA_SOURCE)
        frame = pd.read_csv(path_or_url)
        return _standardize(frame, heavy_only=heavy_only, source=SHEHATA_SOURCE)

    path = Path(path_or_url)
    source_override = _pick_source_label(path)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        engine = "openpyxl" if path.suffix.lower() == ".xlsx" else None
        if engine:
            frame = _read_excel(path, heavy_only=heavy_only)
        else:
            frame = pd.read_excel(path, engine=None)
            frame = _standardize(frame, heavy_only=heavy_only, source=source_override)
        frame["source"] = source_override
        return frame

    frame = pd.read_csv(path)
    standardized = _standardize(frame, heavy_only=heavy_only, source=source_override)
    standardized["source"] = source_override
    return standardized
