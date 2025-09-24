"""Utility helpers for dataset loading."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import pandas as pd

EXPECTED_COLUMNS = ("id", "heavy_seq", "light_seq", "label")
OPTIONAL_COLUMNS = ("source", "is_test")

LOGGER = logging.getLogger("polyreact.data")

_DEFAULT_ALIASES: dict[str, Sequence[str]] = {
    "id": ("id", "sequence_id", "antibody_id", "uid"),
    "heavy_seq": ("heavy_seq", "heavy", "heavy_chain", "H", "H_chain"),
    "light_seq": ("light_seq", "light", "light_chain", "L", "L_chain"),
    "label": ("label", "polyreactive", "is_polyreactive", "class", "target"),
}

DEFAULT_LABEL_MAP: dict[str | int | float | bool, int] = {
    1: 1,
    0: 0,
    "1": 1,
    "0": 0,
    True: 1,
    False: 0,
    "true": 1,
    "false": 0,
    "polyreactive": 1,
    "non-polyreactive": 0,
    "poly": 1,
    "non": 0,
    "positive": 1,
    "negative": 0,
}


def _normalize_label_key(value: object) -> object:
    if isinstance(value, str):
        trimmed = value.strip().lower()
        if trimmed in {
            "polyreactive",
            "non-polyreactive",
            "poly",
            "non",
            "positive",
            "negative",
            "high",
            "low",
            "pos",
            "neg",
            "1",
            "0",
            "true",
            "false",
        }:
            return trimmed
        if trimmed.isdigit():
            return trimmed
    return value


def ensure_columns(frame: pd.DataFrame, *, heavy_only: bool = True) -> pd.DataFrame:
    """Validate and coerce dataframe columns to the canonical format."""

    frame = frame.copy()
    for column in ("id", "heavy_seq", "label"):
        if column not in frame.columns:
            msg = f"Required column '{column}' missing from dataframe"
            raise KeyError(msg)

    if "light_seq" not in frame.columns:
        frame["light_seq"] = ""

    if heavy_only:
        frame["light_seq"] = ""

    frame["id"] = frame["id"].astype(str)
    frame["heavy_seq"] = frame["heavy_seq"].fillna("").astype(str)
    frame["light_seq"] = frame["light_seq"].fillna("").astype(str)
    frame["label"] = frame["label"].astype(int)

    ordered = list(EXPECTED_COLUMNS) + [
        col for col in frame.columns if col not in EXPECTED_COLUMNS
    ]
    return frame[ordered]


def standardize_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    heavy_only: bool = True,
    column_aliases: dict[str, Sequence[str]] | None = None,
    label_map: dict[str | int | float | bool, int] | None = None,
    is_test: bool | None = None,
) -> pd.DataFrame:
    """Rename columns using aliases and coerce labels to integers."""

    aliases = {**_DEFAULT_ALIASES}
    if column_aliases:
        for key, values in column_aliases.items():
            aliases[key] = tuple(values) + tuple(aliases.get(key, ()))

    rename_map: dict[str, str] = {}
    for target, candidates in aliases.items():
        if target in frame.columns:
            continue
        for candidate in candidates:
            if candidate in frame.columns and candidate not in rename_map:
                rename_map[candidate] = target
                break

    normalized = frame.rename(columns=rename_map).copy()

    if "light_seq" not in normalized.columns:
        normalized["light_seq"] = ""

    label_lookup = label_map or DEFAULT_LABEL_MAP
    normalized["label"] = normalized["label"].map(lambda x: label_lookup.get(_normalize_label_key(x)))

    if normalized["label"].isnull().any():
        msg = "Label column contains unmapped or missing values"
        raise ValueError(msg)

    normalized["source"] = source
    if is_test is not None:
        normalized["is_test"] = bool(is_test)

    normalized = ensure_columns(normalized, heavy_only=heavy_only)
    return normalized


def deduplicate_sequences(
    frames: Iterable[pd.DataFrame],
    *,
    heavy_only: bool = True,
    key_columns: Sequence[str] | None = None,
    keep_intra_frames: set[int] | None = None,
) -> list[pd.DataFrame]:
    """Remove duplicate entries across multiple dataframes with configurable keys."""

    if key_columns is None:
        key_columns = ["heavy_seq"] if heavy_only else ["heavy_seq", "light_seq"]
    keep_intra_frames = keep_intra_frames or set()

    seen: set[tuple[str, ...]] = set()
    cleaned: list[pd.DataFrame] = []

    for frame_idx, frame in enumerate(frames):
        valid_columns = [col for col in key_columns if col in frame.columns]
        if not valid_columns:
            valid_columns = ["heavy_seq"]

        mask: list[bool] = []
        frame_seen: set[tuple[str, ...]] = set()
        allow_intra = frame_idx in keep_intra_frames

        for values in frame[valid_columns].itertuples(index=False, name=None):
            key = tuple(_normalise_key_value(value) for value in values)
            if key in seen:
                mask.append(False)
                continue
            if not allow_intra and key in frame_seen:
                mask.append(False)
                continue
            mask.append(True)
            frame_seen.add(key)
        seen.update(frame_seen)
        filtered = frame.loc[mask].reset_index(drop=True)
        removed = len(frame) - len(filtered)
        if removed:
            dataset = "<unknown>"
            if "source" in frame.columns and not frame["source"].empty:
                dataset = str(frame["source"].iloc[0])
            LOGGER.info("Removed %s duplicate sequences from %s", removed, dataset)
        cleaned.append(filtered)
    return cleaned


def _normalise_key_value(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()
