"""Loader for the Boughter et al. 2020 dataset."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .utils import LOGGER, standardize_frame

_COLUMN_ALIASES = {
    "id": ("sequence_id",),
    "heavy_seq": ("heavy", "heavy_chain"),
    "light_seq": ("light", "light_chain"),
    "label": ("polyreactive",),
}


def _find_flag_columns(columns: Iterable[str]) -> list[str]:
    flag_cols: list[str] = []
    for column in columns:
        normalized = column.lower().replace(" ", "")
        if "flag" in normalized:
            flag_cols.append(column)
    return flag_cols


def _apply_flag_policy(frame: pd.DataFrame, flag_columns: list[str]) -> pd.DataFrame:
    if not flag_columns:
        return frame

    flag_values = (
        frame[flag_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    flag_binary = (flag_values > 0).astype(int)
    flags_total = flag_binary.sum(axis=1)

    specific_mask = flags_total == 0
    nonspecific_mask = flags_total >= 4
    keep_mask = specific_mask | nonspecific_mask

    dropped = int((~keep_mask).sum())
    if dropped:
        LOGGER.info("Dropped %s mildly polyreactive sequences (1-3 ELISA flags)", dropped)

    filtered = frame.loc[keep_mask].copy()
    filtered["flags_total"] = flags_total.loc[keep_mask].astype(int)
    filtered["label"] = np.where(nonspecific_mask.loc[keep_mask], 1, 0)
    filtered["polyreactive"] = filtered["label"]
    return filtered


def load_dataframe(path_or_url: str, heavy_only: bool = True) -> pd.DataFrame:
    """Load the Boughter dataset into the canonical format."""

    frame = pd.read_csv(path_or_url)
    flag_columns = _find_flag_columns(frame.columns)
    frame = _apply_flag_policy(frame, flag_columns)

    label_series = frame.get("label")
    if label_series is not None:
        frame = frame[label_series.isin({0, 1})].copy()

    standardized = standardize_frame(
        frame,
        source="boughter2020",
        heavy_only=heavy_only,
        column_aliases=_COLUMN_ALIASES,
        is_test=False,
    )
    if "flags_total" in frame.columns and "flags_total" not in standardized.columns:
        standardized["flags_total"] = frame["flags_total"].to_numpy(dtype=int)
    return standardized
