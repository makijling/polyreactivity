"""Loader for the Harvey et al. 2022 dataset."""

from __future__ import annotations

import pandas as pd

from .utils import standardize_frame

_COLUMN_ALIASES = {
    "id": ("id", "clone_id"),
    "heavy_seq": ("heavy", "heavy_chain", "sequence"),
    "light_seq": ("light", "light_chain"),
    "label": ("polyreactive", "is_polyreactive"),
}

_LABEL_MAP = {
    "polyreactive": 1,
    "non-polyreactive": 0,
    "positive": 1,
    "negative": 0,
    1: 1,
    0: 0,
    "1": 1,
    "0": 0,
}


def load_dataframe(path_or_url: str, heavy_only: bool = True) -> pd.DataFrame:
    """Load the Harvey dataset into the canonical format."""

    frame = pd.read_csv(path_or_url)
    return standardize_frame(
        frame,
        source="harvey2022",
        heavy_only=heavy_only,
        column_aliases=_COLUMN_ALIASES,
        label_map=_LABEL_MAP,
        is_test=True,
    )
