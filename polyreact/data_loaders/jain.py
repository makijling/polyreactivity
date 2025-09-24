"""Loader for the Jain et al. 2017 dataset."""

from __future__ import annotations

import pandas as pd

from .utils import standardize_frame

_COLUMN_ALIASES = {
    "id": ("id", "antibody_id"),
    "heavy_seq": ("heavy", "heavy_sequence", "H_chain"),
    "light_seq": ("light", "light_sequence", "L_chain"),
    "label": ("class", "polyreactive"),
}

_LABEL_MAP = {
    "polyreactive": 1,
    "non-polyreactive": 0,
    "reactive": 1,
    "non-reactive": 0,
    1: 1,
    0: 0,
    1.0: 1,
    0.0: 0,
    "1": 1,
    "0": 0,
}


def load_dataframe(path_or_url: str, heavy_only: bool = True) -> pd.DataFrame:
    """Load the Jain dataset into the canonical format."""

    frame = pd.read_csv(path_or_url)
    return standardize_frame(
        frame,
        source="jain2017",
        heavy_only=heavy_only,
        column_aliases=_COLUMN_ALIASES,
        label_map=_LABEL_MAP,
        is_test=True,
    )
