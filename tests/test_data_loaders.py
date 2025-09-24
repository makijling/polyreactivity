from __future__ import annotations

from pathlib import Path

import pandas as pd

from polyreact.data_loaders import boughter, harvey, jain, shehata
from polyreact.data_loaders.utils import deduplicate_sequences

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _fixture(name: str) -> str:
    path = FIXTURES / name
    if Path(name).suffix:
        return str(path)
    return str(path.with_suffix(".csv"))


def test_boughter_loader_heavy_only() -> None:
    frame = boughter.load_dataframe(_fixture("boughter"))
    assert list(frame.columns)[:4] == ["id", "heavy_seq", "light_seq", "label"]
    assert (frame["light_seq"] == "").all()
    assert frame["source"].unique().tolist() == ["boughter2020"]
    assert not frame.get("is_test", pd.Series([], dtype=bool)).any()
    assert len(frame) == 2  # mild sequences should be removed
    assert set(frame["label"]) == {0, 1}
    assert "flags_total" in frame.columns
    assert frame.loc[frame["label"] == 0, "flags_total"].iloc[0] == 0
    assert frame.loc[frame["label"] == 1, "flags_total"].iloc[0] >= 4


def test_boughter_loader_with_light_chains() -> None:
    frame = boughter.load_dataframe(_fixture("boughter"), heavy_only=False)
    assert (frame["light_seq"] != "").any()


def test_external_loaders_mark_is_test() -> None:
    jain_frame = jain.load_dataframe(_fixture("jain"))
    shehata_frame = shehata.load_dataframe(_fixture("shehata"))
    harvey_frame = harvey.load_dataframe(_fixture("harvey"))
    for frame, source in [
        (jain_frame, "jain2017"),
        (shehata_frame, "shehata2019"),
        (harvey_frame, "harvey2022"),
    ]:
        assert frame["source"].unique().tolist() == [source]
        assert frame["is_test"].all()
        assert frame["label"].isin({0, 1}).all()


def test_shehata_loader_accepts_raw_excel() -> None:
    frame = shehata.load_dataframe(_fixture("shehata_raw.xlsx"))
    assert len(frame) == 2
    assert frame["label"].tolist() == [1, 0]
    assert "psr_score_auc" in frame.columns
    assert frame.loc[0, "psr_score_auc"] == 0.92
    assert frame.loc[0, "heavy_seq"].isupper()


def test_deduplicate_sequences_removes_overlap() -> None:
    b_frame = boughter.load_dataframe(_fixture("boughter"))
    j_frame = jain.load_dataframe(_fixture("jain"))
    j_frame.loc[0, "heavy_seq"] = b_frame.loc[0, "heavy_seq"]
    cleaned_train, cleaned_test = deduplicate_sequences([b_frame, j_frame])
    assert len(cleaned_train) == len(b_frame)
    assert len(cleaned_test) == len(j_frame) - 1
