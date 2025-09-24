"""Prepare public antibody polyreactivity datasets into a unified CSV format."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from Bio import SeqIO
import requests

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from polyreact.features.anarsi import get_default_numberer, trim_variable_domain  # noqa: E402

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

DATA_SOURCES = {
    "boughter/Human_Ab_Poly_Dataset_S1_v2.xlsx": "https://raw.githubusercontent.com/Tessier-Lab-UMich/Human_Ab_Polyreactivity/main/Supplemental%20Datasets/Human%20Ab%20Poly%20Dataset%20S1_v2.xlsx",
    "boughter/Human_Ab_Poly_Dataset_S2_v2.xlsx": "https://raw.githubusercontent.com/Tessier-Lab-UMich/Human_Ab_Polyreactivity/main/Supplemental%20Datasets/Human%20Ab%20Poly%20Dataset%20S2_v2.xlsx",
    "boughter/Human_Ab_Poly_Dataset_S3.xlsx": "https://raw.githubusercontent.com/Tessier-Lab-UMich/Human_Ab_Polyreactivity/main/Supplemental%20Datasets/Human%20Ab%20Poly%20Dataset%20S3.xlsx",
    "boughter/Human_Ab_Poly_Dataset_S8.xlsx": "https://raw.githubusercontent.com/Tessier-Lab-UMich/Human_Ab_Polyreactivity/main/Supplemental%20Datasets/Human%20Ab%20Poly%20Dataset%20S8.xlsx",
    "harvey/filtered_high_polyreactivity_pool.fasta": "https://kruse.hms.harvard.edu/sites/kruse.hms.harvard.edu/files/files/filtered_high_polyreactivity_pool.fasta",
    "harvey/filtered_low_polyreactivity_pool.fasta": "https://kruse.hms.harvard.edu/sites/kruse.hms.harvard.edu/files/files/filtered_low_polyreactivity_pool.fasta",
}


NUMBERER = get_default_numberer()


def _trim_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype(str).map(lambda seq: trim_variable_domain(seq, numberer=NUMBERER))


def ensure_file(relative_path: str) -> Path:
    """Download the requested resource if it is not already cached."""

    url = DATA_SOURCES.get(relative_path)
    if url is None:
        msg = f"No download URL registered for {relative_path}"
        raise KeyError(msg)

    path = DATA_RAW / relative_path
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def prepare_boughter(sample_per_class: int | None = None) -> pd.DataFrame:
    """Return processed dataframe for the Boughter aggregated dataset."""

    sources = [
        ensure_file("boughter/Human_Ab_Poly_Dataset_S1_v2.xlsx"),
        ensure_file("boughter/Human_Ab_Poly_Dataset_S2_v2.xlsx"),
    ]
    frames: list[pd.DataFrame] = []
    for path in sources:
        df = pd.read_excel(path, engine="openpyxl", header=2)
        df = df.rename(columns={"VH": "heavy_seq", "VL": "light_seq"})
        df = df[["Name", "heavy_seq", "light_seq"]].dropna(subset=["heavy_seq"])
        df["heavy_seq"] = _trim_column(df, "heavy_seq")
        df["label"] = df["Name"].str.lower().str.contains("high").astype(int)
        df["source"] = path.stem
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    if sample_per_class is not None:
        sampled = []
        for label in (0, 1):
            chunk = combined[combined["label"] == label]
            if sample_per_class < len(chunk):
                chunk = chunk.sample(sample_per_class, random_state=42)
            sampled.append(chunk)
        combined = pd.concat(sampled, ignore_index=True)

    combined = combined.rename(columns={"Name": "id"})
    combined["light_seq"] = combined["light_seq"].fillna("")
    combined = combined.drop_duplicates(subset=["heavy_seq", "light_seq", "label"])
    return combined[["id", "heavy_seq", "light_seq", "label", "source"]]


def prepare_jain() -> pd.DataFrame:
    """Return Jain et al. dataset with binary polyreactivity labels."""

    path = ensure_file("boughter/Human_Ab_Poly_Dataset_S8.xlsx")
    df = pd.read_excel(path, engine="openpyxl", header=2)
    df = df.rename(
        columns={
            "Name": "id",
            "VH": "heavy_seq",
            "VL": "light_seq",
            "SMP Score": "smp",
            "OVA Score": "ova",
            "Unnamed: 5": "label_str",
        }
    )
    df = df.dropna(subset=["heavy_seq"])
    df["heavy_seq"] = _trim_column(df, "heavy_seq")
    df["label"] = df["label_str"].str.strip().str.lower().map({"high": 1, "low": 0})
    df["source"] = "jain2017"
    return df[["id", "heavy_seq", "light_seq", "label", "source", "smp", "ova"]]


def prepare_shehata() -> pd.DataFrame:
    """Return Shehata et al. dataset derived from curated supplemental data."""

    path = ensure_file("boughter/Human_Ab_Poly_Dataset_S3.xlsx")
    df = pd.read_excel(path, engine="openpyxl", header=2)
    df = df.rename(
        columns={
            "Name": "id",
            "VH": "heavy_seq",
            "VL": "light_seq",
            "SMP": "smp",
            "Unnamed: 7": "label_str",
        }
    )
    df = df.dropna(subset=["heavy_seq"])
    df["heavy_seq"] = _trim_column(df, "heavy_seq")
    df["label"] = df["label_str"].str.strip().str.lower().map({"high": 1, "low": 0})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["source"] = "shehata2019"
    return df[["id", "heavy_seq", "light_seq", "label", "source", "smp"]]


def prepare_harvey(max_per_class: int | None = None) -> pd.DataFrame:
    """Return Harvey et al. nanobody dataset."""

    high_fasta = ensure_file("harvey/filtered_high_polyreactivity_pool.fasta")
    low_fasta = ensure_file("harvey/filtered_low_polyreactivity_pool.fasta")

    rows: list[dict[str, str | int]] = []
    for path, label in ((high_fasta, 1), (low_fasta, 0)):
        for record in SeqIO.parse(path, "fasta"):
            rows.append(
                {
                    "id": record.id,
                    "heavy_seq": trim_variable_domain(str(record.seq), numberer=NUMBERER),
                    "light_seq": "",
                    "label": label,
                    "source": "harvey2022",
                }
            )
    df = pd.DataFrame(rows)
    if max_per_class is not None:
        sampled = []
        for label in (0, 1):
            chunk = df[df["label"] == label]
            if max_per_class < len(chunk):
                chunk = chunk.sample(max_per_class, random_state=42)
            sampled.append(chunk)
        df = pd.concat(sampled, ignore_index=True)
    return df


def write_dataset(df: pd.DataFrame, name: str) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")


def main(sample_per_class: int | None, harvey_per_class: int | None) -> None:
    write_dataset(prepare_boughter(sample_per_class=sample_per_class), "boughter")
    write_dataset(prepare_jain(), "jain")
    write_dataset(prepare_shehata(), "shehata")
    write_dataset(prepare_harvey(max_per_class=harvey_per_class), "harvey")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare public polyreactivity datasets")
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=5000,
        help="Optional number of examples per class to sample for the Boughter dataset",
    )
    parser.add_argument(
        "--harvey-per-class",
        type=int,
        default=500,
        help="Optional number of nanobody sequences per class to sample from the Harvey dataset",
    )
    args = parser.parse_args()
    main(sample_per_class=args.sample_per_class, harvey_per_class=args.harvey_per_class)
