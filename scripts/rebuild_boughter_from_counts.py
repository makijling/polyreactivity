"""Reconstruct Boughter heavy-chain dataset using per-antigen flag counts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from polyreact.features.anarsi import get_default_numberer, trim_variable_domain  # noqa: E402

COUNTS_ROOT = PROJECT_ROOT / "data" / "AIMS_manuscripts" / "app_data" / "full_sequences"
OUTPUT_DEFAULT = PROJECT_ROOT / "data" / "processed" / "boughter_counts.csv"


@dataclass(slots=True)
class DatasetConfig:
    name: str
    fasta_path: Path
    counts_path: Path
    count_kind: str  # "counts" | "binary"
    header: int | None = None
    active_value: int = 4  # value to assign when count_kind == "binary" and flagged
    family: str | None = None


def parse_sequences(fasta_path: Path) -> tuple[list[tuple[str, str]], dict[str, int]]:
    records = list(SeqIO.parse(fasta_path, "fasta"))
    numberer = get_default_numberer()
    sequences: list[tuple[str, str]] = []
    total_records = len(records)
    retained = 0
    empty_trimmed = 0
    translation_fail = 0
    cdrh3_missing = 0

    for record in records:
        raw = str(record.seq).upper()
        protein = _maybe_translate_sequence(raw)
        if not protein:
            translation_fail += 1
            continue
        trimmed = trim_variable_domain(protein, numberer=numberer)
        cdrh3 = _extract_cdrh3(trimmed, numberer)
        if not trimmed:
            empty_trimmed += 1
            continue
        if not cdrh3:
            cdrh3_missing += 1
        sequences.append((trimmed, cdrh3))
        retained += 1

    summary = {
        "total_records": total_records,
        "retained_sequences": retained,
        "translation_failures": translation_fail,
        "empty_after_trim": empty_trimmed,
        "missing_cdrh3": cdrh3_missing,
    }
    return sequences, summary


_NUCLEOTIDE_CHARS = set("ACGTUNKRYSWMDHBVX")
_CANONICAL_STARTS = ("QV", "EV", "DV", "AV")


def _maybe_translate_sequence(sequence: str) -> str:
    letters = {ch for ch in sequence if ch.isalpha()}
    if not letters:
        return ""
    if letters <= _NUCLEOTIDE_CHARS:
        best_peptide = ""
        best_score = float("-inf")
        for frame in range(3):
            frame_seq = sequence[frame:]
            trim_len = len(frame_seq) - (len(frame_seq) % 3)
            if trim_len <= 0:
                continue
            codons = frame_seq[:trim_len]
            translated = str(Seq(codons).translate(to_stop=False)).replace("*", "")
            score = _score_translation(translated)
            if score > best_score:
                best_peptide = translated
                best_score = score
        return best_peptide
    return sequence


def _score_translation(peptide: str) -> float:
    if not peptide:
        return float("-inf")
    score = 0.0
    has_stop = "*" in peptide
    if not has_stop:
        score += 10.0
    prefix = peptide[:4]
    if prefix.startswith(_CANONICAL_STARTS):
        score += 5.0
    # reward presence of conserved motifs
    if "C" in peptide[:25]:
        score += 1.0
    if "W" in peptide[-20:]:
        score += 1.0
    score += -peptide.count("X") * 0.5
    return score


def _extract_cdrh3(sequence: str, numberer) -> str:
    if not sequence:
        return ""
    try:
        numbered = numberer.number_sequence(sequence)
        return numbered.regions.get("CDRH3", "")
    except Exception:
        return ""


def _build_lineage_id(family: str, cdrh3: str, index: int) -> str:
    base = family or "unknown"
    payload = cdrh3 or f"idx{index:05d}"
    return f"{base}|{payload}"


def load_counts(path: Path, *, kind: str, header: int | None, active_value: int) -> pd.Series:
    if kind == "counts":
        df = pd.read_csv(path, header=header, names=["num_flags"])
        # Drop potential header row if it sneaked in as data (e.g. string value)
        df = df[pd.to_numeric(df["num_flags"], errors="coerce").notnull()].copy()
        return df["num_flags"].astype(int)
    if kind == "binary":
        labels: list[int] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip().upper()
                if not stripped:
                    continue
                if stripped in {"Y", "YES", "1", "TRUE"}:
                    labels.append(active_value)
                elif stripped in {"N", "NO", "0", "FALSE"}:
                    labels.append(0)
                else:
                    raise ValueError(f"Unrecognised binary label '{stripped}' in {path}")
        return pd.Series(labels, name="num_flags")
    raise ValueError(f"Unsupported count kind: {kind}")


def build_dataset(configs: Iterable[DatasetConfig]) -> tuple[pd.DataFrame, list[dict[str, int | str]]]:
    rows = []
    stats: list[dict[str, int | str]] = []
    for cfg in configs:
        if not cfg.fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {cfg.fasta_path}")
        if not cfg.counts_path.exists():
            raise FileNotFoundError(f"Counts file not found: {cfg.counts_path}")

        sequences, seq_summary = parse_sequences(cfg.fasta_path)
        counts = load_counts(cfg.counts_path, kind=cfg.count_kind, header=cfg.header, active_value=cfg.active_value)

        if len(sequences) != len(counts):
            raise ValueError(
                f"Sequence/count length mismatch for {cfg.name}: {len(sequences)} sequences vs {len(counts)} counts"
            )

        summary_entry = {
            "dataset": cfg.name,
            "family": cfg.family or cfg.name,
            "source_fasta": str(cfg.fasta_path),
            "source_counts": str(cfg.counts_path),
            **seq_summary,
            "count_zero": int((counts == 0).sum()),
            "count_mild": int(((counts > 0) & (counts <= 3)).sum()),
            "count_high": int((counts > 3).sum()),
        }

        for idx, ((seq, cdrh3), num_flags) in enumerate(zip(sequences, counts, strict=True)):
            if not seq:
                continue
            lineage = _build_lineage_id(cfg.family or cfg.name, cdrh3, idx)
            species = "mouse" if (cfg.family or cfg.name).lower().startswith("mouse") else "human"
            row = {
                "id": f"{cfg.name}_{idx:05d}",
                "heavy_seq": seq,
                "light_seq": "",
                "reactivity_count": int(num_flags),
                "source": cfg.name,
                "family": cfg.family or cfg.name,
                "cdrh3": cdrh3,
                "lineage": lineage,
                "species": species,
            }
            rows.append(row)

        stats.append(summary_entry)

    df = pd.DataFrame(rows)
    return df, stats


def filter_for_training(df: pd.DataFrame, *, deduplicate: bool = True) -> tuple[pd.DataFrame, dict[str, int]]:
    summary: dict[str, int] = {"input_rows": int(len(df))}
    mask = (df["reactivity_count"] == 0) | (df["reactivity_count"] > 3)
    filtered = df.loc[mask].copy()
    summary["after_flag_filter"] = int(len(filtered))
    summary["dropped_mild"] = summary["input_rows"] - summary["after_flag_filter"]
    if deduplicate:
        filtered.sort_values(by=["heavy_seq", "reactivity_count"], ascending=[True, False], inplace=True)
        filtered = filtered.drop_duplicates(subset=["heavy_seq"], keep="first").reset_index(drop=True)
    summary["after_deduplicate"] = int(len(filtered))
    summary["deduplicated"] = summary["after_flag_filter"] - summary["after_deduplicate"]
    max_len = 1022  # allow margin for BOS/EOS tokens when using ESM-1v (limit 1024)
    length_mask = filtered["heavy_seq"].str.len() <= max_len
    dropped = len(filtered) - length_mask.sum()
    if dropped:
        print(f"Dropping {dropped} sequences longer than {max_len} residues to satisfy ESM limits")
    filtered = filtered.loc[length_mask]
    summary["dropped_long"] = int(dropped)
    summary["after_length_filter"] = int(len(filtered))
    filtered["label"] = (filtered["reactivity_count"] > 3).astype(int)
    summary["positives"] = int(filtered["label"].sum())
    summary["negatives"] = int(len(filtered) - summary["positives"])
    return filtered, summary


def main(output_path: Path, *, deduplicate: bool) -> int:
    configs: list[DatasetConfig] = [
        DatasetConfig(
            name="flu",
            fasta_path=COUNTS_ROOT / "flu_fastaH.txt",
            counts_path=COUNTS_ROOT / "flu_NumReact.txt",
            count_kind="counts",
            header=0,
            family="influenza",
        ),
        DatasetConfig(
            name="gut_hiv",
            fasta_path=COUNTS_ROOT / "gut_hiv_fastaH.txt",
            counts_path=COUNTS_ROOT / "gut_hiv_NumReact.txt",
            count_kind="counts",
            header=None,
            family="hiv",
        ),
        DatasetConfig(
            name="nat_hiv",
            fasta_path=COUNTS_ROOT / "nat_hiv_fastaH.txt",
            counts_path=COUNTS_ROOT / "nat_hiv_NumReact.txt",
            count_kind="counts",
            header=None,
            family="hiv",
        ),
        DatasetConfig(
            name="nat_cntrl",
            fasta_path=COUNTS_ROOT / "nat_cntrl_fastaH.txt",
            counts_path=COUNTS_ROOT / "nat_cntrl_NumReact.txt",
            count_kind="counts",
            header=None,
            family="hiv",
        ),
        DatasetConfig(
            name="plos_hiv",
            fasta_path=COUNTS_ROOT / "plos_hiv_fastaH.txt",
            counts_path=COUNTS_ROOT / "plos_hiv_YN.txt",
            count_kind="binary",
            active_value=4,
            family="hiv",
        ),
        DatasetConfig(
            name="mouse_iga",
            fasta_path=COUNTS_ROOT / "mouse_fastaH.dat",
            counts_path=COUNTS_ROOT / "mouse_YN.txt",
            count_kind="binary",
            active_value=7,
            family="mouse_iga",
        ),
    ]

    combined, build_stats = build_dataset(configs)
    filtered, filter_summary = filter_for_training(combined, deduplicate=deduplicate)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    full_path = output_path.with_name(output_path.stem + "_full.csv")
    combined.sort_values(by="id").to_csv(full_path, index=False)
    filtered.sort_values(by="id").to_csv(output_path, index=False)

    audit = {
        "output_filtered": str(output_path),
        "output_full": str(full_path),
        "build_stats": build_stats,
        "filter_summary": filter_summary,
    }
    audit_path = output_path.with_name(output_path.stem + "_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(f"Wrote {len(filtered)} sequences to {output_path}")
    print(filtered["label"].value_counts().rename("count"))
    print(f"Audit details saved to {audit_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild Boughter dataset from ELISA flag counts")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DEFAULT,
        help="Destination CSV (default: data/processed/boughter_counts.csv)",
    )
    parser.add_argument(
        "--no-deduplicate",
        dest="deduplicate",
        action="store_false",
        help="Retain duplicate heavy sequences across families",
    )
    parser.set_defaults(deduplicate=True)
    args = parser.parse_args()
    raise SystemExit(main(args.output, deduplicate=args.deduplicate))
