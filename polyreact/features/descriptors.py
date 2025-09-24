"""Sequence descriptor features for polyreactivity prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.preprocessing import StandardScaler

from .anarsi import AnarciNumberer, NumberedSequence

_VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(slots=True)
class DescriptorConfig:
    """Configuration for descriptor-based features."""

    use_anarci: bool = True
    regions: Sequence[str] = ("CDRH1", "CDRH2", "CDRH3")
    features: Sequence[str] = (
        "length",
        "charge",
        "hydropathy",
        "aromaticity",
        "pI",
        "net_charge",
    )
    ph: float = 7.4


class DescriptorFeaturizer:
    """Compute descriptor features with optional ANARCI-based regions."""

    def __init__(
        self,
        *,
        config: DescriptorConfig,
        numberer: AnarciNumberer | None = None,
        standardize: bool = True,
    ) -> None:
        self.config = config
        self.numberer = numberer if not config.use_anarci else numberer or AnarciNumberer()
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        self.feature_names_: list[str] | None = None

    def fit(self, sequences: Iterable[str]) -> "DescriptorFeaturizer":
        table = self.compute_feature_table(sequences)
        values = table.to_numpy(dtype=float)
        if self.standardize and self.scaler is not None:
            self.scaler.fit(values)
        self.feature_names_ = list(table.columns)
        return self

    def transform(self, sequences: Iterable[str]) -> np.ndarray:
        if self.feature_names_ is None:
            msg = "DescriptorFeaturizer must be fitted before calling transform."
            raise RuntimeError(msg)
        table = self.compute_feature_table(sequences)
        values = table.to_numpy(dtype=float)
        if self.standardize and self.scaler is not None:
            values = self.scaler.transform(values)
        return values

    def fit_transform(self, sequences: Iterable[str]) -> np.ndarray:
        table = self.compute_feature_table(sequences)
        values = table.to_numpy(dtype=float)
        if self.standardize and self.scaler is not None:
            self.scaler.fit(values)
            values = self.scaler.transform(values)
        self.feature_names_ = list(table.columns)
        return values

    def compute_feature_table(self, sequences: Iterable[str]) -> pd.DataFrame:
        rows: list[dict[str, float]] = []
        for sequence in sequences:
            regions = self._prepare_regions(sequence)
            if not self.config.use_anarci:
                region_names = ["FULL"]
            else:
                region_names = [region.upper() for region in self.config.regions]
            row: dict[str, float] = {}
            for region_name in region_names:
                normalized_name = region_name.upper()
                region_sequence = regions.get(normalized_name, "")
                for feature_name in self.config.features:
                    column = f"{normalized_name}_{feature_name}"
                    row[column] = _compute_feature(
                        region_sequence,
                        feature_name,
                        ph=self.config.ph,
                    )
            rows.append(row)

        if not self.config.use_anarci:
            region_names = ["FULL"]
        else:
            region_names = [region.upper() for region in self.config.regions]
        columns = [
            f"{region}_{feature}"
            for region in region_names
            for feature in self.config.features
        ]
        frame = pd.DataFrame(rows, columns=columns)
        return frame.fillna(0.0)

    def _prepare_regions(self, sequence: str) -> dict[str, str]:
        if not self.config.use_anarci:
            return {"FULL": sequence}

        try:
            numbered: NumberedSequence = self.numberer.number_sequence(sequence)
        except (RuntimeError, ValueError):
            return {}
        return {key.upper(): value for key, value in numbered.regions.items()}


def _sanitize_sequence(sequence: str) -> str:
    return "".join(residue for residue in sequence.upper() if residue in _VALID_AMINO_ACIDS)


def _compute_feature(sequence: str, feature_name: str, *, ph: float) -> float:
    sanitized = _sanitize_sequence(sequence)
    if not sanitized:
        return 0.0

    analysis = ProteinAnalysis(sanitized)
    if feature_name == "length":
        return float(len(sanitized))
    if feature_name == "hydropathy":
        return float(analysis.gravy())
    if feature_name == "aromaticity":
        return float(analysis.aromaticity())
    if feature_name == "pI":
        return float(analysis.isoelectric_point())
    if feature_name == "net_charge":
        return float(analysis.charge_at_pH(ph))
    if feature_name == "charge":
        net = analysis.charge_at_pH(ph)
        return float(net / len(sanitized))
    msg = f"Unsupported feature: {feature_name}"
    raise ValueError(msg)
