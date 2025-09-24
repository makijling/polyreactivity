from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from polyreact.features.anarsi import NumberedResidue, NumberedSequence
from polyreact.features.descriptors import DescriptorConfig, DescriptorFeaturizer


@dataclass(slots=True)
class _DummyNumberer:
    regions: dict[str, str]

    def number_sequence(self, sequence: str) -> NumberedSequence:
        residues = [
            NumberedResidue(position=i + 1, insertion=" ", amino_acid=aa)
            for i, aa in enumerate(sequence)
        ]
        mapped = {key: value for key, value in self.regions.items()}
        mapped.setdefault("full", sequence)
        return NumberedSequence(
            sequence=sequence,
            scheme="imgt",
            chain_type="H",
            residues=residues,
            regions=mapped,
        )


def _protparam_features(seq: str, ph: float) -> dict[str, float]:
    clean = seq.replace(" ", "").upper()
    if not clean:
        return {
            "length": 0.0,
            "hydropathy": 0.0,
            "aromaticity": 0.0,
            "pI": 0.0,
            "net_charge": 0.0,
            "charge": 0.0,
        }
    analysis = ProteinAnalysis(clean)
    net_charge = analysis.charge_at_pH(ph)
    length = float(len(clean))
    return {
        "length": length,
        "hydropathy": analysis.gravy(),
        "aromaticity": analysis.aromaticity(),
        "pI": analysis.isoelectric_point(),
        "net_charge": net_charge,
        "charge": net_charge / length,
    }


def test_descriptor_features_match_expected_values():
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    regions = {
        "CDRH1": sequence[:3],
        "CDRH2": sequence[3:6],
        "CDRH3": sequence[6:],
    }
    config = DescriptorConfig(
        use_anarci=True,
        regions=["CDRH1", "CDRH2", "CDRH3"],
        features=[
            "length",
            "hydropathy",
            "aromaticity",
            "pI",
            "net_charge",
            "charge",
        ],
        ph=7.4,
    )
    featurizer = DescriptorFeaturizer(
        config=config,
        numberer=_DummyNumberer(regions=regions),
        standardize=False,
    )

    table = featurizer.compute_feature_table([sequence])
    assert list(table.columns) == [
        "CDRH1_length",
        "CDRH1_hydropathy",
        "CDRH1_aromaticity",
        "CDRH1_pI",
        "CDRH1_net_charge",
        "CDRH1_charge",
        "CDRH2_length",
        "CDRH2_hydropathy",
        "CDRH2_aromaticity",
        "CDRH2_pI",
        "CDRH2_net_charge",
        "CDRH2_charge",
        "CDRH3_length",
        "CDRH3_hydropathy",
        "CDRH3_aromaticity",
        "CDRH3_pI",
        "CDRH3_net_charge",
        "CDRH3_charge",
    ]

    for region_name, region_sequence in regions.items():
        expected = _protparam_features(region_sequence, ph=7.4)
        for feature_name, expected_value in expected.items():
            column = f"{region_name}_{feature_name}"
            assert column in table.columns
            assert table.at[0, column] == pytest.approx(expected_value, rel=1e-5, abs=1e-5)


def test_descriptor_fit_transform_returns_scaled_features():
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    config = DescriptorConfig(
        use_anarci=False,
        regions=["full"],
        features=["length", "hydropathy"],
        ph=7.4,
    )
    featurizer = DescriptorFeaturizer(config=config, standardize=True)
    transformed = featurizer.fit_transform([sequence])
    assert transformed.shape == (1, 2)
    assert np.allclose(transformed, 0.0)
