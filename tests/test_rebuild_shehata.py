from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import rebuild_shehata_psr


def test_rebuild_shehata_creates_expected_csv(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "Antibody Name": ["A1", "A2"],
            "Heavy Chain AA": ["QVQLVQSGAEVKKPGSSVKVSCK", "QVQLVESGGGVVQPGGSLRLSCAAS"],
            "Light Chain AA": ["DIQMTQSPSSLSASVGDRVTITC", "EIVLTQSPATLSLSPGERATLSC"],
            "Binding class": ["High", "Low"],
            "PSR Score": [1.23, 0.45],
        }
    )
    input_path = tmp_path / "mmc1_sample.xlsx"
    data.to_excel(input_path, index=False)

    output_path = tmp_path / "shehata.csv"
    rebuild_shehata_psr.rebuild(input_path, output_path)

    frame = pd.read_csv(output_path)
    assert {"id", "heavy_seq", "light_seq", "label", "source"}.issubset(frame.columns)
    assert frame["label"].tolist() == [1, 0]

    audit_path = output_path.with_suffix(".audit.json")
    assert audit_path.exists()
    import json
    audit = json.loads(audit_path.read_text())
    assert int(audit["rows"]) == 2
