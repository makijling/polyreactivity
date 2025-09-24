"""ANARCI/ANARCII numbering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

try:
    from anarcii.pipeline import Anarcii  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Anarcii = None


@dataclass(slots=True)
class NumberedResidue:
    """Single residue with IMGT numbering metadata."""

    position: int
    insertion: str
    amino_acid: str


@dataclass(slots=True)
class NumberedSequence:
    """Container for numbering results and derived regions."""

    sequence: str
    scheme: str
    chain_type: str
    residues: list[NumberedResidue]
    regions: dict[str, str]


_IMGT_HEAVY_REGIONS: Sequence[Tuple[str, int, int]] = (
    ("FR1", 1, 26),
    ("CDRH1", 27, 38),
    ("FR2", 39, 55),
    ("CDRH2", 56, 65),
    ("FR3", 66, 104),
    ("CDRH3", 105, 117),
    ("FR4", 118, 128),
)

_IMGT_LIGHT_REGIONS: Sequence[Tuple[str, int, int]] = (
    ("FR1", 1, 26),
    ("CDRL1", 27, 38),
    ("FR2", 39, 55),
    ("CDRL2", 56, 65),
    ("FR3", 66, 104),
    ("CDRL3", 105, 117),
    ("FR4", 118, 128),
)

_REGION_MAP: dict[Tuple[str, str], Sequence[Tuple[str, int, int]]] = {
    ("imgt", "H"): _IMGT_HEAVY_REGIONS,
    ("imgt", "L"): _IMGT_LIGHT_REGIONS,
}

_VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
_DEFAULT_SCHEME = "imgt"
_DEFAULT_CHAIN = "H"


_DEFAULT_NUMBERER: AnarciNumberer | None = None


def _sanitize_sequence(sequence: str) -> str:
    return "".join(residue for residue in sequence.upper() if residue in _VALID_AMINO_ACIDS)


def get_default_numberer() -> AnarciNumberer:
    global _DEFAULT_NUMBERER
    if _DEFAULT_NUMBERER is None:
        _DEFAULT_NUMBERER = AnarciNumberer(chain_type=_DEFAULT_CHAIN, cpu=True, ncpu=1, verbose=False)
    return _DEFAULT_NUMBERER


def trim_variable_domain(
    sequence: str,
    *,
    numberer: AnarciNumberer | None = None,
    scheme: str = _DEFAULT_SCHEME,
    chain_type: str = _DEFAULT_CHAIN,
    fallback_length: int = 130,
) -> str:
    """Return the FR1–FR4 variable domain for a heavy/light chain sequence."""

    cleaned = _sanitize_sequence(sequence)
    if not cleaned:
        return ""

    active_numberer = numberer or get_default_numberer()
    try:
        numbered = active_numberer.number_sequence(cleaned)
    except Exception:  # pragma: no cover - best effort safeguard
        return cleaned[:fallback_length]

    region_sets = _region_boundaries(scheme, chain_type)
    pieces: list[str] = []
    for region_name, _start, _end in region_sets:
        segment = numbered.regions.get(region_name, "")
        if segment:
            pieces.append(segment)
    trimmed = "".join(pieces)
    if not trimmed:
        trimmed = numbered.regions.get("full", "")
    if not trimmed:
        trimmed = cleaned[:fallback_length]
    return trimmed


def _normalise_chain_type(chain_type: str) -> str:
    upper = chain_type.upper()
    if upper in {"H", "HV"}:
        return "H"
    if upper in {"L", "K", "LV", "KV"}:
        return "L"
    return upper


class AnarciNumberer:
    """Thin wrapper around the ANARCII pipeline to obtain IMGT regions."""

    def __init__(
        self,
        *,
        scheme: str = "imgt",
        chain_type: str = "H",
        cpu: bool = True,
        ncpu: int = 1,
        verbose: bool = False,
    ) -> None:
        if Anarcii is None:  # pragma: no cover - optional dependency guard
            msg = (
                "anarcii is required for numbering but is not installed."
                " Install 'anarcii' to enable ANARCI-based features."
            )
            raise ImportError(msg)
        self.scheme = scheme
        self.expected_chain_type = _normalise_chain_type(chain_type)
        self.cpu = cpu
        self.ncpu = ncpu
        self.verbose = verbose
        self._runner = None

    def _ensure_runner(self) -> Anarcii:
        if self._runner is None:
            self._runner = Anarcii(
                seq_type="antibody",
                mode="accuracy",
                batch_size=1,
                cpu=self.cpu,
                ncpu=self.ncpu,
                verbose=self.verbose,
            )
        return self._runner

    def number_sequence(self, sequence: str) -> NumberedSequence:
        """Return numbering metadata for a single amino-acid sequence."""

        runner = self._ensure_runner()
        output = runner.number([sequence])
        record = next(iter(output.values()))
        if record.get("error"):
            raise RuntimeError(f"ANARCI failed: {record['error']}")

        scheme = record.get("scheme", self.scheme)
        detected_chain = record.get("chain_type", self.expected_chain_type)
        normalised_chain = _normalise_chain_type(detected_chain)
        if self.expected_chain_type and normalised_chain != self.expected_chain_type:
            msg = (
                f"Expected chain type {self.expected_chain_type!r} but got"
                f" {normalised_chain!r}"
            )
            raise ValueError(msg)

        residues = [
            NumberedResidue(position=pos, insertion=ins, amino_acid=aa)
            for (pos, ins), aa in record["numbering"]
        ]
        regions = _extract_regions(
            residues=residues,
            scheme=scheme,
            chain_type=normalised_chain,
        )
        return NumberedSequence(
            sequence=sequence,
            scheme=scheme,
            chain_type=normalised_chain,
            residues=residues,
            regions=regions,
        )


@lru_cache(maxsize=32)
def _region_boundaries(scheme: str, chain_type: str) -> Sequence[Tuple[str, int, int]]:
    key = (scheme.lower(), chain_type.upper())
    return _REGION_MAP.get(key, ())


def _extract_regions(
    *,
    residues: Sequence[NumberedResidue],
    scheme: str,
    chain_type: str,
) -> dict[str, str]:
    boundaries = _region_boundaries(scheme, chain_type)
    slots: Dict[str, List[str]] = {name: [] for name, _, _ in boundaries}
    slots["full"] = []

    for residue in residues:
        aa = residue.amino_acid
        if aa == "-":
            continue
        slots["full"].append(aa)
        for name, start, end in boundaries:
            if start <= residue.position <= end:
                slots.setdefault(name, []).append(aa)
                break

    return {key: "".join(value) for key, value in slots.items()}
