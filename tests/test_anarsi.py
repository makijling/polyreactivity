import pytest

anarcii = pytest.importorskip("anarcii")

from polyreact.features.anarsi import AnarciNumberer


def test_number_sequence_returns_cdr_regions():
    sequence = (
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWV"
        "SAISSYGSSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARRGYYYGMDV"
    )

    numberer = AnarciNumberer(cpu=True, ncpu=1, verbose=False)
    result = numberer.number_sequence(sequence)

    assert result.chain_type == "H"
    assert result.scheme.lower() == "imgt"
    assert result.regions["CDRH1"] == "GFTFSSYA"
    assert result.regions["CDRH2"] == "ISSYGSST"
    assert result.regions["CDRH3"] == "ARRGYYYGMD"
    assert result.regions["full"].startswith("EVQLV")
