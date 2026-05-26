from __future__ import annotations

from caspian_fish_quality.synth.correlation_loader import (
    get_correlations,
    get_storage_correlations,
    load_correlation_priors,
)


def test_load_correlation_priors_has_simeanu_doi() -> None:
    meta = load_correlation_priors()["meta"]
    assert "10.3390/agriculture12122144" in meta["primary_marginal_source"]["doi"]


def test_get_correlations_table_keys() -> None:
    c = get_correlations()
    assert set(c.keys()) == {1, 2, 3, 6}
    assert c[1][("bodymass", "totallength")] == 0.95


def test_get_storage_correlations_losses_water() -> None:
    stor = get_storage_correlations()
    assert stor[("losses", "water")] == 0.80
