from __future__ import annotations

import pandas as pd

from caspian_fish_quality.transfer import (
    SturgeonReference,
    default_sturgeon_references,
    h_divergence_proxy,
    run_transfer_test,
)


def test_default_sturgeon_references_three_species() -> None:
    refs = default_sturgeon_references()
    assert len(refs) == 3
    species = {r.species for r in refs}
    assert "Acipenser persicus" in species


def test_run_transfer_test_returns_27_rows(
    phd_df: pd.DataFrame,
    static_df: pd.DataFrame,
    storage_df: pd.DataFrame,
) -> None:
    out = run_transfer_test(phd_df, static_df, storage_df)
    assert len(out) == 3 * 3 * 3
    assert {"species", "target", "model", "MAPE", "direction_match"} <= set(out.columns)


def test_run_transfer_test_rejects_unknown_models(
    phd_df: pd.DataFrame,
    static_df: pd.DataFrame,
    storage_df: pd.DataFrame,
) -> None:
    import pytest

    with pytest.raises(ValueError):
        run_transfer_test(phd_df, static_df, storage_df, models=("Nonexistent",))


def test_h_divergence_proxy_close_to_zero_for_same_distribution(
    phd_df: pd.DataFrame,
) -> None:
    shuffled = phd_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    half = len(shuffled) // 2
    a = shuffled.iloc[:half].select_dtypes(include="number")
    b = shuffled.iloc[half:].select_dtypes(include="number")
    out = h_divergence_proxy(a, b)
    assert -0.6 < out["h_divergence_proxy"] < 0.6


def test_sturgeon_reference_dataclass_is_frozen() -> None:
    ref = SturgeonReference(
        species="X",
        bodymass_g=1.0,
        protein_perc=2.0,
        lipids_perc=3.0,
        citation="Test",
    )
    import dataclasses

    assert dataclasses.is_dataclass(ref)
    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.species = "Y"  # type: ignore[misc]
