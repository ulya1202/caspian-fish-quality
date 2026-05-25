from __future__ import annotations

import pandas as pd

from caspian_fish_quality.transfer import (
    SturgeonCase,
    SturgeonReference,
    default_sturgeon_cases,
    default_sturgeon_references,
    h_divergence_proxy,
    run_transfer_test,
)


def test_default_sturgeon_cases_three_species() -> None:
    cases = default_sturgeon_cases()
    assert len(cases) == 3
    species = {c.species for c in cases}
    assert "Acipenser stellatus" in species
    assert "Acipenser baerii" in species
    assert "Huso huso" in species


def test_default_sturgeon_references_alias() -> None:
    assert default_sturgeon_references() == default_sturgeon_cases()


def test_sturgeon_reference_is_sturgeon_case() -> None:
    assert SturgeonReference is SturgeonCase


def test_run_transfer_test_returns_27_rows(
    phd_df: pd.DataFrame,
    static_df: pd.DataFrame,
    storage_df: pd.DataFrame,
) -> None:
    out = run_transfer_test(phd_df, static_df, storage_df)
    n_models = out["model"].nunique()
    assert len(out) == 9 * n_models
    assert n_models >= 2
    assert {
        "species",
        "target",
        "model",
        "MAPE",
        "actual",
        "predicted",
        "error_pct",
    } <= set(out.columns)


def test_run_transfer_test_ridge_protein_huso(
    phd_df: pd.DataFrame,
    static_df: pd.DataFrame,
    storage_df: pd.DataFrame,
) -> None:
    out = run_transfer_test(
        phd_df, static_df, storage_df, models=("Ridge",)
    )
    row = out[
        (out["species"] == "Huso huso")
        & (out["target"] == "Protein (%)")
        & (out["model"] == "Ridge")
    ].iloc[0]
    assert row["actual"] == 14.73
    assert abs(row["error_pct"]) < 10.0


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


def test_sturgeon_case_dataclass_is_frozen() -> None:
    case = default_sturgeon_cases()[0]
    import dataclasses

    assert dataclasses.is_dataclass(case)
    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        case.species = "Y"  # type: ignore[misc]
