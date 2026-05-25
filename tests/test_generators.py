from __future__ import annotations

import math

import numpy as np
import pandas as pd

from caspian_fish_quality.synth.generators import (
    calc_fa_indices,
    calc_yields,
    generate_all_synthetic,
    get_correlations,
    get_storage_correlations,
    parse_mean_sem_str,
    parse_type_a,
    parse_type_b,
    parse_type_c,
    to_float,
    validate,
)


def test_to_float_handles_strings_and_na_tokens() -> None:
    assert to_float("1.5") == 1.5
    assert math.isnan(to_float("ND"))
    assert math.isnan(to_float(None))
    assert to_float(2.5) == 2.5


def test_parse_mean_sem_str() -> None:
    assert parse_mean_sem_str("1.0 ± 0.1") == (1.0, 0.1)
    assert parse_mean_sem_str("foo") == (None, None)


def test_parse_type_a_extracts_per_group(tiny_df_dict: dict[int, pd.DataFrame]) -> None:
    out = parse_type_a(tiny_df_dict[1])
    assert "RG" in out
    assert "AG" in out
    for grp_df in out.values():
        assert {"feature", "mean", "sem", "sd", "min", "max", "n"} <= set(grp_df.columns)


def test_parse_type_b_storage_table(tiny_df_dict: dict[int, pd.DataFrame]) -> None:
    out = parse_type_b(tiny_df_dict[4])
    assert len(out) >= 1


def test_parse_type_c_returns_groups(tiny_df_dict: dict[int, pd.DataFrame]) -> None:
    out = parse_type_c(tiny_df_dict[6])
    for grp in out.values():
        assert "is_derived" in grp.columns


def test_get_correlations_keys() -> None:
    cs = get_correlations()
    assert {1, 2, 3, 6} <= set(cs.keys())


def test_get_storage_correlations_returns_pairs() -> None:
    s = get_storage_correlations()
    assert ("losses", "water") in s


def test_calc_yields_creates_yield_columns() -> None:
    df = pd.DataFrame({"Livemass(g)": [1000.0], "Carcassmass(g)": [500.0]})
    out = calc_yields(df)
    assert any("yield" in c.lower() for c in out.columns)


def test_calc_fa_indices_creates_summary_columns() -> None:
    df = pd.DataFrame(
        {
            "C14:0": [2.0],
            "C16:0": [20.0],
            "C18:0": [5.0],
            "C16:1": [3.0],
            "C18:1": [15.0],
            "C18:2 n-6": [10.0],
            "C18:3 n-3": [2.0],
            "C20:5 n-3": [3.0],
            "C22:6 n-3": [8.0],
        }
    )
    out = calc_fa_indices(df)
    assert "ΣSFA" in out.columns
    assert "ΣPUFA" in out.columns
    assert "AI" in out.columns
    assert "TI" in out.columns


def test_generate_all_synthetic_round_trip(
    tiny_df_dict: dict[int, pd.DataFrame],
) -> None:
    syn = generate_all_synthetic(tiny_df_dict, n_per_group=50, seed=42)
    assert {1, 2, 3, 4, 5, 6} <= set(syn.keys())
    for k, v in syn.items():
        assert "group" in v.columns
        assert len(v) > 0
        _ = k


def test_validate_returns_dataframe(synthetic_dict: dict[int, pd.DataFrame]) -> None:
    out = validate(synthetic_dict, 1)
    assert isinstance(out, pd.DataFrame)
    assert "group" in out.columns


def test_to_float_handles_dates() -> None:
    assert math.isnan(to_float({"obj": "weird"}))
    assert isinstance(to_float("2026-04-27"), float)


def test_generate_all_synthetic_seeded_reproducible(
    tiny_df_dict: dict[int, pd.DataFrame],
) -> None:
    a = generate_all_synthetic(tiny_df_dict, n_per_group=20, seed=42)
    b = generate_all_synthetic(tiny_df_dict, n_per_group=20, seed=42)
    np.testing.assert_array_almost_equal(
        a[1].select_dtypes("number").to_numpy(),
        b[1].select_dtypes("number").to_numpy(),
    )
