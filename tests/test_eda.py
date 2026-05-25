from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from caspian_fish_quality.eda.parsing import (
    auto_clean_values,
    clean_data,
    clean_scientific_notation,
    coerce_numeric,
    divide_by_symbol,
)
from caspian_fish_quality.eda.regex_extract import (
    extract_sample_sizes,
    split_mean_sd_columns,
)
from caspian_fish_quality.eda.summary import describe_dict, table_shapes


def test_divide_by_symbol_splits_pm() -> None:
    df = pd.DataFrame({"x": ["1.0 ± 0.1", "2.0 ± 0.2"]})
    out = divide_by_symbol("±", df, "x", "x_mean", "x_sd")
    assert list(out.columns) == ["x_mean", "x_sd"]
    assert list(out["x_mean"]) == ["1.0", "2.0"]


def test_clean_scientific_notation_known_inputs() -> None:
    assert clean_scientific_notation("1.5 × 10<sup>-3</sup>") == 1.5e-3
    assert clean_scientific_notation("2.0 × 10\u22125") == 2.0e-5
    assert clean_scientific_notation("not numeric") == "notnumeric"
    assert clean_scientific_notation(3.14) == 3.14


@given(st.floats(min_value=-1e10, max_value=1e10, allow_nan=False))
def test_clean_scientific_notation_idempotent_on_floats(x: float) -> None:
    assert clean_scientific_notation(x) == x


def test_auto_clean_values_strips_html_and_converts() -> None:
    assert auto_clean_values("<sup>2</sup>3.5") == 23.5
    assert auto_clean_values(7.5) == 7.5
    assert auto_clean_values("1.5 × 10<sup>-3</sup>") == pytest.approx(1.5e-3)


def test_clean_data_applies_known_corrections() -> None:
    df1 = pd.DataFrame(
        {
            "Biometric Traits": ["Bodymaximumheight(cm)", "Other"],
            "AG (n=50) Mean": [0.01, 1.0],
        }
    )
    df6 = pd.DataFrame(
        {
            "Fatty Acids": ["C20:1n-9", "C16:0"],
            "AG (n=6) SEM": ["1.23e-", "0.5"],
            "RG (n=6) SEM": ["0.4", "0.6"],
        }
    )
    out = clean_data({1: df1, 6: df6})
    assert out[1].loc[0, "AG (n=50) Mean"] == pytest.approx(10.65)
    assert out[6].loc[0, "AG (n=6) SEM"] == pytest.approx(1.23e-05)


def test_coerce_numeric_preserves_object_columns() -> None:
    df = pd.DataFrame({"a": ["1", "2", "3"], "b": ["x", "y", "z"]})
    out = coerce_numeric(df)
    assert pd.api.types.is_numeric_dtype(out["a"])
    assert out["b"].tolist() == ["x", "y", "z"]


def test_extract_sample_sizes_finds_n_pattern() -> None:
    df = pd.DataFrame(columns=["AG (n=50)", "RG (n=50)", "Other"])
    sizes = extract_sample_sizes({1: df})
    assert sizes[1] == {50}


def test_split_mean_sd_columns_handles_combined_header() -> None:
    df = pd.DataFrame({"AG: 1.0 ± 0.1": ["1.0 ± 0.1"]})
    out = split_mean_sd_columns({1: df})
    cols = list(out[1].columns)
    assert any("AG 1.0" in c for c in cols)


def test_table_shapes_returns_summary() -> None:
    out = table_shapes({1: pd.DataFrame({"a": [1, 2, 3]})})
    assert out.loc[1, "rows"] == 3


def test_describe_dict_includes_object_when_requested() -> None:
    out = describe_dict({1: pd.DataFrame({"a": ["x", "y"]})}, include_object=True)
    assert "a" in out[1].columns
