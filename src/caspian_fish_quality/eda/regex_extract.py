"""Regex helpers: sample-size detection and ``±`` column splitting."""

from __future__ import annotations

import re

import pandas as pd

from caspian_fish_quality.eda.parsing import divide_by_symbol

_N_PATTERN = re.compile(r"n=(\d+)")


def extract_sample_sizes(df_dict: dict[int, pd.DataFrame]) -> dict[int, set[int]]:
    """Find every ``n=<int>`` token in column headers across all tables.

    Parameters
    ----------
    df_dict : dict[int, pandas.DataFrame]

    Returns
    -------
    dict[int, set[int]]
        Per-table set of sample sizes detected in column headers.
    """
    out: dict[int, set[int]] = {}
    for key, df in df_dict.items():
        sizes: set[int] = set()
        for column in df.columns:
            match = _N_PATTERN.search(str(column))
            if match:
                sizes.add(int(match.group(1)))
        if sizes:
            out[key] = sizes
    return out


def split_mean_sd_columns(df_dict: dict[int, pd.DataFrame]) -> dict[int, pd.DataFrame]:
    """Split every column whose name embeds ``±`` into ``_mean`` / ``_std``.

    Headers may take the form ``"AG (n=50): 12.3 ± 0.4"`` (colon-separated)
    or just ``"12.3 ± 0.4"``; both are handled.

    Parameters
    ----------
    df_dict : dict[int, pandas.DataFrame]

    Returns
    -------
    dict[int, pandas.DataFrame]
        Updated dictionary with ``±`` columns split.
    """
    out: dict[int, pd.DataFrame] = {}
    for key, df in df_dict.items():
        local = df.copy()
        columns_to_change = [col for col in local.columns if "±" in str(col)]
        for column in columns_to_change:
            try:
                parts = column.split(":")
                main_name = parts[0].strip()
                if len(parts) > 1 and "±" in parts[1]:
                    sub_parts = parts[1].split("±")
                    new_col_name_1 = f"{main_name} {sub_parts[0].strip()}"
                    new_col_name_2 = f"{main_name} {sub_parts[1].strip()}"
                    local = divide_by_symbol("±", local, column, new_col_name_1, new_col_name_2)
                else:
                    local = divide_by_symbol("±", local, column, f"{column}_mean", f"{column}_std")
            except (KeyError, IndexError):
                continue
        out[key] = local
    return out


__all__ = ["extract_sample_sizes", "split_mean_sd_columns"]
