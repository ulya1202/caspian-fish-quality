"""Column-cleaning helpers for the six peer-reviewed source tables.

These functions are direct ports of the notebook's exploratory column
parsers, with Russian comments translated to English and ``applymap``
migrated to ``DataFrame.map`` (pandas 2.x).
"""

from __future__ import annotations

import re
from typing import Any, cast

import pandas as pd


def divide_by_symbol(
    symb: str,
    df: pd.DataFrame,
    target_column: str,
    new_col_1: str,
    new_col_2: str,
) -> pd.DataFrame:
    """Split a column on ``symb`` into two new columns and drop the original.

    The original notebook used this to split mean/SEM columns of the form
    ``"12.3 ± 0.4"`` into separate ``mean`` and ``std`` columns.

    Parameters
    ----------
    symb : str
        Delimiter character (typically ``±``).
    df : pandas.DataFrame
        DataFrame to mutate (a copy is not made).
    target_column : str
        Column whose values will be split.
    new_col_1, new_col_2 : str
        Names assigned to the two resulting columns.

    Returns
    -------
    pandas.DataFrame
        The (mutated) DataFrame.
    """
    split_data = df[target_column].astype(str).str.split(symb, expand=True)
    df = df.drop(columns=[target_column])
    if split_data.shape[1] == 2:
        df[new_col_1] = split_data[0].str.strip()
        df[new_col_2] = split_data[1].str.strip()
    return df


def clean_scientific_notation(text: Any) -> Any:
    """Convert Unicode-formatted scientific notation to a Python ``float``.

    Strips ``<sup>`` HTML wrappers, replaces ``× 10`` with ``e``, normalises
    Unicode minus to ASCII hyphen, and collapses whitespace.

    Parameters
    ----------
    text : object
        Possibly a string with embedded scientific notation.

    Returns
    -------
    float or object
        ``float`` when conversion succeeds; the input value otherwise.
    """
    if not isinstance(text, str):
        return text

    text = re.sub(r"<sup>", "", text)
    text = re.sub(r"</sup>", "", text)
    text = re.sub(r"\s*×\s*10", "e", text)
    text = text.replace("\u2212", "-").replace(" ", "")

    try:
        return float(text)
    except ValueError:
        return text


def auto_clean_values(val: Any) -> Any:
    """Aggressive cleanup for table cell values.

    Drops HTML tags, normalises scientific notation, and attempts numeric
    conversion. Used inside ``DataFrame.map`` (formerly ``applymap``).

    Parameters
    ----------
    val : object
        Cell value.

    Returns
    -------
    float or object
        ``float`` on successful conversion; original-or-cleaned string
        otherwise.
    """
    if not isinstance(val, str):
        return val

    clean_val = re.sub(r"<[^>]+>", "", val)
    clean_val = clean_val.replace("×", "e").replace("10", "").replace("\u2212", "-")
    clean_val = re.sub(r"\s+", "", clean_val)

    if re.search(r"\d+e-?\d+", clean_val):
        try:
            return float(clean_val)
        except ValueError:
            pass

    try:
        return float(clean_val)
    except ValueError:
        return clean_val


def clean_data(df_dict: dict[int, pd.DataFrame]) -> dict[int, pd.DataFrame]:
    """Apply known per-table corrections from the notebook's QA pass.

    Two corrections are documented:

    * Table 1 - ``Bodymaximumheight(cm)`` AG mean was 0.01 (typo); replaced
      with the midpoint of min/max (10.65 cm).
    * Table 6 - ``C20:1n-9`` AG SEM string ``"1.23e-"`` was truncated;
      replaced with ``1.23e-05``.

    Parameters
    ----------
    df_dict : dict[int, pandas.DataFrame]
        Mapping from table index (1-6) to source DataFrame.

    Returns
    -------
    dict[int, pandas.DataFrame]
        Cleaned ``df_dict`` (a shallow copy with corrected rows).
    """
    out = dict(df_dict)

    if 1 in out:
        bio = out[1].copy()
        if "Biometric Traits" in bio.columns and "AG (n=50) Mean" in bio.columns:
            mask = bio["Biometric Traits"] == "Bodymaximumheight(cm)"
            bio.loc[mask, "AG (n=50) Mean"] = 10.65
        out[1] = bio

    if 6 in out:
        fa = out[6].copy()
        if "AG (n=6) SEM" in fa.columns:
            fa["AG (n=6) SEM"] = pd.to_numeric(fa["AG (n=6) SEM"], errors="coerce")
        if "RG (n=6) SEM" in fa.columns:
            fa["RG (n=6) SEM"] = pd.to_numeric(fa["RG (n=6) SEM"], errors="coerce")
        if "Fatty Acids" in fa.columns and "AG (n=6) SEM" in fa.columns:
            mask = fa["Fatty Acids"].astype(str).str.strip() == "C20:1n-9"
            fa.loc[mask, "AG (n=6) SEM"] = 1.23e-05
        out[6] = fa

    return out


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort numeric coercion of every column in ``df``.

    Replaces deprecated ``pd.to_numeric(..., errors='ignore')`` (pandas 2.2+)
    with an explicit per-column try/except that preserves non-numeric data.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Columns converted to numeric where every value parses; otherwise
        left untouched.
    """
    out = df.copy()
    for col in out.columns:
        try:
            converted = pd.to_numeric(out[col], errors="raise")
        except (ValueError, TypeError):
            continue
        out[col] = converted
    return cast("pd.DataFrame", out)


def auto_clean_dict(df_dict: dict[int, pd.DataFrame]) -> dict[int, pd.DataFrame]:
    """Apply :func:`auto_clean_values` and :func:`coerce_numeric` to all tables.

    Parameters
    ----------
    df_dict : dict[int, pandas.DataFrame]

    Returns
    -------
    dict[int, pandas.DataFrame]
    """
    cleaned: dict[int, pd.DataFrame] = {}
    for key, df in df_dict.items():
        mapped = df.map(auto_clean_values)
        cleaned[key] = coerce_numeric(mapped)
    return cleaned


__all__ = [
    "auto_clean_dict",
    "auto_clean_values",
    "clean_data",
    "clean_scientific_notation",
    "coerce_numeric",
    "divide_by_symbol",
]
