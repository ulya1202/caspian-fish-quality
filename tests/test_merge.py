from __future__ import annotations

import pandas as pd


def test_merge_static_has_required_prefixes(static_df: pd.DataFrame) -> None:
    assert "group" in static_df.columns
    cols = static_df.columns.tolist()
    assert any(c.startswith("bio_") for c in cols)
    assert any(c.startswith("idx_") for c in cols)
    assert any(c.startswith("cut_") for c in cols)
    assert any(c.startswith("fa_") for c in cols)


def test_merge_static_first_column_is_group(static_df: pd.DataFrame) -> None:
    assert static_df.columns[0] == "group"


def test_merge_static_keeps_sample_id(static_df: pd.DataFrame) -> None:
    assert "sample_id" in static_df.columns


def test_merge_storage_has_day_suffixed_columns(storage_df: pd.DataFrame) -> None:
    cols = storage_df.columns.tolist()
    assert any("day" in c.lower() for c in cols)


def test_merge_storage_first_column_is_group(storage_df: pd.DataFrame) -> None:
    assert storage_df.columns[0] == "group"


def test_merge_static_no_duplicate_columns(static_df: pd.DataFrame) -> None:
    assert len(static_df.columns) == len(set(static_df.columns))
