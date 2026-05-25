"""Shared pytest fixtures for caspian_fish_quality tests."""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pandas as pd
import pytest

from caspian_fish_quality.merge import merge_static, merge_storage
from caspian_fish_quality.ml.datasets import generate_phd_dataset
from caspian_fish_quality.synth.generators import (
    generate_all_synthetic,
    load_default_df_dict,
)


@pytest.fixture(scope="session")
def seeded_rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def tiny_df_dict() -> dict[int, pd.DataFrame]:
    """Six minimal source tables used as parser fixtures."""
    return load_default_df_dict()


@pytest.fixture(scope="session")
def synthetic_dict(
    tiny_df_dict: dict[int, pd.DataFrame],
) -> dict[int, pd.DataFrame]:
    return generate_all_synthetic(tiny_df_dict, n_per_group=200, seed=42)


@pytest.fixture(scope="session")
def static_df(synthetic_dict: dict[int, pd.DataFrame]) -> pd.DataFrame:
    return merge_static(synthetic_dict)


@pytest.fixture(scope="session")
def storage_df(synthetic_dict: dict[int, pd.DataFrame]) -> pd.DataFrame:
    return merge_storage(synthetic_dict)


@pytest.fixture(scope="session")
def phd_df() -> pd.DataFrame:
    return generate_phd_dataset(n_per_group=200, seed=42)


@pytest.fixture()
def fresh_rng() -> Generator[np.random.Generator, None, None]:
    yield np.random.default_rng(42)
