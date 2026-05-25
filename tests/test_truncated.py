from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import kstest, truncnorm

from caspian_fish_quality.synth.moments import sem_to_sd
from caspian_fish_quality.synth.truncated import sample_truncated


def test_sem_to_sd_known() -> None:
    assert sem_to_sd(1.0, 4) == pytest.approx(2.0)
    assert sem_to_sd(0.5, 100) == pytest.approx(5.0)


def test_sem_to_sd_validates() -> None:
    with pytest.raises(ValueError):
        sem_to_sd(-1.0, 5)
    with pytest.raises(ValueError):
        sem_to_sd(1.0, 0)


def test_sample_truncated_constant_when_sd_zero(fresh_rng: np.random.Generator) -> None:
    out = sample_truncated(5.0, 0.0, 0.0, 10.0, 100, fresh_rng)
    assert np.allclose(out, 5.0)


def test_sample_truncated_respects_bounds(fresh_rng: np.random.Generator) -> None:
    out = sample_truncated(5.0, 1.0, 0.0, 10.0, 1000, fresh_rng)
    assert (out >= 0.0).all()
    assert (out <= 10.0).all()


def test_sample_truncated_distribution_matches_scipy(
    fresh_rng: np.random.Generator,
) -> None:
    n = 10_000
    mean, sd, lo, hi = 5.0, 1.5, 2.0, 8.0
    samples = sample_truncated(mean, sd, lo, hi, n, fresh_rng)
    a, b = (lo - mean) / sd, (hi - mean) / sd
    stat, p = kstest(samples, lambda x: truncnorm.cdf(x, a, b, loc=mean, scale=sd))
    assert stat < 0.05
    _ = p


def test_sample_truncated_negative_n_raises(fresh_rng: np.random.Generator) -> None:
    with pytest.raises(ValueError):
        sample_truncated(0.0, 1.0, -1.0, 1.0, -5, fresh_rng)


def test_sample_truncated_handles_nan_bounds(fresh_rng: np.random.Generator) -> None:
    out = sample_truncated(10.0, 1.0, math.nan, math.nan, 500, fresh_rng)
    assert len(out) == 500
    assert np.all(np.isfinite(out))


def test_sample_truncated_clamps_mean_outside_bounds(
    fresh_rng: np.random.Generator,
) -> None:
    out = sample_truncated(100.0, 1.0, 0.0, 10.0, 200, fresh_rng)
    assert out.min() >= 0.0
    assert out.max() <= 10.0


@given(
    mean=st.floats(min_value=-100, max_value=100, allow_nan=False),
    sd=st.floats(min_value=0.01, max_value=20, allow_nan=False),
    delta=st.floats(min_value=0.5, max_value=10, allow_nan=False),
    n=st.integers(min_value=10, max_value=200),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=30, deadline=None)
def test_sample_truncated_inside_bounds_property(
    mean: float, sd: float, delta: float, n: int, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    lo = mean - delta
    hi = mean + delta
    out = sample_truncated(mean, sd, lo, hi, n, rng)
    assert (out >= lo - 1e-9).all()
    assert (out <= hi + 1e-9).all()
