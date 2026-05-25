from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from caspian_fish_quality.validation import (
    correlation_residual,
    frobenius_distance,
    ks_per_variable,
    mwu_per_variable,
    tstr_protocol,
    wasserstein_per_variable,
)


def test_ks_per_variable_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"x": rng.standard_normal(500)})
    b = pd.DataFrame({"x": rng.standard_normal(500)})
    out = ks_per_variable(a, b)
    assert (out["p_raw"] > 0.001).all()


def test_ks_per_variable_bonferroni_clips_at_one() -> None:
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"x": rng.standard_normal(500), "y": rng.standard_normal(500)})
    b = pd.DataFrame({"x": rng.standard_normal(500), "y": rng.standard_normal(500)})
    out = ks_per_variable(a, b, adjust="bonferroni")
    assert (out["p_adj"] <= 1.0).all()


def test_mwu_per_variable_runs() -> None:
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"x": rng.standard_normal(200)})
    b = pd.DataFrame({"x": rng.standard_normal(200) + 0.5})
    out = mwu_per_variable(a, b)
    assert "U" in out.columns


def test_frobenius_distance_zero_for_identical_matrices() -> None:
    R = np.eye(3)
    out = frobenius_distance(R, R)
    assert out["frobenius"] == 0.0
    assert out["max_abs_dev"] == 0.0


def test_correlation_residual_difference() -> None:
    a = np.eye(2)
    b = np.array([[1.0, 0.1], [0.1, 1.0]])
    res = correlation_residual(a, b)
    assert res[0, 1] == 0.1


def test_correlation_residual_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        correlation_residual(np.eye(2), np.eye(3))


def test_empirical_correlation_returns_square_matrix() -> None:
    from caspian_fish_quality.validation.joint import empirical_correlation

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.standard_normal(100),
            "b": rng.standard_normal(100),
            "tag": ["x"] * 100,
        }
    )
    corr = empirical_correlation(df)
    assert corr.shape == (2, 2)
    assert np.allclose(np.diag(corr), 1.0)


def test_mwu_per_variable_bonferroni() -> None:
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"x": rng.standard_normal(50), "y": rng.standard_normal(50)})
    b = pd.DataFrame({"x": rng.standard_normal(50), "y": rng.standard_normal(50)})
    out = mwu_per_variable(a, b, adjust="bonferroni")
    assert "p_adj" in out.columns
    assert (out["p_adj"] <= 1.0).all()


def test_wasserstein_per_variable_unnormalized() -> None:
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"x": rng.standard_normal(500)})
    b = pd.DataFrame({"x": rng.standard_normal(500) + 1.5})
    out = wasserstein_per_variable(a, b, normalize=False)
    assert "W1" in out.columns
    assert (out["W1"] > 0).all()


def test_wasserstein_per_variable_close_for_same_dist() -> None:
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"x": rng.standard_normal(2000)})
    b = pd.DataFrame({"x": rng.standard_normal(2000)})
    out = wasserstein_per_variable(a, b)
    assert (out["W1_normalized"] < 0.2).all()


def test_tstr_protocol_runs(phd_df: pd.DataFrame) -> None:
    rng = np.random.default_rng(0)
    feats = ["Water_Temp_C", "Water_pH", "Water_O2_mgL"]
    real = phd_df[[*feats, "Protein_perc"]].copy()
    syn = real.copy()
    syn[feats] = real[feats].to_numpy() + rng.standard_normal(real[feats].shape) * 0.1
    out = tstr_protocol(real, syn, "Protein_perc", feats)
    assert "TRTR" in out.columns
    assert "TSTR" in out.columns
    assert "gap" in out.columns
