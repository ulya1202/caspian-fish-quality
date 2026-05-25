from __future__ import annotations

import numpy as np
import pytest

from caspian_fish_quality.synth.copula import (
    build_corr,
    copula_generate,
    ensure_psd,
    gaussian_copula_sample,
)


def test_ensure_psd_idempotent_on_psd_matrix() -> None:
    R = np.array([[1.0, 0.5], [0.5, 1.0]])
    out = ensure_psd(R)
    assert np.allclose(np.linalg.eigvalsh(out).min(), 0.5, atol=1e-8)


def test_ensure_psd_repairs_indefinite_matrix() -> None:
    R = np.array([[1.0, 0.99, 0.99], [0.99, 1.0, -0.99], [0.99, -0.99, 1.0]])
    out = ensure_psd(R)
    assert np.linalg.eigvalsh(out).min() >= -1e-8


def test_build_corr_substring_match() -> None:
    feats = ["bodymass(g)", "totallength(cm)"]
    R = build_corr(feats, {("bodymass", "totallength"): 0.9})
    assert R[0, 1] == pytest.approx(0.9)
    assert R[1, 0] == pytest.approx(0.9)
    assert R[0, 0] == pytest.approx(1.0)


def test_gaussian_copula_sample_correlation_close_to_target() -> None:
    rng = np.random.default_rng(42)
    R = np.array([[1.0, 0.7], [0.7, 1.0]])
    out = gaussian_copula_sample(
        means=[0.0, 0.0],
        sds=[1.0, 1.0],
        los=[-10.0, -10.0],
        his=[10.0, 10.0],
        corr_matrix=R,
        n=20_000,
        rng=rng,
    )
    emp = float(np.corrcoef(out.T)[0, 1])
    assert abs(emp - 0.7) < 0.03


def test_copula_generate_returns_dataframe_with_correct_columns() -> None:
    df = copula_generate(
        ["a", "b", "c"],
        means=[1.0, 2.0, 3.0],
        sds=[0.1, 0.2, 0.3],
        mins=[0.0, 0.0, 0.0],
        maxs=[5.0, 5.0, 5.0],
        R=np.eye(3),
        n_samples=200,
        seed=7,
    )
    assert list(df.columns) == ["a", "b", "c"]
    assert len(df) == 200
    assert df["a"].between(0, 5).all()


def test_gaussian_copula_sample_input_validation() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        gaussian_copula_sample(
            means=[0.0, 0.0],
            sds=[1.0],
            los=[-1.0, -1.0],
            his=[1.0, 1.0],
            corr_matrix=np.eye(2),
            n=10,
            rng=rng,
        )


def test_build_corr_unknown_features_yield_identity() -> None:
    feats = ["a", "b"]
    R = build_corr(feats, {("c", "d"): 0.9})
    assert np.allclose(R, np.eye(2))


def test_gaussian_copula_sample_zero_sd_yields_constant() -> None:
    rng = np.random.default_rng(0)
    out = gaussian_copula_sample(
        means=[5.0, 1.0],
        sds=[0.0, 1.0],
        los=[0.0, -10.0],
        his=[10.0, 10.0],
        corr_matrix=np.eye(2),
        n=100,
        rng=rng,
    )
    assert np.all(out[:, 0] == 5.0)
