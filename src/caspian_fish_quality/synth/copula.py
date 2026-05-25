"""Gaussian copula sampling for correlated multivariate synthetic data.

This module implements the NORTA (NORmal-To-Anything) construction
(Cario & Nelson, 1997) grounded in Sklar's theorem (Sklar, 1959; Nelsen,
2006). A target Pearson correlation matrix R is Cholesky-decomposed,
multivariate-normal samples are drawn via Z = L * eps with eps ~ N(0, I),
each Z_j is mapped through Phi to U_j in (0, 1), and the inverse of the
target marginal F_j^{-1} is applied to obtain X_j with the prescribed
marginal and (approximately) the desired correlation. As shown by Ghosh &
Henderson (2003), at moderate dimensionality some target correlation
matrices are NORTA-infeasible; we project R onto the nearest positive-
definite matrix when necessary.

References
----------
Cario, M. C., & Nelson, B. L. (1997). *Modeling and Generating Random
    Vectors with Arbitrary Marginal Distributions and Correlation Matrix*
    [Tech. Rep.]. Northwestern University.
Ghosh, S., & Henderson, S. G. (2003). Behavior of the NORTA method for
    correlated random vector generation as the dimension increases.
    *ACM Transactions on Modeling and Computer Simulation*, 13(3),
    276-294. https://doi.org/10.1145/937332.937336
Higham, N. J. (2002). Computing the nearest correlation matrix --
    a problem from finance. *IMA J. Numer. Anal.*, 22(3), 329-343.
Nelsen, R. B. (2006). *An Introduction to Copulas* (2nd ed.). Springer.
    https://doi.org/10.1007/0-387-28678-0
Sklar, A. (1959). Fonctions de repartition a n dimensions et leurs
    marges. *Pub. Inst. Stat. Univ. Paris*, 8, 229-231.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.linalg import cholesky
from scipy.stats import norm, truncnorm


def ensure_psd(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project ``R`` onto the nearest positive-semidefinite correlation matrix.

    Uses ``statsmodels.stats.correlation_tools.corr_nearest`` (Higham, 2002)
    when available; otherwise falls back to direct eigenvalue clipping.

    Parameters
    ----------
    R : numpy.typing.NDArray[numpy.float64]
        Possibly-indefinite correlation matrix.

    Returns
    -------
    numpy.typing.NDArray[numpy.float64]
        A PSD correlation matrix with unit diagonal.
    """
    if np.min(np.linalg.eigvalsh(R)) >= 1e-10:
        return R
    try:
        from statsmodels.stats.correlation_tools import corr_nearest

        return cast(NDArray[np.float64], corr_nearest(R, threshold=1e-10))
    except ImportError:
        ev = np.maximum(np.linalg.eigvalsh(R), 1e-8)
        eigvecs = np.linalg.eigh(R)[1]
        repaired = eigvecs @ np.diag(ev) @ eigvecs.T
        d = np.sqrt(np.diag(repaired))
        repaired /= np.outer(d, d)
        np.fill_diagonal(repaired, 1.0)
        return cast(NDArray[np.float64], repaired)


def _normalize_name(s: str) -> str:
    """Lowercase and strip non-alphanumeric (except ``:``) for fuzzy lookup."""
    import re

    return re.sub(r"[^a-z0-9:]", "", s.lower())


def build_corr(
    features: Sequence[str], spec: Mapping[tuple[str, str], float]
) -> NDArray[np.float64]:
    """Build a correlation matrix from a sparse ``(feature_a, feature_b) -> r``.

    Names are matched case-insensitively after normalisation, with substring
    fallback to handle minor wording differences across source tables.
    The result is forced PSD via :func:`ensure_psd`.

    Parameters
    ----------
    features : sequence of str
        Ordered feature names; the matrix is ``len(features) x len(features)``.
    spec : Mapping
        Mapping from ``(name_a, name_b)`` to Pearson correlation. Diagonal
        and missing entries default to ``1.0`` and ``0.0`` respectively.

    Returns
    -------
    numpy.typing.NDArray[numpy.float64]
        PSD correlation matrix.
    """
    n = len(features)
    R = np.eye(n, dtype=np.float64)
    norm_idx = {_normalize_name(f): i for i, f in enumerate(features)}

    for (a, b), r in spec.items():
        na, nb = _normalize_name(a), _normalize_name(b)
        ia: int | None = norm_idx.get(na)
        ib: int | None = norm_idx.get(nb)
        if ia is None:
            for nf, idx in norm_idx.items():
                if na in nf or nf in na:
                    ia = idx
                    break
        if ib is None:
            for nf, idx in norm_idx.items():
                if nb in nf or nf in nb:
                    ib = idx
                    break
        if ia is not None and ib is not None and ia != ib:
            R[ia, ib] = r
            R[ib, ia] = r

    return ensure_psd(R)


def gaussian_copula_sample(
    means: Sequence[float],
    sds: Sequence[float],
    los: Sequence[float],
    his: Sequence[float],
    corr_matrix: NDArray[np.float64],
    n: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Sample ``n`` rows from a Gaussian copula with truncated-normal margins.

    Pipeline: correlated MVN -> uniform via Phi -> truncated-normal via the
    column-wise inverse CDF (Sklar 1959; Cario & Nelson 1997).

    Parameters
    ----------
    means, sds, los, his : sequences of float
        Per-column marginal parameters; must all share length ``k``.
    corr_matrix : numpy.typing.NDArray[numpy.float64]
        Target ``k x k`` Pearson correlation matrix (will be PSD-projected).
    n : int
        Number of rows to sample.
    rng : numpy.random.Generator

    Returns
    -------
    numpy.typing.NDArray[numpy.float64]
        Array of shape ``(n, k)``.
    """
    means_a = np.asarray(means, dtype=np.float64)
    sds_a = np.asarray(sds, dtype=np.float64)
    los_a = np.asarray(los, dtype=np.float64)
    his_a = np.asarray(his, dtype=np.float64)
    k = len(means_a)
    if not (len(sds_a) == len(los_a) == len(his_a) == k):
        raise ValueError("means, sds, los, his must share the same length")

    psd = ensure_psd(corr_matrix)
    L = np.linalg.cholesky(psd)
    z = rng.standard_normal((n, k)) @ L.T

    out = np.zeros_like(z)
    for j in range(k):
        u = norm.cdf(z[:, j])
        lo, hi = los_a[j], his_a[j]
        mean, sd = means_a[j], sds_a[j]

        if sd <= 0:
            out[:, j] = mean
            continue
        if mean < lo or mean > hi:
            mean = (lo + hi) / 2
        a = (lo - mean) / sd
        b = (hi - mean) / sd
        if a >= b:
            out[:, j] = mean
            continue
        out[:, j] = truncnorm.ppf(np.clip(u, 1e-10, 1 - 1e-10), a, b, loc=mean, scale=sd)

    return cast(NDArray[np.float64], out)


def copula_generate(
    features: Sequence[str],
    means: Sequence[float] | NDArray[np.float64],
    sds: Sequence[float] | NDArray[np.float64],
    mins: Sequence[float] | NDArray[np.float64],
    maxs: Sequence[float] | NDArray[np.float64],
    R: NDArray[np.float64],
    n_samples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """High-level ``DataFrame``-returning wrapper around :func:`gaussian_copula_sample`.

    Parameters
    ----------
    features : sequence of str
    means, sds, mins, maxs : sequences of float
        Per-feature marginal parameters.
    R : numpy.typing.NDArray[numpy.float64]
        Target correlation matrix; will be PSD-projected.
    n_samples : int, default 1000
    seed : int, default 42

    Returns
    -------
    pandas.DataFrame
        Synthetic frame with one column per ``feature``.
    """
    rng = np.random.default_rng(seed)
    means_a = np.asarray(means, dtype=np.float64)
    sds_a = np.asarray(sds, dtype=np.float64)
    mins_a = np.asarray(mins, dtype=np.float64)
    maxs_a = np.asarray(maxs, dtype=np.float64)
    k = len(features)

    psd = ensure_psd(R)
    L = cholesky(psd, lower=True)
    Z = L @ rng.standard_normal((k, n_samples))
    U = norm.cdf(Z)
    X = np.zeros((n_samples, k), dtype=np.float64)

    for i in range(k):
        if sds_a[i] < 1e-12:
            X[:, i] = means_a[i]
            continue
        a = (mins_a[i] - means_a[i]) / sds_a[i]
        b = (maxs_a[i] - means_a[i]) / sds_a[i]
        if a >= b:
            X[:, i] = means_a[i]
            continue
        X[:, i] = truncnorm.ppf(
            np.clip(U[i], 1e-10, 1 - 1e-10),
            a,
            b,
            loc=means_a[i],
            scale=sds_a[i],
        )

    return pd.DataFrame(X, columns=list(features))


__all__ = [
    "build_corr",
    "copula_generate",
    "ensure_psd",
    "gaussian_copula_sample",
]
