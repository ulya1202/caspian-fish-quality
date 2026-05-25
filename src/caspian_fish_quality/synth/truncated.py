"""Truncated normal sampling for biologically-bounded variables.

Used wherever a quantity must be non-negative (mass, length) or bounded in
a physiologically plausible interval (lipid %, fatty-acid fraction).
Implementation follows the optimal exponential-proposal accept-reject
sampler of Robert (1995); Burkardt (2014) is the practical formulary.
Johnson, Kotz, & Balakrishnan (1994) provide the canonical analytical
properties (moments, mode, MLE).

References
----------
Burkardt, J. (2014). The truncated normal distribution [Tech. note].
    Florida State University.
Johnson, N. L., Kotz, S., & Balakrishnan, N. (1994). *Continuous
    Univariate Distributions, Volume 1* (2nd ed.). Wiley.
Robert, C. P. (1995). Simulation of truncated normal variables.
    *Statistics and Computing*, 5(2), 121-125.
    https://doi.org/10.1007/BF00143942
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.stats import truncnorm


def sample_truncated(
    mean: float,
    sd: float,
    lo: float,
    hi: float,
    n: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Draw ``n`` independent samples from a truncated normal distribution.

    The mean is auto-clamped to ``(lo, hi)`` if it falls outside, ``NaN``
    bounds are replaced by ``mean ± 6 * sd``, and ``sd <= 0`` returns a
    constant array.

    Parameters
    ----------
    mean, sd : float
        Target mean and standard deviation of the un-truncated distribution.
    lo, hi : float
        Inclusive truncation bounds. Either may be ``NaN`` to indicate that
        a default (``mean ± 6 * sd``) should be used.
    n : int
        Sample size (must be >= 0).
    rng : numpy.random.Generator
        Random generator used to seed ``scipy.stats.truncnorm``.

    Returns
    -------
    numpy.typing.NDArray[numpy.float64]
        Length-``n`` array of samples in ``[lo, hi]``.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if sd <= 0:
        return np.full(n, mean, dtype=np.float64)

    lo_is_nan = math.isnan(lo)
    hi_is_nan = math.isnan(hi)

    if not (lo_is_nan or hi_is_nan) and (mean < lo or mean > hi):
        mean = (lo + hi) / 2

    if lo_is_nan:
        lo = max(0.0, mean - 6.0 * sd)
    if hi_is_nan:
        hi = mean + 6.0 * sd

    a = (lo - mean) / sd
    b = (hi - mean) / sd
    if not math.isfinite(a) or not math.isfinite(b) or a >= b:
        return np.full(n, mean, dtype=np.float64)

    seed_int = int(rng.integers(1, 2**31 - 1))
    samples = truncnorm.rvs(a, b, loc=mean, scale=sd, size=n, random_state=seed_int)
    return np.asarray(samples, dtype=np.float64)


__all__ = ["sample_truncated"]
