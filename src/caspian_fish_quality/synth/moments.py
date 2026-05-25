"""SEM <-> SD conversion utilities for literature-derived inputs.

When peer-reviewed papers report mean ± SEM rather than mean ± SD, we
recover SD = SEM * sqrt(n) following Altman & Bland (2005). This is
critical because copula sampling of marginals requires SD; using SEM in
its place underestimates variance by a factor of sqrt(n).

References
----------
Altman, D. G., & Bland, J. M. (2005). Standard deviations and standard
    errors. *BMJ*, 331(7521), 903. https://doi.org/10.1136/bmj.331.7521.903
"""

from __future__ import annotations

import math


def sem_to_sd(sem: float, n: int) -> float:
    """Recover SD from SEM and sample size.

    Parameters
    ----------
    sem : float
        Standard error of the mean (must be non-negative).
    n : int
        Sample size used to compute ``sem`` (must be >= 1).

    Returns
    -------
    float
        ``sem * sqrt(n)``.

    Raises
    ------
    ValueError
        If ``sem < 0`` or ``n < 1``.
    """
    if sem < 0:
        raise ValueError("sem must be non-negative")
    if n < 1:
        raise ValueError("n must be at least 1")
    return float(sem) * math.sqrt(n)


__all__ = ["sem_to_sd"]
