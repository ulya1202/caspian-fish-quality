"""Synthetic-data validation: marginals, joint, Wasserstein, TSTR."""

from __future__ import annotations

from caspian_fish_quality.validation.joint import (
    correlation_residual,
    frobenius_distance,
)
from caspian_fish_quality.validation.marginal import (
    ks_per_variable,
    mwu_per_variable,
)
from caspian_fish_quality.validation.tstr import tstr_protocol
from caspian_fish_quality.validation.wasserstein import wasserstein_per_variable

__all__ = [
    "correlation_residual",
    "frobenius_distance",
    "ks_per_variable",
    "mwu_per_variable",
    "tstr_protocol",
    "wasserstein_per_variable",
]
