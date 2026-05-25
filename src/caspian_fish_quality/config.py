"""Runtime configuration for caspian_fish_quality.

Pydantic v2 settings expose a single ``Settings`` instance with the master
random seed and the path to the bundled literature CSV directory. All
stochastic functions in the package accept a ``numpy.random.Generator`` so
that callers can override the default seed deterministically.

References
----------
Wilkinson, M. D., Dumontier, M., Aalbersberg, I. J., et al. (2016). The FAIR
    Guiding Principles for scientific data management and stewardship.
    *Scientific Data*, 3, 160018. https://doi.org/10.1038/sdata.2016.18
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Top-level package settings.

    Parameters are read from environment variables prefixed with
    ``CFQ_`` (e.g. ``CFQ_SEED=123``). Defaults are dissertation-grade and
    reproduce the published synthetic dataset.

    Attributes
    ----------
    seed : int
        Master random seed for ``numpy.random.default_rng``.
    n_per_group : int
        Default sample size per group for ``generate_all_synthetic``.
    """

    model_config = SettingsConfigDict(env_prefix="CFQ_", frozen=True)

    seed: int = Field(default=42, ge=0, description="Master PCG64 seed.")
    n_per_group: int = Field(default=1000, ge=1, description="Default n per group.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a memoised :class:`Settings` instance."""
    return Settings()


def literature_dir() -> Path:
    """Return the absolute path to the bundled literature CSV directory."""
    with resources.as_file(
        resources.files("caspian_fish_quality").joinpath("data").joinpath("literature")
    ) as path:
        return Path(path)
