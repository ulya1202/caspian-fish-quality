from __future__ import annotations

from pathlib import Path

import pytest

from caspian_fish_quality.config import Settings, get_settings, literature_dir


def test_settings_defaults_are_dissertation_grade() -> None:
    s = Settings()
    assert s.seed == 42
    assert s.n_per_group == 1000


def test_settings_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CFQ_SEED", "7")
    monkeypatch.setenv("CFQ_N_PER_GROUP", "5")
    get_settings.cache_clear()
    s = get_settings()
    assert s.seed == 7
    assert s.n_per_group == 5
    get_settings.cache_clear()


def test_literature_dir_returns_existing_directory() -> None:
    path = literature_dir()
    assert isinstance(path, Path)
    assert path.is_dir()
    assert any(path.glob("data_*.csv"))


def test_settings_is_frozen() -> None:
    from pydantic import ValidationError

    s = Settings()
    with pytest.raises(ValidationError):
        s.seed = 99  # type: ignore[misc]
