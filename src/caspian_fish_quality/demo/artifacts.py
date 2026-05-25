"""Save and load pre-trained models + tables for the defense demo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

_INFERENCE_TARGETS = ("Protein_perc", "Lipids_perc", "Bodymass_g")


@dataclass(frozen=True)
class DemoArtifacts:
    """Bundled inference models and precomputed dissertation tables."""

    manifest: dict[str, Any]
    models: dict[str, Pipeline]
    feat_cols: list[str]
    regression: pd.DataFrame
    transfer: pd.DataFrame
    marginal: pd.DataFrame
    table_37: pd.DataFrame


def default_demo_dir(repo_root: Path | None = None) -> Path:
    """Return ``demo_artifacts/`` at the repository root."""
    if repo_root is not None:
        return repo_root / "demo_artifacts"
    return Path(__file__).resolve().parents[3] / "demo_artifacts"


def save_demo_artifacts(
    out_dir: Path,
    *,
    manifest: dict[str, Any],
    models: dict[str, Pipeline],
    feat_cols: list[str],
    regression: pd.DataFrame,
    transfer: pd.DataFrame,
    marginal: pd.DataFrame,
    table_37: pd.DataFrame,
    ml_results: pd.DataFrame | None = None,
) -> None:
    """Write joblib models, CSV tables, and ``manifest.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "models"
    model_dir.mkdir(exist_ok=True)

    for target, pipe in models.items():
        joblib.dump(pipe, model_dir / f"ridge_{target}.joblib")

    manifest = {
        **manifest,
        "feat_cols": feat_cols,
        "inference_targets": list(models.keys()),
        "exported_at": datetime.now(UTC).isoformat(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    regression.to_csv(out_dir / "regression_cv_r2.csv", index=False)
    transfer.to_csv(out_dir / "transfer_zero_shot.csv", index=False)
    marginal.to_csv(out_dir / "marginal_relative_error.csv", index=False)
    table_37.to_csv(out_dir / "transfer_table_3_5_7.csv", index=False)
    if ml_results is not None:
        ml_results.to_csv(out_dir / "ml_results_all.csv", index=False)


def load_demo_artifacts(demo_dir: Path | None = None) -> DemoArtifacts:
    """Load artifacts produced by :func:`save_demo_artifacts`."""
    root = demo_dir or default_demo_dir()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Demo artifacts not found at {root}. "
            "Run: python scripts/export_demo_artifacts.py"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feat_cols = list(manifest["feat_cols"])
    models: dict[str, Pipeline] = {}
    for target in manifest.get("inference_targets", _INFERENCE_TARGETS):
        path = root / "models" / f"ridge_{target}.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Missing model weights: {path}")
        models[target] = joblib.load(path)

    return DemoArtifacts(
        manifest=manifest,
        models=models,
        feat_cols=feat_cols,
        regression=pd.read_csv(root / "regression_cv_r2.csv"),
        transfer=pd.read_csv(root / "transfer_zero_shot.csv"),
        marginal=pd.read_csv(root / "marginal_relative_error.csv"),
        table_37=pd.read_csv(root / "transfer_table_3_5_7.csv"),
    )
