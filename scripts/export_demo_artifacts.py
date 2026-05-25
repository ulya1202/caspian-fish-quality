#!/usr/bin/env python3
"""Train models once and export weights + tables for the Streamlit demo.

Writes committed artifacts under ``demo_artifacts/`` (inference-only demo).

Usage::

    python scripts/export_demo_artifacts.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

import caspian_fish_quality as cfq
from caspian_fish_quality.config import get_settings
from caspian_fish_quality.demo.artifacts import default_demo_dir, save_demo_artifacts
from caspian_fish_quality.ml.datasets import (
    generate_phd_dataset,
    prepare_full_dataset,
    run_all,
)
from caspian_fish_quality.merge import merge_static, merge_storage
from caspian_fish_quality.synth.generators import generate_all_synthetic, load_default_df_dict
from caspian_fish_quality.transfer import run_transfer_test
from caspian_fish_quality.transfer.sturgeon_eval import (
    TARGET_LABELS,
    fit_water_to_flesh_models,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = default_demo_dir(ROOT)

# Reuse dissertation table builders from reproduce script
sys.path.insert(0, str(ROOT / "scripts"))
import reproduce_section_3_5_3 as repro  # noqa: E402


def main() -> int:
    t0 = time.time()
    settings = get_settings()
    seed = settings.seed
    n_per_group = settings.n_per_group

    print("=" * 70)
    print("Export demo artifacts (train once → inference-only Streamlit)")
    print(f"  seed={seed}  n_per_group={n_per_group}  out={OUT}")
    print("=" * 70)

    df_dict = load_default_df_dict()
    synthetic_dict = generate_all_synthetic(
        df_dict, n_per_group=n_per_group, seed=seed, verbose=True
    )
    static_df = merge_static(synthetic_dict)
    storage_df = merge_storage(synthetic_dict)
    phd_df = generate_phd_dataset(n_per_group=n_per_group, seed=seed)
    print(f"  PHD dataset: {phd_df.shape[0]} rows")

    marginal = repro.marginal_relative_errors(df_dict, synthetic_dict)
    full = prepare_full_dataset(phd_df, static_df, storage_df)

    print("  Running 5-fold CV catalogue (run_all)…")
    results, _importance = run_all(full, random_state=seed)
    reg = results[results["experiment"].str.startswith("R")]
    table_36 = repro.build_table_36(reg)

    print("  Training Ridge inference models on full synthetic dataset…")
    trained, feat_cols = fit_water_to_flesh_models(full, models=("Ridge",), random_state=seed)
    inference_models: dict[str, object] = {}
    for label, (col, _attr) in TARGET_LABELS.items():
        key = (label, "Ridge")
        if key in trained:
            inference_models[col] = trained[key]

    print("  Zero-shot transfer evaluation…")
    transfer = run_transfer_test(
        phd_df, static_df, storage_df, models=("Ridge",), random_state=seed
    )
    table_37 = repro.build_table_37(transfer)

    manifest = {
        "package_version": cfq.__version__,
        "seed": seed,
        "n_per_group": n_per_group,
        "n_rows_phd": int(len(phd_df)),
        "n_rows_full": int(len(full)),
        "note": "Pre-trained on synthetic S. glanis data; Streamlit uses inference only.",
    }

    save_demo_artifacts(
        OUT,
        manifest=manifest,
        models=inference_models,
        feat_cols=feat_cols,
        regression=table_36,
        transfer=transfer,
        marginal=marginal,
        table_37=table_37,
        ml_results=results,
    )

    elapsed = time.time() - t0
    print(f"\nSaved to {OUT}/")
    print(f"  models/: {len(inference_models)} Ridge pipelines (joblib)")
    print(f"  regression_cv_r2.csv ({len(table_36)} rows)")
    print(f"  transfer_zero_shot.csv ({len(transfer)} rows)")
    print(f"  marginal_relative_error.csv ({len(marginal)} rows)")
    print(f"Done in {elapsed:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
