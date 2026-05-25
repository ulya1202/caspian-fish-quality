from __future__ import annotations

import pandas as pd

from caspian_fish_quality.ml.datasets import (
    generate_phd_dataset,
    prepare_full_dataset,
    run_all,
    run_experiment,
)


def test_generate_phd_dataset_balanced_groups() -> None:
    df = generate_phd_dataset(n_per_group=50, seed=0)
    assert len(df) == 100
    assert set(df["Group"].unique()) == {"AG", "RG"}
    assert {"Bodymass_g", "Protein_perc", "Lipids_perc", "Water_O2_mgL"} <= set(df.columns)
    assert df["Bodymass_g"].between(1200, 2400).all()
    assert df["Protein_perc"].between(10, 25).all()


def test_prepare_full_dataset_encodes_group(
    phd_df: pd.DataFrame, static_df: pd.DataFrame, storage_df: pd.DataFrame
) -> None:
    full = prepare_full_dataset(phd_df, static_df, storage_df)
    assert "Group_enc" in full.columns
    assert full["Group_enc"].isin([0, 1]).all()
    assert not full.isna().any().any()


def test_run_experiment_smoke(
    phd_df: pd.DataFrame, static_df: pd.DataFrame, storage_df: pd.DataFrame
) -> None:
    full = prepare_full_dataset(phd_df, static_df, storage_df)
    res, imp = run_experiment(
        full,
        "T1",
        ["Water_Temp_C", "Water_O2_mgL", "Group_enc"],
        "Protein_perc",
        task="reg",
        n_splits=3,
    )
    assert {"experiment", "model", "R2", "RMSE", "MAE"} <= set(res.columns)
    assert (res["experiment"] == "T1").all()
    assert {"experiment", "feature", "importance"} <= set(imp.columns)
    assert len(imp) <= 10


def test_run_experiment_classification(
    phd_df: pd.DataFrame, static_df: pd.DataFrame, storage_df: pd.DataFrame
) -> None:
    full = prepare_full_dataset(phd_df, static_df, storage_df)
    res, _imp = run_experiment(
        full,
        "C1",
        ["Water_Temp_C", "Water_O2_mgL", "Feed_Type"],
        "Group_enc",
        task="cls",
        n_splits=3,
    )
    assert {"experiment", "model", "Accuracy", "F1"} <= set(res.columns)


def test_run_all_returns_results_and_importance(
    phd_df: pd.DataFrame, static_df: pd.DataFrame, storage_df: pd.DataFrame
) -> None:
    full = prepare_full_dataset(phd_df, static_df, storage_df)
    res, imp = run_all(full, random_state=0)
    assert not res.empty
    assert not imp.empty
    assert "experiment" in res.columns
