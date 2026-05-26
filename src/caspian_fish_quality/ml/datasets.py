"""End-to-end dataset orchestrators ported from the notebook.

Three responsibilities:

1. Generate the v2 PHD water-quality -> fish-quality dataset with built-in
   biological correlations (:func:`generate_phd_dataset`).
2. Merge the PHD dataset with synthetic static and storage frames
   (:func:`prepare_full_dataset`).
3. Run the leakage-free experiment grid (:func:`run_experiment`,
   :func:`run_all`).

Mechanistic directions in :func:`generate_phd_dataset` (separate from copula CSV
marginals; see ``docs/LITERATURE_SOURCES.md`` → ``phd_mechanistic``):

* Temperature → lipid (+) — Hallier et al. (2007), https://doi.org/10.1002/jsfa.2779
* Dissolved oxygen → protein (+) — Huss (1995), FAO Fish. Tech. Paper 348
* Feed type → lipid (+) — Jankowska et al. (2006), https://doi.org/10.1007/s00217-006-0348-9
* AG vs RG water priors — Simeanu et al. (2022), https://doi.org/10.3390/agriculture12122144
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from caspian_fish_quality.ml.cv import Task, make_cv
from caspian_fish_quality.ml.metrics import (
    classification_metrics,
    regression_metrics,
)
from caspian_fish_quality.ml.models import (
    classification_pipelines,
    regression_pipelines,
)


def generate_phd_dataset(n_per_group: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate the water-quality / fish-composition synthetic PHD dataset.

    Parameters
    ----------
    n_per_group : int, default 1000
    seed : int, default 42

    Returns
    -------
    pandas.DataFrame
        ``n_per_group * 2`` rows; AG (aquaculture) and RG (riverine) groups.
    """
    rng = np.random.default_rng(seed)

    def _gen(group: str, n: int) -> pd.DataFrame:
        ag = group == "AG"
        temp = np.clip(rng.normal(22 if ag else 12, 3, n), 8, 30)
        ph = np.clip(rng.normal(7.5 if ag else 7.3, 0.15, n), 6.8, 8.2)
        o2 = np.clip(rng.normal(6.5 if ag else 9.0, 1.2, n), 3.5, 12)
        cl = np.clip(rng.normal(95 if ag else 75, 8, n), 50, 120)
        no2 = np.clip(rng.normal(0.14 if ag else 0.08, 0.03, n), 0.01, 0.25)
        no3 = np.clip(rng.normal(1.8 if ag else 1.0, 0.4, n), 0.1, 3.0)
        nh4 = np.clip(rng.normal(0.22 if ag else 0.10, 0.06, n), 0.02, 0.40)
        po4 = np.clip(rng.normal(0.14 if ag else 0.06, 0.04, n), 0.001, 0.25)
        feed = np.ones(n) if ag else np.zeros(n)

        mass = np.clip(
            1500 + 12 * temp + 15 * o2 - 2.5 * cl + 300 * feed + rng.normal(0, 120, n),
            1200,
            2400,
        )
        prot = np.clip(
            15.0
            + 0.35 * o2
            - 0.12 * temp
            - 8.0 * nh4
            + 5.0 * no2
            + 0.5 * feed
            + rng.normal(0, 1.5, n),
            10,
            25,
        )
        lip = np.clip(
            1.5
            + 0.06 * temp
            + 0.8 * feed
            - 0.08 * o2
            + 3.0 * po4
            - 1.5 * no2
            + rng.normal(0, 0.3, n),
            1.0,
            5.5,
        )

        return pd.DataFrame(
            {
                "Group": group,
                "Water_Temp_C": temp.round(2),
                "Water_pH": ph.round(3),
                "Water_O2_mgL": o2.round(3),
                "Chlorides_mgL": cl.round(2),
                "Nitrites_mgL": no2.round(4),
                "Nitrates_mgL": no3.round(3),
                "Ammonium_mgL": nh4.round(4),
                "Phosphates_mgL": po4.round(4),
                "Feed_Type": feed.astype(int),
                "Bodymass_g": mass.round(2),
                "Protein_perc": prot.round(3),
                "Lipids_perc": lip.round(3),
            }
        )

    df = pd.concat([_gen("AG", n_per_group), _gen("RG", n_per_group)], ignore_index=True)
    df["sample_id"] = df.groupby("Group").cumcount().astype(int)
    df.insert(0, "Fish_ID", range(1, len(df) + 1))
    return df


def _ensure_sample_id(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = df.copy()
    if "sample_id" not in out.columns:
        out["sample_id"] = out.groupby(group_col).cumcount().astype(int)
    return out


def prepare_full_dataset(
    phd_df: pd.DataFrame,
    static_df: pd.DataFrame,
    storage_df: pd.DataFrame,
    *,
    null_thresh: float = 0.3,
) -> pd.DataFrame:
    """Merge PHD water-quality rows with copula static/storage traits by group + sample_id.

    Parameters
    ----------
    phd_df : pandas.DataFrame
        Output of :func:`generate_phd_dataset`.
    static_df : pandas.DataFrame
        Output of :func:`caspian_fish_quality.merge.merge_static`.
    storage_df : pandas.DataFrame
        Output of :func:`caspian_fish_quality.merge.merge_storage`.
    null_thresh : float, default 0.3
        Columns with > ``null_thresh`` fraction of nulls are dropped.

    Returns
    -------
    pandas.DataFrame
        One row per virtual individual; ``Group`` is AG/RG (same codes as ``group`` in merges).
    """
    phd = _ensure_sample_id(phd_df, "Group")
    static = _ensure_sample_id(static_df, "group")
    storage = _ensure_sample_id(storage_df, "group")

    static_renamed = static.rename(columns={"group": "Group"})
    storage_renamed = storage.rename(columns={"group": "Group"})

    static_cols = [c for c in static_renamed.columns if c not in ("Group", "sample_id")]
    storage_cols = [c for c in storage_renamed.columns if c not in ("Group", "sample_id")]

    full = phd.merge(
        static_renamed[["Group", "sample_id", *static_cols]],
        on=["Group", "sample_id"],
        how="inner",
        validate="one_to_one",
    )
    full = full.merge(
        storage_renamed[["Group", "sample_id", *storage_cols]],
        on=["Group", "sample_id"],
        how="left",
        validate="one_to_one",
    )
    full = full.drop(columns=[c for c in full.columns if full[c].isna().mean() > null_thresh])
    full = full.fillna(full.median(numeric_only=True))
    le = LabelEncoder()
    full["Group_enc"] = le.fit_transform(full["Group"])
    return cast("pd.DataFrame", full)


def _scoring_for(task: Task) -> str:
    return "r2" if task == "reg" else "accuracy"


def run_experiment(
    full_df: pd.DataFrame,
    name: str,
    features: Iterable[str],
    target: str,
    task: Task = "reg",
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train every model in the catalogue and return (results, importance).

    Parameters
    ----------
    full_df : pandas.DataFrame
        Output of :func:`prepare_full_dataset`.
    name : str
        Display name for the experiment row.
    features : iterable of str
        Feature column names; missing columns are silently dropped.
    target : str
        Target column name.
    task : {'reg', 'cls'}, default 'reg'
    test_size : float, default 0.2
    random_state : int, default 42
    n_splits : int, default 5

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Results table and Random-Forest top-10 feature importance.
    """
    feats = [f for f in features if f in full_df.columns]
    x = full_df[feats].to_numpy()
    y = full_df[target].to_numpy()

    stratify = full_df["Group_enc"] if task == "cls" else None
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    cv = make_cv(task, n_splits=n_splits, seed=random_state)
    pipes: dict[str, Pipeline] = (
        regression_pipelines(random_state=random_state)
        if task == "reg"
        else classification_pipelines(random_state=random_state)
    )
    rows: list[dict[str, float | str]] = []

    for mname, pipe in pipes.items():
        pipe.fit(x_tr, y_tr)
        yp = pipe.predict(x_te)
        cvs = cross_val_score(pipe, x, y, cv=cv, scoring=_scoring_for(task))
        score_label = "CV_R2" if task == "reg" else "CV_Acc"

        if task == "reg":
            row: dict[str, float | str] = {
                "experiment": name,
                "model": mname,
                **regression_metrics(y_te, yp),
                f"{score_label}_mean": round(float(cvs.mean()), 4),
                f"{score_label}_std": round(float(cvs.std()), 4),
            }
        else:
            row = {
                "experiment": name,
                "model": mname,
                **classification_metrics(y_te, yp),
                f"{score_label}_mean": round(float(cvs.mean()), 4),
                f"{score_label}_std": round(float(cvs.std()), 4),
            }
        rows.append(row)

    rf = pipes["Random Forest"]
    rf.fit(x, y)
    rf_model = rf.named_steps["model"]
    importances: np.ndarray = getattr(rf_model, "feature_importances_", np.zeros(len(feats)))
    imp = (
        pd.DataFrame({"experiment": name, "feature": feats, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(10)
    )

    return pd.DataFrame(rows), imp


_WATER_COLS = (
    "Water_Temp_C",
    "Water_pH",
    "Water_O2_mgL",
    "Chlorides_mgL",
    "Nitrites_mgL",
    "Nitrates_mgL",
    "Ammonium_mgL",
    "Phosphates_mgL",
)


def _experiment_grid(full_df: pd.DataFrame) -> list[tuple[str, list[str], str, Task]]:
    water = list(_WATER_COLS)
    fa = [c for c in full_df.columns if c.startswith("fa_")]
    allx = (
        water
        + ["Group_enc", "Feed_Type"]
        + [
            c
            for c in full_df.columns
            if c.startswith(("bio_", "idx_", "cut_", "fa_", "stor_", "chem_"))
        ]
    )
    return [
        ("R1_Water_Protein", [*water, "Group_enc", "Feed_Type"], "Protein_perc", "reg"),
        ("R2_Water_Lipids", [*water, "Group_enc", "Feed_Type"], "Lipids_perc", "reg"),
        ("R3_Water_Bodymass", [*water, "Group_enc", "Feed_Type"], "Bodymass_g", "reg"),
        ("R4_All_Protein", allx, "Protein_perc", "reg"),
        ("R5_All_Lipids", allx, "Lipids_perc", "reg"),
        ("C1_Water_Group", [*water, "Feed_Type"], "Group_enc", "cls"),
        ("C2_FA_Group", fa, "Group_enc", "cls"),
        ("C3_All_Group", allx, "Group_enc", "cls"),
    ]


def run_all(
    full_df: pd.DataFrame,
    *,
    verbose: bool = False,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the eight pre-registered experiments.

    Parameters
    ----------
    full_df : pandas.DataFrame
    verbose : bool, default False
    random_state : int, default 42

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
    """
    all_res: list[pd.DataFrame] = []
    all_imp: list[pd.DataFrame] = []

    for name, feats, target, task in _experiment_grid(full_df):
        if verbose:
            print(f"\n{'=' * 55}\n{name}")
        if not feats:
            continue
        res, imp = run_experiment(full_df, name, feats, target, task, random_state=random_state)
        all_res.append(res)
        all_imp.append(imp)

    return (
        pd.concat(all_res, ignore_index=True) if all_res else pd.DataFrame(),
        pd.concat(all_imp, ignore_index=True) if all_imp else pd.DataFrame(),
    )


__all__ = [
    "generate_phd_dataset",
    "prepare_full_dataset",
    "run_all",
    "run_experiment",
]
