#!/usr/bin/env python3
"""Reproduce dissertation section 3.5.3 tables from the canonical pipeline.

Writes CSV summaries and ``tables_az.md`` under ``results/section_3_5_3/``.

Usage::

    python scripts/reproduce_section_3_5_3.py
"""

from __future__ import annotations

import hashlib
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import caspian_fish_quality as cfq
from caspian_fish_quality.config import get_settings
from caspian_fish_quality.ml.datasets import (
    generate_phd_dataset,
    prepare_full_dataset,
    run_all,
)
from caspian_fish_quality.merge import merge_static, merge_storage
from caspian_fish_quality.synth.generators import (
    generate_all_synthetic,
    load_default_df_dict,
    parse_type_a,
    parse_type_b,
    parse_type_c,
)
from caspian_fish_quality.transfer import run_transfer_test

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "section_3_5_3"

EXPERIMENT_ALIASES: dict[str, str] = {
    "R1_Water_Protein": "R1_Water→Protein",
    "R2_Water_Lipids": "R2_Water→Lipids",
    "R3_Water_Bodymass": "R3_Water→Bodymass",
    "R4_All_Protein": "R4_All→Protein",
    "R5_All_Lipids": "R5_All→Lipids",
    "C1_Water_Group": "C1_Water→Group",
    "C2_FA_Group": "C2_FA→Group",
    "C3_All_Group": "C3_All→Group",
}

TABLE_36_MODELS = ("Ridge", "Random Forest", "LightGBM", "XGBoost")
TABLE_36_EXPERIMENTS = {
    "Lipid (%)": "R2_Water_Lipids",
    "Body mass (g)": "R3_Water_Bodymass",
    "Protein (%)": "R1_Water_Protein",
}

DISSERTATION_DRAFT_R2: dict[str, dict[str, float]] = {
    "Lipid (%)": {
        "Ridge": 0.91,
        "Random Forest": 0.90,
        "LightGBM": 0.90,
        "XGBoost": 0.89,
    },
    "Body mass (g)": {
        "Ridge": 0.68,
        "Random Forest": 0.67,
        "LightGBM": 0.66,
        "XGBoost": 0.65,
    },
    "Protein (%)": {
        "Ridge": 0.39,
        "Random Forest": 0.37,
        "LightGBM": 0.38,
        "XGBoost": 0.36,
    },
}


def _normalize_feature(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def marginal_relative_errors(
    df_dict: dict[int, pd.DataFrame],
    synthetic_dict: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Compare literature target means vs synthetic sample means."""
    rows: list[dict[str, object]] = []

    parsers = {
        "A": parse_type_a,
        "B": parse_type_b,
        "C": parse_type_c,
    }

    for key, lit_df in df_dict.items():
        syn_df = synthetic_dict.get(key)
        if syn_df is None:
            continue
        col0 = str(lit_df.columns[0]).lower()
        if "fatty" in col0:
            ptype = "C"
        elif "storage" in col0:
            ptype = "B"
        else:
            ptype = "A"
        parsed = parsers[ptype](lit_df)

        for grp, priors in parsed.items():
            syn_g = syn_df[syn_df["group"] == grp]
            if syn_g.empty:
                continue
            for _, prow in priors.iterrows():
                feat = str(prow["feature"])
                lit_mean = float(prow["mean"])
                if lit_mean == 0:
                    continue
                norm = _normalize_feature(feat)
                syn_col = None
                for c in syn_g.columns:
                    if c == "group":
                        continue
                    if _normalize_feature(str(c)) == norm:
                        syn_col = c
                        break
                if syn_col is None:
                    continue
                syn_mean = float(syn_g[syn_col].mean())
                rel_pct = abs(syn_mean - lit_mean) / abs(lit_mean) * 100.0
                rows.append(
                    {
                        "table": key,
                        "group": grp,
                        "feature": feat,
                        "literature_mean": round(lit_mean, 4),
                        "synthetic_mean": round(syn_mean, 4),
                        "relative_error_pct": round(rel_pct, 4),
                    }
                )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("relative_error_pct", ascending=False)
    return out


def build_table_36(reg_results: pd.DataFrame) -> pd.DataFrame:
    """Dissertation table 3.5.6: water-quality regressions, CV R²."""
    rows: list[dict[str, object]] = []
    for target_label, exp_id in TABLE_36_EXPERIMENTS.items():
        sub = reg_results[
            (reg_results["experiment"] == exp_id)
            & (reg_results["model"].isin(TABLE_36_MODELS))
        ]
        for model in TABLE_36_MODELS:
            mrow = sub[sub["model"] == model]
            if mrow.empty:
                continue
            r = mrow.iloc[0]
            rows.append(
                {
                    "target": target_label,
                    "experiment": EXPERIMENT_ALIASES.get(exp_id, exp_id),
                    "model": model,
                    "cv_r2_mean": r["CV_R2_mean"],
                    "cv_r2_std": r["CV_R2_std"],
                    "draft_r2": DISSERTATION_DRAFT_R2.get(target_label, {}).get(model),
                    "delta_vs_draft": (
                        round(float(r["CV_R2_mean"]) - DISSERTATION_DRAFT_R2[target_label][model], 4)
                        if target_label in DISSERTATION_DRAFT_R2
                        and model in DISSERTATION_DRAFT_R2[target_label]
                        else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_table_37(transfer: pd.DataFrame) -> pd.DataFrame:
    """Dissertation table 3.5.7: Ridge zero-shot protein and lipid."""
    sub = transfer[
        (transfer["model"] == "Ridge")
        & (transfer["target"].isin(["Protein (%)", "Lipids (%)"]))
    ].copy()
    sub = sub.rename(
        columns={
            "species": "Növ",
            "target": "Hədəf",
            "model": "Model",
            "predicted": "Proqnoz",
            "actual": "Faktiki",
            "error_pct": "Xəta %",
        }
    )
    return sub[
        [
            "Növ",
            "Hədəf",
            "Model",
            "Proqnoz",
            "Faktiki",
            "Xəta %",
            "flesh_composition_ref",
        ]
    ]


def transfer_mape_summary(transfer: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute percent error by target (Ridge only)."""
    sub = transfer[transfer["model"] == "Ridge"]
    rows: list[dict[str, object]] = []
    for target in ("Protein (%)", "Lipids (%)"):
        tsub = sub[sub["target"] == target]
        if tsub.empty:
            continue
        rows.append(
            {
                "target": target,
                "mean_abs_error_pct": round(float(tsub["error_pct"].abs().mean()), 1),
                "mape_mean": round(float(tsub["MAPE"].mean()) * 100, 1),
            }
        )
    return pd.DataFrame(rows)


def _fmt_az_num(x: float, decimals: int = 2) -> str:
    return f"{x:.{decimals}f}".replace(".", ",")


def write_tables_az(
    table_36: pd.DataFrame,
    table_37: pd.DataFrame,
    mape_summary: pd.DataFrame,
    marginal: pd.DataFrame,
    cls_results: pd.DataFrame,
    *,
    seed: int,
    n_per_group: int,
) -> None:
    """Write Azerbaijani markdown snippets for dissertation paste-in."""
    lines: list[str] = [
        "# Bölmə 3.5.3 — avtomatik reproduksiya nəticələri",
        "",
        f"**Seed:** {seed} | **n_per_group:** {n_per_group} | **N (ümumi):** {n_per_group * 2}",
        "",
        "> Mənbə: `python scripts/reproduce_section_3_5_3.py`",
        "",
        "## Cədvəl 3.5.6 — Regressiya (5-qat CV, orta R²)",
        "",
        "| Hədəf dəyişən | Ridge | Random Forest | LightGBM | XGBoost |",
        "|---|---:|---:|---:|---:|",
    ]

    for target in TABLE_36_EXPERIMENTS:
        row_cells = [target]
        sub = table_36[table_36["target"] == target]
        for model in TABLE_36_MODELS:
            m = sub[sub["model"] == model]
            if m.empty:
                row_cells.append("—")
            else:
                row_cells.append(_fmt_az_num(float(m.iloc[0]["cv_r2_mean"]), 2))
        lines.append("| " + " | ".join(row_cells) + " |")

    lines.extend(
        [
            "",
            "*Qeyd: Hər xana 5-qat kross-validasiyanın orta R²-sidir.*",
            "",
            "## Cədvəl 3.5.7 — Zero-shot transfer (Ridge)",
            "",
            "| Növ | Hədəf | Proqnoz | Faktiki | Xəta % | Mənbə |",
            "|---|---|---:|---:|---:|---|",
        ]
    )

    for _, r in table_37.iterrows():
        lines.append(
            f"| {r['Növ']} | {r['Hədəf']} | {_fmt_az_num(float(r['Proqnoz']), 2)} | "
            f"{_fmt_az_num(float(r['Faktiki']), 2)} | "
            f"{_fmt_az_num(float(r['Xəta %']), 1)} | {r['flesh_composition_ref']} |"
        )

    if not mape_summary.empty:
        lines.append("")
        lines.append("**Orta mütləq faiz xətası (Ridge):**")
        for _, r in mape_summary.iterrows():
            lines.append(
                f"- {r['target']}: orta |xəta| = {r['mean_abs_error_pct']}%; MAPE ≈ {r['mape_mean']}%"
            )

    lines.extend(
        [
            "",
            "## Klassifikasiya (sintetik AG/RG)",
            "",
        ]
    )
    for exp in ("C1_Water_Group", "C2_FA_Group", "C3_All_Group"):
        sub = cls_results[cls_results["experiment"] == exp]
        if sub.empty:
            continue
        acc = float(sub["CV_Acc_mean"].max())
        lines.append(f"- {EXPERIMENT_ALIASES.get(exp, exp)}: max CV accuracy = {_fmt_az_num(acc, 4)}")

    if not marginal.empty:
        n_ok = int((marginal["relative_error_pct"] <= 5.0).sum())
        n_all = len(marginal)
        max_err = float(marginal["relative_error_pct"].max())
        lines.extend(
            [
                "",
                "## Sintetik marginal uyğunluq",
                "",
                f"- İzlənən parametr sayı: {n_all}",
                f"- ≤5% nisbi xəta: {n_ok} / {n_all}",
                f"- Maksimum nisbi xəta: {_fmt_az_num(max_err, 2)}%",
            ]
        )

    lines.extend(
        [
            "",
            "## Cədvəl 3.5.8",
            "",
            "Bu cədvəl (XGBoost fine-tuning, TVB-N, LOO) notebook-da **yoxdur** — "
            "dissertasiya mətnini bu repodakı nəticələrə uyğun yeniləyin və ya gələcək iş kimi qeyd edin.",
            "",
        ]
    )

    (OUT / "tables_az.md").write_text("\n".join(lines), encoding="utf-8")


def compare_phd_csv(phd_df: pd.DataFrame, path: Path) -> str:
    if not path.is_file():
        return "bundled CSV not found (skipped)"
    ref = pd.read_csv(path)
    if ref.shape != phd_df.shape:
        return f"shape mismatch repo={phd_df.shape} csv={ref.shape}"
    digest_new = hashlib.sha256(phd_df.to_csv(index=False).encode()).hexdigest()[:12]
    digest_ref = hashlib.sha256(ref.to_csv(index=False).encode()).hexdigest()[:12]
    if digest_new == digest_ref:
        return "matches data/silurus_glanis_phd_dataset_v2.csv"
    return f"differs from bundled CSV (new={digest_new}, ref={digest_ref})"


def main() -> int:
    t0 = time.time()
    settings = get_settings()
    seed = settings.seed
    n_per_group = settings.n_per_group

    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Reproducing §3.5.3 (seed={seed}, n_per_group={n_per_group})")

    df_dict = load_default_df_dict()
    synthetic_dict = generate_all_synthetic(df_dict, n_per_group=n_per_group, seed=seed)
    static_df = merge_static(synthetic_dict)
    storage_df = merge_storage(synthetic_dict)
    phd_df = generate_phd_dataset(n_per_group=n_per_group, seed=seed)

    phd_check = compare_phd_csv(phd_df, ROOT / "data" / "silurus_glanis_phd_dataset_v2.csv")
    print(f"  PHD dataset check: {phd_check}")

    marginal = marginal_relative_errors(df_dict, synthetic_dict)
    marginal.to_csv(OUT / "marginal_relative_error.csv", index=False)

    full = prepare_full_dataset(phd_df, static_df, storage_df)
    results, importance = run_all(full, random_state=seed)

    results["experiment_az"] = results["experiment"].map(
        lambda x: EXPERIMENT_ALIASES.get(x, x)
    )
    results.to_csv(OUT / "ml_results_all.csv", index=False)
    importance.to_csv(OUT / "feature_importance.csv", index=False)

    reg = results[results["experiment"].str.startswith("R")]
    cls = results[results["experiment"].str.startswith("C")]
    reg.to_csv(OUT / "regression_all.csv", index=False)
    cls.to_csv(OUT / "classification_all.csv", index=False)

    table_36 = build_table_36(reg)
    table_36.to_csv(OUT / "regression_cv_r2.csv", index=False)

    cls_export = cls[cls["model"].isin(TABLE_36_MODELS + ("Logistic Regression",))].copy()
    cls_export.to_csv(OUT / "classification_cv.csv", index=False)

    transfer = run_transfer_test(phd_df, static_df, storage_df)
    transfer.to_csv(OUT / "transfer_zero_shot.csv", index=False)

    table_37 = build_table_37(transfer)
    table_37.to_csv(OUT / "transfer_table_3_5_7.csv", index=False)

    mape_summary = transfer_mape_summary(transfer)
    mape_summary.to_csv(OUT / "transfer_mape_summary.csv", index=False)

    write_tables_az(
        table_36,
        table_37,
        mape_summary,
        marginal,
        cls,
        seed=seed,
        n_per_group=n_per_group,
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Outputs:")
    for p in sorted(OUT.glob("*")):
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
