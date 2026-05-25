"""End-to-end smoke test for the caspian_fish_quality package."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import caspian_fish_quality as cfq
from caspian_fish_quality.ml.datasets import (
    generate_phd_dataset,
    prepare_full_dataset,
    run_all,
)
from caspian_fish_quality.synth.generators import load_default_df_dict
from caspian_fish_quality.transfer import h_divergence_proxy
from caspian_fish_quality.validation import (
    frobenius_distance,
    ks_per_variable,
    tstr_protocol,
    wasserstein_per_variable,
)
from caspian_fish_quality.validation.joint import empirical_correlation


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    t0 = time.time()
    print(f"caspian_fish_quality version: {cfq.__version__}")

    banner("STEP 1 - Load 6 bundled literature CSVs")
    df_dict = load_default_df_dict()
    for k, v in df_dict.items():
        print(f"  table {k}: {v.shape[0]:>3} rows x {v.shape[1]:>2} cols")

    banner("STEP 2 - Generate synthetic data via Gaussian copula")
    syn = cfq.generate_all_synthetic(df_dict, n_per_group=400, seed=42)
    for k, v in syn.items():
        groups = ", ".join(sorted(v["group"].unique()))
        print(f"  table {k}: {len(v):>4} rows | groups: {groups} | cols: {v.shape[1]}")

    banner("STEP 3 - Merge static + storage frames")
    static_df = cfq.merge_static(syn)
    storage_df = cfq.merge_storage(syn)
    print(f"  static_df:  {static_df.shape}")
    print(f"  storage_df: {storage_df.shape}")
    print(f"  static cols (first 12): {list(static_df.columns[:12])}")

    banner("STEP 4 - Generate PHD water-quality dataset")
    phd_df = generate_phd_dataset(n_per_group=400, seed=42)
    print(f"  phd_df: {phd_df.shape}")
    print(f"  groups: {phd_df['Group'].value_counts().to_dict()}")
    print(
        f"  Bodymass range: [{phd_df['Bodymass_g'].min():.0f}, "
        f"{phd_df['Bodymass_g'].max():.0f}] g"
    )

    banner("STEP 5 - Build full leakage-free ML dataset")
    full = prepare_full_dataset(phd_df, static_df, storage_df)
    print(f"  full dataset: {full.shape[0]} rows x {full.shape[1]} cols")
    print(f"  Group_enc unique: {sorted(full['Group_enc'].unique())}")

    banner("STEP 6 - Run experiment grid (regression + classification)")
    res, imp = run_all(full, random_state=42)
    print(f"  results table:    {res.shape}")
    print(f"  importance table: {imp.shape}")
    print("\n  Top regression rows:")
    reg_cols = [c for c in ["experiment", "model", "R2", "RMSE", "MAE"] if c in res.columns]
    reg = res[res["experiment"].str.startswith("R")][reg_cols].head(5)
    print(reg.to_string(index=False))
    print("\n  Top classification rows:")
    cls_cols = [
        c for c in ["experiment", "model", "Accuracy", "F1"] if c in res.columns
    ]
    cls = res[res["experiment"].str.startswith("C")][cls_cols].head(5)
    print(cls.to_string(index=False))

    banner("STEP 7 - Marginal validation (KS, Wasserstein)")
    cols = ["Bodymass_g", "Protein_perc", "Lipids_perc", "Water_O2_mgL"]
    real = phd_df[cols].copy()
    rng = np.random.default_rng(0)
    syn_phd = real + rng.normal(0, 0.05 * real.std().to_numpy(), size=real.shape)
    ks = ks_per_variable(real, syn_phd)
    print("  KS p-values:")
    print(ks[["feature", "D", "p_raw"]].to_string(index=False))
    w1 = wasserstein_per_variable(real, syn_phd)
    print("\n  W1 (normalized):")
    print(w1.to_string(index=False))

    banner("STEP 8 - Joint validation (Frobenius distance)")
    target = empirical_correlation(real)
    emp = empirical_correlation(syn_phd)
    fro = frobenius_distance(target, emp)
    print(f"  Frobenius={fro['frobenius']:.4f} per-dim={fro['frobenius_per_dim']:.4f} "
          f"max_abs={fro['max_abs_dev']:.4f}")

    banner("STEP 9 - TSTR protocol")
    real_full = phd_df[[*cols, "Group_enc" if "Group_enc" in phd_df.columns else "Group"]].copy()
    real_full["target"] = phd_df["Protein_perc"]
    syn_full = syn_phd.copy()
    syn_full["target"] = syn_phd["Protein_perc"]
    tstr = tstr_protocol(
        real_full[cols + ["target"]], syn_full[cols + ["target"]], "target", cols
    )
    print(tstr.to_string(index=False))

    banner("STEP 10 - Cross-species transfer (sturgeon)")
    transfer = cfq.run_transfer_test(phd_df, static_df, storage_df)
    print(f"  transfer rows: {len(transfer)}")
    print("  by species/target/model:")
    print(transfer.head(9).to_string(index=False))

    banner("STEP 11 - H-divergence proxy")
    a = phd_df[phd_df["Group"] == "AG"].select_dtypes(include="number")
    b = phd_df[phd_df["Group"] == "RG"].select_dtypes(include="number")
    h = h_divergence_proxy(a, b)
    print(f"  AG vs RG -> {h}")

    print(f"\n{'=' * 70}")
    print(f"OK - finished in {time.time() - t0:.1f}s. Package works end-to-end.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 12)
    main()
