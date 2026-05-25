# Synthetic reference datasets

This directory holds **synthetic** CSV outputs used for dissertation
reproducibility. No live-animal measurements are included.

| File | Description |
|------|-------------|
| `silurus_glanis_phd_dataset_v2.csv` | Canonical PHD water-quality dataset (2 000 rows) generated from literature priors via Gaussian-copula NORTA sampling (`seed=42`, `n_per_group=1000`). |

Intermediate merge outputs (`synthetic_static.csv`, `synthetic_storage.csv`,
`synthetic_all_merged.csv`) and ML result tables are **regeneratable** via
`scripts/run_pipeline.py` or the tutorial in `docs/tutorials/quickstart.md`.

Literature source tables (Silurus glanis priors) live in
`src/caspian_fish_quality/data/literature/` and are loaded by
`load_default_df_dict()`.
