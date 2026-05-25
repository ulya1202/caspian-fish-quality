# Notebooks

## `syntetic_fish_cleaned.ipynb`

Canonical **dissertation §3.5.3** workflow. It does not duplicate package code; it runs:

```bash
python scripts/reproduce_section_3_5_3.py
```

and displays outputs from [`results/section_3_5_3/`](../results/section_3_5_3/).

### Setup

```bash
cd /path/to/caspian-fish-quality
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,test]"
pip install jupyter  # or: pip install -e ".[dev,test]" after jupyter is added to dev extras
jupyter notebook notebooks/syntetic_fish_cleaned.ipynb
```

- **Python:** 3.10–3.12 (matches CI; 3.14 may work but is not in the matrix).
- **Runtime:** ~1–3 minutes for full reproduction with `n_per_group=1000`.
- **Legacy notebook:** [`archive/syntetic_fish_legacy.ipynb`](../archive/syntetic_fish_legacy.ipynb) (Colab-era, not maintained).

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `CFQ_SEED` | `42` | Master RNG seed |
| `CFQ_N_PER_GROUP` | `1000` | Samples per AG/RG group |
