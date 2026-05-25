# caspian_fish_quality

Synthetic-data and machine-learning pipeline for *Silurus glanis* (European wels catfish), with zero-shot transfer evaluation toward three sturgeon species. Built for dissertation §3.5.3 — **all results are synthetic**; see [LIMITATIONS.md](LIMITATIONS.md).

**Repository:** https://github.com/ulya1202/caspian-fish-quality

## Canlı demo (müdafiə)

| Link | Təsvir |
|------|--------|
| [Layihə səhifəsi](https://ulya1202.github.io/caspian-fish-quality/) | Qısa AZ təqdimat + linklər |
| [Canlı demo (Streamlit)](https://caspian-fish-quality.streamlit.app) | İnteraktiv: proqnoz, transfer, cədvəllər |

Lokal demo:

```bash
pip install -e ".[demo,test]"
streamlit run app/streamlit_app.py
```

---

## Clone and reproduce (§3.5.3)

Requirements: **Python 3.10–3.12**, git.

```bash
git clone https://github.com/ulya1202/caspian-fish-quality.git
cd caspian-fish-quality
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,test]"
python scripts/reproduce_section_3_5_3.py
```

This runs (~1–2 minutes):

1. Load six literature prior tables from `src/caspian_fish_quality/data/literature/`
2. Generate **N = 2000** synthetic individuals (1000 AG + 1000 RG) via Gaussian copula (NORTA)
3. Train leakage-free ML models (5-fold CV) and export regression/classification metrics
4. Zero-shot sturgeon transfer (*A. stellatus*, *A. baerii*, *H. huso*)

**Outputs** (created locally, not required in git):

```
results/section_3_5_3/
  RUN_REPORT.md             # Auto report for this run (AZ)
  RUN_REPORT.txt            # Same report as plain text
  tables_az.md              # Paste-ready Azerbaijani table snippets
  regression_cv_r2.csv      # Table 3.5.6
  transfer_table_3_5_7.csv  # Table 3.5.7
  …
```

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `CFQ_SEED` | `42` | Random seed |
| `CFQ_N_PER_GROUP` | `1000` | Samples per aquaculture (AG) / riverine (RG) group |

### Notebook

```bash
pip install -e ".[dev,test]"
jupyter notebook notebooks/syntetic_fish_cleaned.ipynb
```

The notebook only calls the reproduction script and displays `results/section_3_5_3/`.

### Tests and smoke test

```bash
pytest
python scripts/run_pipeline.py    # shorter end-to-end check (~40s with N=1000)
```

---

## Project layout

```
src/caspian_fish_quality/
  data/literature/     data_1.csv … data_6.csv (only bundled inputs)
  synth/              copula + generators
  merge/              static / storage frames
  ml/                 CV experiments
  transfer/           sturgeon zero-shot evaluation
app/
  streamlit_app.py    # canlı müdafiə demosu (AZ)
scripts/
  reproduce_section_3_5_3.py   # main reproduction entry point
  run_pipeline.py              # optional smoke test
notebooks/
  syntetic_fish_cleaned.ipynb
tests/
LIMITATIONS.md
CITATION.cff
```

---

## Citing

Update author fields in [CITATION.cff](CITATION.cff), then use GitHub “Cite this repository” or:

> Aliyeva, U. (2026). *caspian_fish_quality* (Version 0.1.3) [Computer software]. https://github.com/ulya1202/caspian-fish-quality

## License

[MIT](LICENSE)
