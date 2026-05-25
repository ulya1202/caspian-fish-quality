# Changelog

All notable changes to `caspian_fish_quality` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-05-25

### Added
- `scripts/reproduce_section_3_5_3.py` and frozen `results/section_3_5_3/`
  outputs for dissertation §3.5.3 (tables 3.5.6–3.5.7).
- Slim `notebooks/syntetic_fish_cleaned.ipynb` wrapper and `notebooks/README.md`.

### Changed
- `transfer.sturgeon_eval` now matches notebook zero-shot protocol
  (*A. stellatus*, *A. baerii*, *H. huso*; Dorojan / Lopez / Ghomi refs).
- `run_pipeline.py` uses `Settings` defaults (`n_per_group=1000`, `seed=42`).

## [0.1.0] - 2026-04-27

### Added
- Initial release migrated from `notebooks/syntetic_fish_cleaned.ipynb`.
- `eda` subpackage with column parsing, scientific-notation cleaning, and
  ± splitter helpers.
- `synth` subpackage:
  - `moments.sem_to_sd` (Altman & Bland 2005).
  - `truncated.sample_truncated` (Robert 1995).
  - `copula.gaussian_copula_sample` / `copula.copula_generate` NORTA
    construction (Sklar 1959; Cario & Nelson 1997; Ghosh & Henderson 2003).
  - `generators.generate_all_synthetic` master orchestrator over six
    Silurus glanis literature tables.
- `merge.merge_static` and `merge.merge_storage`.
- `ml` subpackage with leakage-free Pipeline factories, regression and
  classification model dictionaries, CV, metrics, and dataset orchestration.
- `transfer.run_transfer_test` for Silurus glanis to Acipenseriformes
  evaluation, with `SturgeonReference` dataclass.
- `transfer.h_divergence_proxy` for Ben-David et al. (2010) domain check.
- `validation` subpackage: KS, Mann-Whitney U, 1-Wasserstein, Frobenius,
  and TSTR (Esteban et al. 2017).
- Sphinx + Furo + MyST documentation with bilingual (English autodoc API,
  Azerbaijani narrative) chapters.
- `pytest` suite with Hypothesis property tests and pipeline-leakage
  invariants.
- `CITATION.cff`, `REFERENCES.bib` (80+ verified Crossref entries),
  `METHODS.md`, `LIMITATIONS.md`, `GLOSSARY.md`.
- GitHub Actions CI matrix (3.10 - 3.13) and PyPI release workflow.
