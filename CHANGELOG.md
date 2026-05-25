# Changelog

All notable changes to `caspian_fish_quality` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-27

### Added
- Initial release migrated from `syntetic_fish_cleaned.ipynb`.
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
