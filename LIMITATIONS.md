# Limitations and caveats (English summary)

For the full Azerbaijani limitations chapter, see
[`docs/az/limitations.md`](docs/az/limitations.md).

## 1. Synthetic data is not real data

**No live experimentation has been performed.** Every numeric result in
this package was obtained on synthetic samples drawn via Gaussian-copula
NORTA (Sklar, 1959; Cario & Nelson, 1997) from literature-derived priors
for *Silurus glanis* (Bergstrom et al., 2022; Yazici & Yazicioglu, 2020;
Simeanu et al., 2022; Ljubojevic et al., 2013; Jankowska et al., 2007;
Hallier et al., 2007). Marginal/correlation priors come from small N
studies (N = 6 to 50); per-feature truncated-normal marginals do not
capture all biochemical detail.

## 2. Cross-species transfer is weak / out-of-distribution

Transfer from *Silurus glanis* (Siluriformes) to *Acipenser/Huso*
(Acipenseriformes) crosses 200+ MY of evolutionary divergence and is
classified as a hard transfer regime (Pan & Yang, 2010). Only direction-
of-effect agreement should be interpreted as a meaningful success signal;
absolute MAPE is informational only. Use `transfer.h_divergence_proxy`
(Ben-David et al., 2010) to quantify domain similarity before relying on
transfer outputs.

## 3. Citation verification

All references in `REFERENCES.bib` were cross-checked against Crossref,
PubMed, publisher pages, arXiv, or institutional URLs. Items the user
flagged as `[VERIFY]` and could not be matched 1:1 have been replaced
with verified alternatives:

| Originally flagged | Verified replacement |
|---|---|
| Florescu Gune et al. (2021) — Crossref no match | Bronzi & Rosenthal (2014) — Caspian sturgeon production |
| Dorojan (2020) — publisher page unreachable | Boscari et al. (2017) — sterlet biochemistry |
| Bergstrom et al. (2022) — initial version retracted | post-erratum 2022 release |

## 4. Ethics and CITES

Caspian sturgeons (*Huso huso*, *A. persicus*, *A. gueldenstaedtii*,
*A. nudiventris*, *A. stellatus*) are CITES Appendix II and IUCN
Red List CR/EN (CITES, 2018; IUCN, 2022). Live experimentation requires
institutional ethics review and CITES export permits; the synthetic-data
approach used here was chosen specifically because such permits were not
available within the dissertation timeline.

## 5. ML caveats

* Model variance is high in small-N regimes; 5-fold CV does not fully
  account for this.
* TSTS (Train-Synthetic, Test-Synthetic) protocol is optimistically
  biased; always use TSTR (`validation.tstr.tstr_protocol`) for utility
  claims.
* Default hyperparameters were inherited from the dissertation notebook
  and may not be optimal for new data.

## 6. Future work (non-goals of v0.1.0)

* CTGAN / TVAE / vine copulas for stronger generative models.
* Direct integration with ANAS (Azerbaijan National Academy of Sciences)
  empirical samples once ethics approvals are in hand.
* Real Caspian sturgeon validation samples.
* Bayesian hierarchical priors (multi-study evidence synthesis).

## 7. Terminology corrections (Azerbaijani)

* **Huso huso = bölgə (or ağ baliq)**, NOT *uzunburun*.
* **Acipenser nudiventris = uzunburun (ship)**.
* **Silurus glanis = naqqa balığı**.

These corrections are applied in `docs/az/glossary.md` and propagated
throughout the documentation.
