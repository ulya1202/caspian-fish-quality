# caspian_fish_quality

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Typed](https://img.shields.io/badge/type-checked-brightgreen)](https://mypy.readthedocs.io/)

Synthetic-data + machine-learning pipeline for sturgeon and herring quality
research in Azerbaijani Caspian waters. Uses Gaussian-copula NORTA sampling
of literature-derived priors for *Silurus glanis*, with leakage-free ML
pipelines and cross-species transfer evaluation toward Acipenseriformes.

This package was developed to support an 80-page Azerbaijani PhD
dissertation. **It is a proof-of-concept built entirely on synthetic data —
see [LIMITATIONS.md](LIMITATIONS.md) before using results scientifically.**

---

## Azərbaycan dilində qısa təsvir

`caspian_fish_quality` paketi Xəzər dənizinin Azərbaycan zonasında nərə
(Acipenseriformes) və kilka (*Clupeonella* spp.) keyfiyyəti tədqiqatları
üçün **sintetik məlumat generasiyası** və **maşın öyrənməsi (ML)
emal zənciri** təqdim edir. Sintetik nümunələr Qauss kopulası (NORTA, Sklar
1959; Cario & Nelson 1997) və kəsilmiş normal paylanma (Robert 1995)
əsasında *Silurus glanis* (naqqa balığı) üzrə peer-reviewed ədəbiyyatdan
çıxarılan priorlar (mean, SD, korrelyasiya matrisi) ilə yaradılır. Növə-
çapraz transfer modulu nərə növləri üçün proxy qiymətləndirmə təmin edir
(zəif / out-of-distribution; bax: `docs/az/limitations.md`).

> **Vacib qeyd:** Heç bir canlı orqanizm üzərində birbaşa təcrübə
> aparılmamışdır. Bütün modelləşdirmə sintetik məlumat üzərində
> aparılıb; etika və CITES məhdudiyyətləri üçün
> [`LIMITATIONS.md`](LIMITATIONS.md) sənədinə baxın.

---

## Installation

```bash
pip install -e ".[dev,test,docs]"
```

Or for production use only:

```bash
pip install caspian_fish_quality
```


## Testing

```bash
pip install -e ".[dev,test]"
pytest
```

## Reproduce dissertation §3.5.3

One command regenerates synthetic data (N=2000), ML tables, and sturgeon
transfer results used in section 3.5.3:

```bash
pip install -e ".[dev,test]"
python scripts/reproduce_section_3_5_3.py
```

Outputs are written to [`results/section_3_5_3/`](results/section_3_5_3/)
including paste-ready Azerbaijani snippets in `tables_az.md`.
Override defaults with `CFQ_SEED=42` and `CFQ_N_PER_GROUP=1000`.

Interactive run: [`notebooks/syntetic_fish_cleaned.ipynb`](notebooks/syntetic_fish_cleaned.ipynb).

## Quick start

```python
import numpy as np
from caspian_fish_quality import (
    generate_all_synthetic,
    merge_static,
    merge_storage,
    sample_truncated,
    gaussian_copula_sample,
)

rng = np.random.default_rng(42)

samples = sample_truncated(mean=1500.0, sd=200.0, lo=1200.0, hi=2400.0, n=1000, rng=rng)
print(samples.mean(), samples.std())
```

For end-to-end orchestration, see
[docs/tutorials/quickstart.md](docs/tutorials/quickstart.md).

## Project layout

```
src/caspian_fish_quality/   installable package source
  data/literature/          Silurus glanis literature priors (data_1–6)
data/                       synthetic reference CSVs (see data/README.md)
notebooks/                  canonical analysis notebook
archive/                    legacy Colab notebook (historical only)
docs/                       Sphinx documentation sources
scripts/                    run_pipeline.py end-to-end smoke test
tests/                      pytest suite
```

## Citing

If you use this package in your research, please cite via
[CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE)
