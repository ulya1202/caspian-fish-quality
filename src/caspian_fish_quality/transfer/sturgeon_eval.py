"""Cross-species transfer evaluation: Silurus glanis -> Acipenseriformes.

Models trained on synthetic *Silurus glanis* water-quality and flesh-composition
data are applied in a **zero-shot** setting: published sturgeon water conditions
are fed to donor-trained regressors and predictions are compared to literature
proximate composition (Dorojan et al., 2020; Lopez et al., 2020; Ghomi et al.,
2013).

This is *out-of-distribution* transfer (Pan & Yang, 2010), not domain adaptation.
Caspian sturgeons are CITES Appendix II / IUCN CR-EN; live sampling is limited.

References
----------
Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. *IEEE TKDE*,
    22(10), 1345-1359. https://doi.org/10.1109/TKDE.2009.191
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from caspian_fish_quality.ml.datasets import prepare_full_dataset
from caspian_fish_quality.ml.models import _try_import_lightgbm


@dataclass(frozen=True)
class SturgeonCase:
    """One sturgeon species with published water quality and flesh composition."""

    species: str
    common_name: str
    source: str
    water_ref: str
    flesh_ref: str
    water_temp_c: float
    water_ph: float
    water_o2_mgl: float
    chlorides_mgl: float
    nitrites_mgl: float
    nitrates_mgl: float
    ammonium_mgl: float
    phosphates_mgl: float
    group_enc: int
    feed_type: int
    lipids_perc: float
    protein_perc: float
    bodymass_g: float

    def feature_row(self, feature_names: list[str]) -> np.ndarray:
        """Build a single-row feature vector in ``feature_names`` order."""
        mapping: dict[str, float] = {
            "Water_Temp_C": self.water_temp_c,
            "Water_pH": self.water_ph,
            "Water_O2_mgL": self.water_o2_mgl,
            "Chlorides_mgL": self.chlorides_mgl,
            "Nitrites_mgL": self.nitrites_mgl,
            "Nitrates_mgL": self.nitrates_mgl,
            "Ammonium_mgL": self.ammonium_mgl,
            "Phosphates_mgL": self.phosphates_mgl,
            "Group_enc": float(self.group_enc),
            "Feed_Type": float(self.feed_type),
        }
        return np.array([[mapping[f] for f in feature_names]], dtype=np.float64)


def default_sturgeon_cases() -> list[SturgeonCase]:
    """Three peer-reviewed sturgeon benchmarks (dissertation notebook)."""
    return [
        SturgeonCase(
            species="Acipenser stellatus",
            common_name="Sevruga sturgeon",
            source="RAS, Galați, Romania",
            water_ref="Florescu Gune et al. (2021)",
            flesh_ref="Dorojan et al. (2020)",
            water_temp_c=25.0,
            water_ph=7.70,
            water_o2_mgl=6.60,
            chlorides_mgl=90.0,
            nitrites_mgl=0.32,
            nitrates_mgl=32.20,
            ammonium_mgl=0.43,
            phosphates_mgl=0.10,
            group_enc=0,
            feed_type=1,
            lipids_perc=1.32,
            protein_perc=17.86,
            bodymass_g=331.3,
        ),
        SturgeonCase(
            species="Acipenser baerii",
            common_name="Siberian sturgeon",
            source="Farm, Agroittica Lombarda, Italy",
            water_ref="Estimated from facility reports",
            flesh_ref="Lopez et al. (2020)",
            water_temp_c=18.0,
            water_ph=7.40,
            water_o2_mgl=8.50,
            chlorides_mgl=80.0,
            nitrites_mgl=0.10,
            nitrates_mgl=5.0,
            ammonium_mgl=0.15,
            phosphates_mgl=0.08,
            group_enc=0,
            feed_type=1,
            lipids_perc=5.60,
            protein_perc=17.60,
            bodymass_g=6500.0,
        ),
        SturgeonCase(
            species="Huso huso",
            common_name="Beluga sturgeon",
            source="Concrete tanks, Sari, Iran",
            water_ref="Estimated from RAS standards",
            flesh_ref="Ghomi et al. (2013)",
            water_temp_c=22.0,
            water_ph=7.50,
            water_o2_mgl=7.00,
            chlorides_mgl=95.0,
            nitrites_mgl=0.15,
            nitrates_mgl=2.0,
            ammonium_mgl=0.20,
            phosphates_mgl=0.12,
            group_enc=0,
            feed_type=1,
            lipids_perc=3.92,
            protein_perc=14.73,
            bodymass_g=15000.0,
        ),
    ]


def default_sturgeon_references() -> list[SturgeonCase]:
    """Alias for :func:`default_sturgeon_cases` (backward compatibility)."""
    return default_sturgeon_cases()


SturgeonReference = SturgeonCase  # type: ignore[misc, assignment]


_TARGET_LABELS: dict[str, tuple[str, str]] = {
    "Lipids (%)": ("Lipids_perc", "lipids_perc"),
    "Protein (%)": ("Protein_perc", "protein_perc"),
    "Body mass (g)": ("Bodymass_g", "bodymass_g"),
}


def _transfer_model_defs(
    random_state: int,
) -> list[tuple[str, Any]]:
    """Ridge, Random Forest, LightGBM — matches dissertation notebook."""
    defs: list[tuple[str, Any]] = [
        (
            "Ridge",
            lambda: Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        ),
        (
            "Random Forest",
            lambda: Pipeline(
                [
                    ("s", StandardScaler()),
                    (
                        "m",
                        RandomForestRegressor(
                            n_estimators=150,
                            max_depth=8,
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
    ]
    lgb = _try_import_lightgbm()
    if lgb is not None:
        defs.append(
            (
                "LightGBM",
                lambda: Pipeline(
                    [
                        ("s", StandardScaler()),
                        (
                            "m",
                            lgb.LGBMRegressor(
                                n_estimators=150,
                                max_depth=4,
                                learning_rate=0.05,
                                random_state=random_state,
                                verbose=-1,
                            ),
                        ),
                    ]
                ),
            )
        )
    return defs


def run_transfer_test(
    phd_df: pd.DataFrame,
    static_df: pd.DataFrame,
    storage_df: pd.DataFrame,
    *,
    cases: list[SturgeonCase] | None = None,
    references: list[SturgeonCase] | None = None,
    models: tuple[str, ...] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train on *S. glanis*, predict sturgeon flesh traits from literature water.

    Parameters
    ----------
    phd_df, static_df, storage_df
        Donor synthetic frames (same as ML pipeline).
    cases, references
        Optional sturgeon benchmarks; default :func:`default_sturgeon_cases`.
    models
        Subset of ``('Ridge', 'Random Forest', 'LightGBM')``. ``None`` runs all
        available definitions.
    random_state
        RNG seed for Random Forest / LightGBM.

    Returns
    -------
    pandas.DataFrame
        One row per species × target × model with prediction errors and MAPE.
    """
    sturgeons = cases if cases is not None else (references or default_sturgeon_cases())
    full = prepare_full_dataset(phd_df, static_df, storage_df)

    features = [
        "Water_Temp_C",
        "Water_pH",
        "Water_O2_mgL",
        "Chlorides_mgL",
        "Nitrites_mgL",
        "Nitrates_mgL",
        "Ammonium_mgL",
        "Phosphates_mgL",
        "Group_enc",
        "Feed_Type",
    ]
    features = [f for f in features if f in full.columns]

    model_defs = _transfer_model_defs(random_state)
    if models is not None:
        wanted = set(models)
        model_defs = [(n, fn) for n, fn in model_defs if n in wanted]
        if not model_defs:
            raise ValueError("None of the requested model names exist")

    x_train = full[features].to_numpy()
    trained: dict[tuple[str, str], Pipeline] = {}

    for target_label, (col, _attr) in _TARGET_LABELS.items():
        if col not in full.columns:
            continue
        y = full[col].to_numpy()
        for mname, mfn in model_defs:
            pipe = cast(Pipeline, mfn())
            pipe.fit(x_train, y)
            trained[(target_label, mname)] = pipe

    rows: list[dict[str, Any]] = []
    for case in sturgeons:
        x_test = case.feature_row(features)
        for target_label, (_, attr) in _TARGET_LABELS.items():
            actual = float(getattr(case, attr))
            for mname, _ in model_defs:
                key = (target_label, mname)
                if key not in trained:
                    continue
                pred = float(trained[key].predict(x_test)[0])
                err = pred - actual
                pct = (err / actual) * 100.0 if actual != 0 else 0.0
                mape = float(mean_absolute_percentage_error([actual], [pred]))
                rows.append(
                    {
                        "species": case.species,
                        "common_name": case.common_name,
                        "rearing_system": case.source,
                        "water_quality_ref": case.water_ref,
                        "flesh_composition_ref": case.flesh_ref,
                        "target": target_label,
                        "target_col": _TARGET_LABELS[target_label][0],
                        "model": mname,
                        "actual": round(actual, 2),
                        "predicted": round(pred, 2),
                        "absolute_error": round(abs(err), 2),
                        "error_pct": round(pct, 1),
                        "MAPE": round(mape, 4),
                        "citation": case.flesh_ref,
                    }
                )

    return pd.DataFrame(rows)


__all__ = [
    "SturgeonCase",
    "SturgeonReference",
    "default_sturgeon_cases",
    "default_sturgeon_references",
    "run_transfer_test",
]
