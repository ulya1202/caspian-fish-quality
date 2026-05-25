"""Canlı müdafiə demosu — caspian_fish_quality (Azərbaycan dili).

Streamlit Cloud: ``streamlit run app/streamlit_app.py``
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import caspian_fish_quality as cfq
from caspian_fish_quality.merge import merge_static, merge_storage
from caspian_fish_quality.ml.datasets import (
    generate_phd_dataset,
    prepare_full_dataset,
    run_all,
)
from caspian_fish_quality.synth.generators import (
    generate_all_synthetic,
    load_default_df_dict,
    parse_type_a,
    parse_type_b,
    parse_type_c,
)
from caspian_fish_quality.transfer import (
    default_sturgeon_cases,
    run_transfer_test,
)

REPO_URL = "https://github.com/ulya1202/caspian-fish-quality"
PAGES_URL = "https://ulya1202.github.io/caspian-fish-quality/"

WATER_FEATURES = [
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

TABLE_36 = {
    "Lipid (%)": "R2_Water_Lipids",
    "Bədən kütləsi (q)": "R3_Water_Bodymass",
    "Zülal (%)": "R1_Water_Protein",
}


def _normalize(name: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "", name.lower())


def marginal_summary(
    df_dict: dict[int, pd.DataFrame],
    synthetic_dict: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    parsers = {"A": parse_type_a, "B": parse_type_b, "C": parse_type_c}
    rows: list[dict[str, Any]] = []
    for key, lit_df in df_dict.items():
        syn_df = synthetic_dict.get(key)
        if syn_df is None:
            continue
        col0 = str(lit_df.columns[0]).lower()
        ptype = "C" if "fatty" in col0 else ("B" if "storage" in col0 else "A")
        for grp, priors in parsers[ptype](lit_df).items():
            syn_g = syn_df[syn_df["group"] == grp]
            for _, prow in priors.iterrows():
                lit_mean = float(prow["mean"])
                if lit_mean == 0:
                    continue
                norm = _normalize(str(prow["feature"]))
                col = next(
                    (
                        c
                        for c in syn_g.columns
                        if c != "group" and _normalize(str(c)) == norm
                    ),
                    None,
                )
                if col is None:
                    continue
                syn_mean = float(syn_g[col].mean())
                rel = abs(syn_mean - lit_mean) / abs(lit_mean) * 100.0
                rows.append(
                    {
                        "cədvəl": key,
                        "qrup": grp,
                        "parametr": prow["feature"],
                        "ədəbiyyat": round(lit_mean, 3),
                        "sintetik": round(syn_mean, 3),
                        "nisbi_xəta_%": round(rel, 2),
                    }
                )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Pipeline işləyir (sintetik + ML)…")
def run_pipeline(n_per_group: int, seed: int) -> dict[str, Any]:
    df_dict = load_default_df_dict()
    syn = generate_all_synthetic(df_dict, n_per_group=n_per_group, seed=seed, verbose=False)
    static_df = merge_static(syn)
    storage_df = merge_storage(syn)
    phd_df = generate_phd_dataset(n_per_group=n_per_group, seed=seed)
    full = prepare_full_dataset(phd_df, static_df, storage_df)
    results, _ = run_all(full, random_state=seed)
    transfer = run_transfer_test(phd_df, static_df, storage_df, models=("Ridge",))
    marginal = marginal_summary(df_dict, syn)

    feat_cols = [c for c in WATER_FEATURES if c in full.columns]
    x = full[feat_cols].to_numpy()
    models: dict[str, Pipeline] = {}
    for target in ("Protein_perc", "Lipids_perc", "Bodymass_g"):
        pipe = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))])
        pipe.fit(x, full[target].to_numpy())
        models[target] = pipe

    return {
        "syn": syn,
        "phd_df": phd_df,
        "static_df": static_df,
        "storage_df": storage_df,
        "full": full,
        "results": results,
        "transfer": transfer,
        "marginal": marginal,
        "models": models,
        "feat_cols": feat_cols,
    }


def predict_flesh(
    models: dict[str, Pipeline],
    feat_cols: list[str],
    params: dict[str, float],
) -> dict[str, float]:
    row = {c: params.get(c, 0.0) for c in feat_cols}
    x = np.array([[row[c] for c in feat_cols]], dtype=np.float64)
    return {
        "Zülal (%)": float(models["Protein_perc"].predict(x)[0]),
        "Lipid (%)": float(models["Lipids_perc"].predict(x)[0]),
        "Kütlə (q)": float(models["Bodymass_g"].predict(x)[0]),
    }


st.set_page_config(
    page_title="caspian_fish_quality — canlı demo",
    page_icon="🐟",
    layout="wide",
)

st.title("caspian_fish_quality")
st.caption("Dissertasiya §3.5.3 — sintetik məlumat + maşın öyrənməsi (Ulviyya Aliyeva)")

st.warning(
    "**Vacib:** Bu demo yalnız **sintetik məlumat** üzərində işləyir. "
    "Canlı orqanizm üzərində təcrübə aparılmamışdır. Nəticələr proof-of-concept "
    "xarakterlidir (CITES / IUCN məhdudiyyətləri)."
)

with st.sidebar:
    st.header("Parametrlər")
    seed = st.number_input("Seed", min_value=0, value=42, step=1)
    n_per_group = st.slider("n / qrup (demo sürəti)", 100, 500, 200, 50)
    st.caption(f"Ümumi N ≈ {2 * n_per_group}")
    st.link_button("GitHub repo", REPO_URL, use_container_width=True)
    st.link_button("Layihə səhifəsi", PAGES_URL, use_container_width=True)

tab_about, tab_synth, tab_pred, tab_transfer, tab_results = st.tabs(
    [
        "Layihə",
        "Sintetik generasiya",
        "Proqnoz (S. glanis)",
        "Nərə transferi",
        "Nəticələr",
    ]
)

with tab_about:
    st.subheader("Problem və həll")
    st.markdown(
        """
Xəzər nərə keyfiyyəti tədqiqatları **CITES Appendix II** və **IUCN CR/EN**
statusuna görə məhdudlaşır. Bu layihə:

1. *Silurus glanis* üzrə ədəbiyyat priorlarından **Qauss kopulası (NORTA)** ilə sintetik data yaradır
2. Su keyfiyyəti → ət tərkibi (**zülal, lipid, kütlə**) proqnozunu **ML** ilə öyrənir
3. Modeli **zero-shot** olaraq üç nərə növünə tətbiq edir (*A. stellatus*, *A. baerii*, *H. huso*)

**Pipeline:**
```
Ədəbiyyat CSV (6 cədvəl) → Sintetik generasiya → ML (5-fold CV) → Nərə transferi
```
        """
    )
    st.info(
        "Tam reproduksiya (N=2000): `python scripts/reproduce_section_3_5_3.py` — "
        f"[repo]({REPO_URL})"
    )

with tab_synth:
    st.subheader("Sintetik generasiya")
    if st.button("Generasiya et və yoxla", type="primary"):
        with st.spinner("Hesablanır…"):
            data = run_pipeline(n_per_group, seed)
        st.session_state["pipeline"] = data

    if "pipeline" not in st.session_state:
        st.info("Başlamaq üçün **Generasiya et və yoxla** düyməsini basın.")
    else:
        data = st.session_state["pipeline"]
        st.success(f"PHD dataset: {len(data['phd_df'])} sətir")
        cols = st.columns(len(data["syn"]))
        for i, (k, v) in enumerate(sorted(data["syn"].items())):
            cols[i % len(cols)].metric(f"Cədvəl {k}", f"{len(v)} sətir")

        marg = data["marginal"]
        if not marg.empty:
            n_ok = int((marg["nisbi_xəta_%"] <= 5.0).sum())
            st.metric("Marginal uyğunluq (≤5% xəta)", f"{n_ok} / {len(marg)}")
            st.dataframe(
                marg.sort_values("nisbi_xəta_%").head(15),
                use_container_width=True,
                hide_index=True,
            )

with tab_pred:
    st.subheader("Su keyfiyyəti → ət tərkibi (S. glanis)")
    if "pipeline" not in st.session_state:
        st.info("Əvvəlcə **Sintetik generasiya** tabında pipeline işə salın.")
    else:
        data = st.session_state["pipeline"]
        c1, c2 = st.columns(2)
        with c1:
            temp = st.slider("Su temperaturu (°C)", 8.0, 30.0, 22.0, 0.5)
            ph = st.slider("pH", 6.8, 8.2, 7.5, 0.05)
            o2 = st.slider("Həll olunmuş O₂ (mg/L)", 3.5, 12.0, 6.5, 0.1)
            feed = st.selectbox("Yem (akvakultura)", [("Bəli", 1), ("Xeyr", 0)], format_func=lambda x: x[0])
        with c2:
            cl = st.slider("Xloridlər (mg/L)", 50.0, 120.0, 95.0, 1.0)
            no2 = st.slider("Nitrit (mg/L)", 0.01, 0.25, 0.14, 0.01)
            no3 = st.slider("Nitrat (mg/L)", 0.1, 3.0, 1.8, 0.1)
            group = st.selectbox("Qrup", [("AG — akvakultura", 0), ("RG — vəhşi", 1)], format_func=lambda x: x[0])

        params = {
            "Water_Temp_C": temp,
            "Water_pH": ph,
            "Water_O2_mgL": o2,
            "Chlorides_mgL": cl,
            "Nitrites_mgL": no2,
            "Nitrates_mgL": no3,
            "Ammonium_mgL": 0.22,
            "Phosphates_mgL": 0.14,
            "Group_enc": float(group[1]),
            "Feed_Type": float(feed[1]),
        }
        preds = predict_flesh(data["models"], data["feat_cols"], params)
        m1, m2, m3 = st.columns(3)
        m1.metric("Zülal (%)", f"{preds['Zülal (%)']:.2f}")
        m2.metric("Lipid (%)", f"{preds['Lipid (%)']:.2f}")
        m3.metric("Kütlə (q)", f"{preds['Kütlə (q)']:.0f}")
        st.caption(
            "Temperatur artdıqca lipid biosintezi meylinə uyğun proqnoz dəyişir "
            "(Hallier et al., 2007 — sintetik model)."
        )

with tab_transfer:
    st.subheader("Zero-shot nərə transferi (Ridge)")
    if "pipeline" not in st.session_state:
        st.info("Əvvəlcə **Sintetik generasiya** tabında pipeline işə salın.")
    else:
        cases = default_sturgeon_cases()
        labels = {c.species: c for c in cases}
        choice = st.selectbox("Növ", list(labels.keys()))
        case = labels[choice]
        st.markdown(f"**Mənbə:** {case.flesh_ref} | **Sistem:** {case.source}")

        tr = st.session_state["pipeline"]["transfer"]
        sub = tr[(tr["species"] == choice) & (tr["model"] == "Ridge")]
        if sub.empty:
            st.error("Transfer nəticəsi tapılmadı.")
        else:
            show = sub[sub["target"].isin(["Protein (%)", "Lipids (%)"])].copy()
            chart_df = pd.DataFrame(
                {
                    "Göstərici": show["target"].str.replace(" (%)", "", regex=False),
                    "Proqnoz": show["predicted"],
                    "Ədəbiyyat (faktiki)": show["actual"],
                }
            )
            st.bar_chart(chart_df.set_index("Göstərici"))
            st.dataframe(
                show[
                    ["target", "predicted", "actual", "error_pct", "flesh_composition_ref"]
                ].rename(
                    columns={
                        "target": "Hədəf",
                        "predicted": "Proqnoz",
                        "actual": "Faktiki",
                        "error_pct": "Xəta %",
                        "flesh_composition_ref": "Mənbə",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Ədəbiyyat su parametrləri"):
            st.json({k: v for k, v in asdict(case).items() if k.startswith("water_")})

with tab_results:
    st.subheader("Dissertasiya cədvəlləri (§3.5.3)")
    if "pipeline" not in st.session_state:
        st.info("Əvvəlcə **Sintetik generasiya** tabında pipeline işə salın.")
    else:
        reg = st.session_state["pipeline"]["results"]
        st.markdown("#### Cədvəl 3.5.6 — Regressiya (CV R², orta)")
        rows: list[dict[str, Any]] = []
        for label, exp in TABLE_36.items():
            sub = reg[(reg["experiment"] == exp) & (reg["model"].isin(["Ridge", "Random Forest"]))]
            for _, r in sub.iterrows():
                rows.append(
                    {
                        "Hədəf": label,
                        "Model": r["model"],
                        "R² (CV)": round(float(r["CV_R2_mean"]), 4),
                        "±SD": round(float(r["CV_R2_std"]), 4),
                    }
                )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("#### Cədvəl 3.5.7 — Transfer (Ridge)")
        tr = st.session_state["pipeline"]["transfer"]
        t37 = tr[(tr["model"] == "Ridge") & (tr["target"].isin(["Protein (%)", "Lipids (%)"]))]
        st.dataframe(
            t37[
                ["species", "target", "predicted", "actual", "error_pct", "flesh_composition_ref"]
            ].rename(
                columns={
                    "species": "Növ",
                    "target": "Hədəf",
                    "predicted": "Proqnoz",
                    "actual": "Faktiki",
                    "error_pct": "Xəta %",
                    "flesh_composition_ref": "Mənbə",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        cls = reg[reg["experiment"].str.startswith("C")]
        if not cls.empty:
            st.markdown("#### Klassifikasiya (AG vs RG, sintetik)")
            best = cls.loc[cls.groupby("experiment")["CV_Acc_mean"].idxmax()]
            st.dataframe(
                best[["experiment", "model", "CV_Acc_mean"]].rename(
                    columns={
                        "experiment": "Eksperiment",
                        "model": "Model",
                        "CV_Acc_mean": "CV dəqiqlik",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

st.divider()
st.caption(f"v{cfq.__version__} | [GitHub]({REPO_URL}) | Sintetik proof-of-concept")
