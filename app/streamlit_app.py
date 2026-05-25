"""Canlı müdafiə demosu — caspian_fish_quality (Azərbaycan dili).

Yalnız **öncədən öyrədilmiş** modellərdən inferensiya (``demo_artifacts/``).
Yeniləmək üçün: ``python scripts/export_demo_artifacts.py``

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

import caspian_fish_quality as cfq
from caspian_fish_quality.demo.artifacts import DemoArtifacts, default_demo_dir, load_demo_artifacts
from caspian_fish_quality.transfer import default_sturgeon_cases

REPO_URL = "https://github.com/ulya1202/caspian-fish-quality"
PAGES_URL = "https://ulya1202.github.io/caspian-fish-quality/"

TABLE_36 = {
    "Lipid (%)": "R2_Water_Lipids",
    "Bədən kütləsi (q)": "R3_Water_Bodymass",
    "Zülal (%)": "R1_Water_Protein",
}

_DISPLAY = {
    "Protein_perc": "Zülal (%)",
    "Lipids_perc": "Lipid (%)",
    "Bodymass_g": "Kütlə (q)",
}


@st.cache_resource(show_spinner="Öyrədilmiş modellər yüklənir…")
def get_artifacts() -> DemoArtifacts:
    """Load joblib weights and precomputed tables (no training)."""
    return load_demo_artifacts(default_demo_dir(_ROOT))


def predict_flesh(
    art: DemoArtifacts,
    params: dict[str, float],
) -> dict[str, float]:
    row = {c: params.get(c, 0.0) for c in art.feat_cols}
    x = np.array([[row[c] for c in art.feat_cols]], dtype=np.float64)
    out: dict[str, float] = {}
    for col, pipe in art.models.items():
        label = _DISPLAY.get(col, col)
        out[label] = float(pipe.predict(x)[0])
    return out


def marginal_for_display(marginal: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for AZ UI."""
    rename = {
        "table": "cədvəl",
        "group": "qrup",
        "feature": "parametr",
        "literature_mean": "ədəbiyyat",
        "synthetic_mean": "sintetik",
        "relative_error_pct": "nisbi_xəta_%",
    }
    cols = {k: v for k, v in rename.items() if k in marginal.columns}
    return marginal.rename(columns=cols)


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

try:
    art = get_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

m = art.manifest
with st.sidebar:
    st.header("Öyrədilmiş model")
    st.caption(
        f"N={m.get('n_rows_phd', '—')} | seed={m.get('seed')} | "
        f"n/qrup={m.get('n_per_group')} | v{m.get('package_version', cfq.__version__)}"
    )
    st.success("Yalnız inferensiya — təlim bu səhifədə aparılmır.")
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
        "Modellər və cədvəllər repo-da saxlanılıb (`demo_artifacts/`). "
        "Tam təlim/reproduksiya: `python scripts/reproduce_section_3_5_3.py` — "
        f"[repo]({REPO_URL})"
    )

with tab_synth:
    st.subheader("Sintetik generasiya (saxlanılmış nəticələr)")
    st.caption(
        "Marginal uyğunluq və N ölçüsü `export_demo_artifacts.py` ilə bir dəfə hesablanıb; "
        "bu tab yalnız saxlanılmış cədvəli göstərir."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("PHD sətir sayı", m.get("n_rows_phd", "—"))
    c2.metric("Tam dataset", m.get("n_rows_full", "—"))
    c3.metric("Seed", m.get("seed", "—"))

    marg = marginal_for_display(art.marginal)
    if not marg.empty:
        err_col = "nisbi_xəta_%" if "nisbi_xəta_%" in marg.columns else "relative_error_pct"
        n_ok = int((marg[err_col] <= 5.0).sum())
        st.metric("Marginal uyğunluq (≤5% xəta)", f"{n_ok} / {len(marg)}")
        st.dataframe(
            marg.sort_values(err_col).head(15),
            use_container_width=True,
            hide_index=True,
        )

with tab_pred:
    st.subheader("Su keyfiyyəti → ət tərkibi (S. glanis)")
    st.caption("Öyrədilmiş Ridge modelləri — slider dəyişəndə yalnız proqnoz (~ms).")
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
    preds = predict_flesh(art, params)
    m1, m2, m3 = st.columns(3)
    m1.metric("Zülal (%)", f"{preds.get('Zülal (%)', 0):.2f}")
    m2.metric("Lipid (%)", f"{preds.get('Lipid (%)', 0):.2f}")
    m3.metric("Kütlə (q)", f"{preds.get('Kütlə (q)', 0):.0f}")
    st.caption(
        "Temperatur artdıqca lipid biosintezi meylinə uyğun proqnoz dəyişir "
        "(Hallier et al., 2007 — sintetik model)."
    )

with tab_transfer:
    st.subheader("Zero-shot nərə transferi (Ridge)")
    cases = default_sturgeon_cases()
    labels = {c.species: c for c in cases}
    choice = st.selectbox("Növ", list(labels.keys()))
    case = labels[choice]
    st.markdown(f"**Mənbə:** {case.flesh_ref} | **Sistem:** {case.source}")

    tr = art.transfer
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
    st.caption("5-fold CV və transfer — `export_demo_artifacts.py` ilə saxlanılıb.")

    reg = art.regression
    st.markdown("#### Cədvəl 3.5.6 — Regressiya (CV R², orta)")
    if not reg.empty:
        display = reg.copy()
        if "target" in display.columns:
            display = display.rename(columns={"target": "Hədəf", "model": "Model"})
        show_cols = [c for c in ("Hədəf", "Model", "cv_r2_mean", "cv_r2_std") if c in display.columns]
        if show_cols:
            show = display[show_cols].copy()
            show = show.rename(
                columns={"cv_r2_mean": "R² (CV)", "cv_r2_std": "±SD"}
            )
            for col in ("R² (CV)", "±SD"):
                if col in show.columns:
                    show[col] = show[col].astype(float).round(4)
            st.dataframe(show, use_container_width=True, hide_index=True)

    st.markdown("#### Cədvəl 3.5.7 — Transfer (Ridge)")
    t37 = art.table_37
    st.dataframe(t37, use_container_width=True, hide_index=True)

st.divider()
st.caption(f"v{cfq.__version__} | [GitHub]({REPO_URL}) | Sintetik proof-of-concept | inferensiya-only")
