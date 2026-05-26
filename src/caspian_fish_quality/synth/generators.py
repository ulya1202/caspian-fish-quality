"""Master synthetic-data orchestrator over the six literature tables.

Marginal priors (mean, SD, min, max) are extracted from bundled CSV files
derived from Simeanu et al. (2022); see ``docs/LITERATURE_SOURCES.md``.
Pearson *r* targets for the Gaussian copula are structural priors in
``data/correlation_priors.yaml`` (see ``docs/CORRELATION_PRIORS.md``).
"""

from __future__ import annotations

import re
from importlib import resources
from typing import Any

import numpy as np
import pandas as pd

from caspian_fish_quality.synth.copula import build_corr, copula_generate
from caspian_fish_quality.synth.correlation_loader import (
    get_correlations,
    get_storage_correlations,
)

_NUM_LIKE = re.compile(r"^[\d\s.,+-]+$")
_DATE_LIKE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_NA_TOKENS = {"", "-", "ND", "nd", "NaN", "nan", "None", "NaT", ">0.9999"}

_FA_SUMMARY_NAMES = {
    "ΣSFA",
    "ΣMUFA",
    "ΣPUFA",
    "Σ SFA",
    "Σ MUFA",
    "Σ PUFA",
    "n-3",
    "n-6",
    "n−3",
    "n−6",
    "n-3/n-6",
    "n-6/n-3",
    "n−3/n−6",
    "n−6/n−3",
    "PUFA/SFA",
    "USFA/SFA",
    "PI",
    "AI",
    "TI",
    "HFA",
    "hFA",
    "h/H",
}


def fix_index_shift(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and undo a stray numeric index column at column 0.

    Skips the operation when column 0 carries a meaningful keyword like
    ``"storage"``, ``"days"``, ``"interval"``, ``"period"``, ``"time"``,
    ``"fatty"``, ``"acid"``, or ``"group"``.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """
    df = df.copy()
    col0 = str(df.columns[0]).lower()

    skip_keywords = (
        "storage",
        "days",
        "interval",
        "period",
        "time",
        "fatty",
        "acid",
        "group",
    )
    if any(kw in col0 for kw in skip_keywords):
        return df

    vals = df.iloc[:, 0].dropna()
    all_numeric = True
    for v in vals:
        try:
            int(float(v))
        except (ValueError, TypeError):
            if str(v).lower() not in ("p-value", "nan", ""):
                all_numeric = False
                break

    if all_numeric and len(vals) > 0:
        col1_vals = df.iloc[:, 1].dropna()
        all_str = all(
            isinstance(v, str) and not v.replace(".", "").replace("-", "").isdigit()
            for v in col1_vals.head(5)
        )
        if all_str:
            df = df.iloc[:, 1:].reset_index(drop=True)

    return df


def to_float(val: Any) -> float:
    """Convert any value to ``float``, gracefully handling dates and ND tokens.

    Parameters
    ----------
    val : object

    Returns
    -------
    float
        ``np.nan`` when conversion is impossible.
    """
    if val is None:
        return float("nan")
    if isinstance(val, (int, float, np.integer, np.floating)):
        v = float(val)
        return v if np.isfinite(v) else float("nan")
    s = str(val).strip()
    if s in _NA_TOKENS:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        pass

    if hasattr(val, "year"):
        y, m, d = val.year, val.month, val.day
        if y > 2030:
            return float(f"{y}.{m:02d}")
        if 2025 <= y <= 2030:
            return float(f"{d}.0{m}") if m < 10 else float(f"{d}.{m}")

    dm = _DATE_LIKE.match(s)
    if dm:
        y, m, d = int(dm.group(1)), int(dm.group(2)), int(dm.group(3))
        if y > 2030:
            return float(f"{y}.{m:02d}")
        if y >= 2025:
            return float(f"{d}.0{m}") if m < 10 else float(f"{d}.{m}")

    return float("nan")


def find_cols(columns: pd.Index, group: str, keyword: str) -> list[str]:
    """Return columns belonging to ``group`` whose name contains ``keyword``."""
    return [c for c in columns if group in c and keyword.lower() in c.lower()]


def parse_mean_sem_str(val: Any) -> tuple[float | None, float | None]:
    """Parse a combined ``"mean ± SEM"`` string.

    Parameters
    ----------
    val : object

    Returns
    -------
    tuple[float | None, float | None]
        Pair of ``(mean, sem)`` parsed values; ``(None, None)`` when no
        ``±`` is present or when either component fails to parse.
    """
    s = str(val).strip()
    if "±" in s:
        parts = s.split("±")
        try:
            return float(parts[0].strip()), float(parts[1].strip())
        except (ValueError, IndexError):
            return None, None
    return None, None


def parse_type_a(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Parse Tables 1, 2, 3: per-trait Mean/SEM with optional Min/Max."""
    df = fix_index_shift(df)
    feature_col = df.columns[0]
    result: dict[str, pd.DataFrame] = {}

    for group in ("RG", "AG"):
        mean_cols = find_cols(df.columns, group, "Mean")
        sem_cols = find_cols(df.columns, group, "SEM")
        min_cols = find_cols(df.columns, group, "Min")
        max_cols = find_cols(df.columns, group, "Max")

        combined = [c for c in mean_cols if "SEM" in c or "±" in c]
        mean_only = [c for c in mean_cols if c not in combined and "SEM" not in c]
        sem_only = [c for c in sem_cols if c not in combined and "Mean" not in c]

        if mean_only and sem_only:
            mc, sc, use_combined = mean_only[0], sem_only[0], False
        elif combined:
            mc, sc, use_combined = combined[0], None, True
        else:
            continue

        n_match = re.search(r"n=(\d+)", mc)
        n_orig = int(n_match.group(1)) if n_match else 50
        min_c = min_cols[0] if min_cols else None
        max_c = max_cols[0] if max_cols else None

        rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            feat = str(row[feature_col]).strip()
            try:
                float(feat)
                continue
            except ValueError:
                pass

            if use_combined:
                mean, sem = parse_mean_sem_str(row[mc])
            else:
                assert sc is not None  # guaranteed by branch above
                mean = to_float(row[mc])
                sem = to_float(row[sc])

            if mean is None or (isinstance(mean, float) and np.isnan(mean)):
                continue
            if sem is None or (isinstance(sem, float) and np.isnan(sem)):
                sem = 0.0

            sd = sem * np.sqrt(n_orig)
            mn = to_float(row[min_c]) if min_c else float("nan")
            mx = to_float(row[max_c]) if max_c else float("nan")
            if np.isnan(mn):
                mn = mean - 3 * sd if sd > 0 else mean * 0.8
            if np.isnan(mx):
                mx = mean + 3 * sd if sd > 0 else mean * 1.2
            mn = min(mn, mean)
            mx = max(mx, mean)
            if mn >= mx:
                mn, mx = mean * 0.9, mean * 1.1

            rows.append(
                {
                    "feature": feat,
                    "mean": mean,
                    "sem": sem,
                    "sd": max(sd, 1e-6),
                    "min": mn,
                    "max": mx,
                    "n": n_orig,
                }
            )

        if rows:
            result[group] = pd.DataFrame(rows)

    return result


def parse_type_b(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Parse Tables 4, 5: time-series with separate Mean/SEM columns."""
    df = fix_index_shift(df)
    time_col = df.columns[0]
    group_col: str | None = None
    n_col: str | None = None

    for c in df.columns:
        cl = c.lower()
        if "group" in cl:
            group_col = c
        elif cl == "n":
            n_col = c

    if group_col is None and len(df.columns) > 2:
        group_col = df.columns[2]
    if n_col is None and len(df.columns) > 1:
        n_col = df.columns[1]

    skip = {time_col, n_col, group_col}
    measure_info: list[tuple[str, str, str | None]] = []
    remaining = [c for c in df.columns if c not in skip]
    used: set[str] = set()

    for c in remaining:
        if c in used:
            continue
        cl = c.lower()
        if "mean" in cl:
            base = c.split("Mean")[0].strip().rstrip(":").strip()
            sem_c: str | None = None
            for c2 in remaining:
                if c2 != c and c2 not in used and "sem" in c2.lower():
                    base2 = c2.split("SEM")[0].strip().rstrip(":").strip()
                    if base2 == base or base.lower() in c2.lower():
                        sem_c = c2
                        break
            measure_info.append((base, c, sem_c))
            used.add(c)
            if sem_c:
                used.add(sem_c)
        elif "±" in cl or ("mean" in cl and "sem" in cl):
            base = c.split(":")[0].strip()
            measure_info.append((base, c, None))
            used.add(c)

    if not measure_info:
        for c in remaining:
            if c not in used:
                base = c.split(":")[0].strip()
                measure_info.append((base, c, None))

    result: dict[str, list[dict[str, Any]]] = {}
    for _, row in df.iterrows():
        t = str(row[time_col]).strip()
        g = str(row[group_col]).strip() if group_col else ""
        if t.lower() == "p-value" or g in ("NaN", "nan", "", "None"):
            continue
        try:
            day = int(float(t))
        except (ValueError, TypeError):
            continue

        nv = to_float(row[n_col]) if n_col else 6.0
        n_orig = int(nv) if not np.isnan(nv) else 6

        for base, mc, sc in measure_info:
            if sc:
                mean = to_float(row[mc])
                sem_v = to_float(row[sc])
            else:
                mean_opt, sem_opt = parse_mean_sem_str(row[mc])
                if mean_opt is None:
                    mean = to_float(row[mc])
                    sem_v = 0.0
                else:
                    mean = mean_opt
                    sem_v = sem_opt if sem_opt is not None else 0.0

            if np.isnan(mean):
                continue
            if np.isnan(sem_v):
                sem_v = 0.0

            sd = sem_v * np.sqrt(n_orig) if sem_v > 0 else 0.01
            entry = {
                "feature": f"{base}_day{day}",
                "feature_base": base,
                "mean": mean,
                "sem": sem_v,
                "sd": sd,
                "min": max(0.0, mean - 3 * sd),
                "max": min(100.0, mean + 3 * sd),
                "n": n_orig,
                "day": day,
            }
            result.setdefault(g, []).append(entry)

    return {k: pd.DataFrame(v) for k, v in result.items()}


def parse_type_c(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Parse Table 6 (fatty acids) with separate or combined Mean/SEM."""
    df = fix_index_shift(df)
    fa_col = df.columns[0]
    result: dict[str, pd.DataFrame] = {}

    for group in ("AG", "RG"):
        mean_cols = find_cols(df.columns, group, "Mean")
        sem_cols = find_cols(df.columns, group, "SEM")
        combined = [c for c in mean_cols if "SEM" in c or "±" in c]
        mean_only = [c for c in mean_cols if c not in combined and "SEM" not in c]
        sem_only = [c for c in sem_cols if c not in combined and "Mean" not in c]

        if mean_only and sem_only:
            mc, sc, use_combined = mean_only[0], sem_only[0], False
        elif combined:
            mc, sc, use_combined = combined[0], None, True
        else:
            continue

        n_match = re.search(r"n=(\d+)", mc)
        n_orig = int(n_match.group(1)) if n_match else 6

        rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            fa = str(row[fa_col]).strip()
            is_derived = fa in _FA_SUMMARY_NAMES

            if use_combined:
                val_str = str(row[mc]).strip()
                if val_str.upper() == "ND":
                    continue
                mean_opt, sem_opt = parse_mean_sem_str(row[mc])
                if mean_opt is None:
                    mean = to_float(row[mc])
                    sem_v: float | None = None
                else:
                    mean = mean_opt
                    sem_v = sem_opt
            else:
                assert sc is not None
                raw_mean = row[mc]
                if str(raw_mean).strip().upper() == "ND":
                    continue
                mean = to_float(raw_mean)
                sem_v = to_float(row[sc])

            if np.isnan(mean):
                continue
            if sem_v is None or (isinstance(sem_v, float) and np.isnan(sem_v)):
                sem_v = 0.0

            sd = sem_v * np.sqrt(n_orig) if sem_v > 0 else max(mean * 0.05, 0.001)

            rows.append(
                {
                    "feature": fa,
                    "mean": mean,
                    "sem": sem_v,
                    "sd": max(sd, 0.001),
                    "min": max(0.0, mean - 3 * sd),
                    "max": mean + 3 * sd,
                    "n": n_orig,
                    "is_derived": is_derived,
                }
            )

        if rows:
            result[group] = pd.DataFrame(rows)

    return result


def calc_yields(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``*yield(%)`` columns derived from any ``*mass`` columns vs ``Livemass``."""
    df = df.copy()
    lm_col = next((c for c in df.columns if "live" in c.lower() and "mass" in c.lower()), None)
    if not lm_col:
        return df
    lm = df[lm_col]
    for c in df.columns:
        cl = c.lower()
        if "mass" in cl and "live" not in cl:
            yc = c.replace("mass", "yield").replace("Mass", "Yield")
            if "(g)" in yc:
                yc = yc.replace("(g)", "(%)")
            df[yc] = (df[c] / lm) * 100
    return df


def calc_fa_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SFA/MUFA/PUFA, n-3/n-6, AI, TI fatty-acid indices.

    AI and TI follow Ulbricht & Southgate (1991, *Lancet* 338:985-992).
    """
    df = df.copy()
    cols = df.columns.tolist()
    sfa = [c for c in cols if re.match(r"^C\s*\d+:\s*0$", c.strip())]
    mufa = [c for c in cols if re.match(r"^C\s*\d+:\s*1", c.strip())]
    n3 = [c for c in cols if ("n-3" in c or "n−3" in c) and c.strip().startswith("C")]
    n6 = [c for c in cols if ("n-6" in c or "n−6" in c) and c.strip().startswith("C")]
    pufa = list(set(n3 + n6))

    if sfa:
        df["ΣSFA"] = df[sfa].sum(axis=1)
    if mufa:
        df["ΣMUFA"] = df[mufa].sum(axis=1)
    if pufa:
        df["ΣPUFA"] = df[pufa].sum(axis=1)
    if n3:
        df["n-3"] = df[n3].sum(axis=1)
    if n6:
        df["n-6"] = df[n6].sum(axis=1)

    if "n-3" in df.columns and "n-6" in df.columns:
        df["n-3/n-6"] = df["n-3"] / df["n-6"].replace(0, np.nan)
        df["n-6/n-3"] = df["n-6"] / df["n-3"].replace(0, np.nan)
    if "ΣPUFA" in df.columns and "ΣSFA" in df.columns:
        df["PUFA/SFA"] = df["ΣPUFA"] / df["ΣSFA"].replace(0, np.nan)
    if all(x in df.columns for x in ("ΣMUFA", "ΣPUFA", "ΣSFA")):
        df["USFA/SFA"] = (df["ΣMUFA"] + df["ΣPUFA"]) / df["ΣSFA"].replace(0, np.nan)

    c14 = df.get("C14:0", df.get("C 14:0", pd.Series([0.0] * len(df), index=df.index)))
    c16 = df.get("C16:0", df.get("C 16:0", pd.Series([0.0] * len(df), index=df.index)))
    c18s = df.get("C18:0", df.get("C 18:0", pd.Series([0.0] * len(df), index=df.index)))
    mu = df.get("ΣMUFA", pd.Series([0.0] * len(df), index=df.index))
    n3s = df.get("n-3", pd.Series([0.0] * len(df), index=df.index))
    n6s = df.get("n-6", pd.Series([0.0] * len(df), index=df.index))

    ai_d = mu + n6s + n3s
    df["AI"] = (4 * c14 + c16) / ai_d.replace(0, np.nan)

    n3n6 = (n3s / n6s.replace(0, np.nan)).fillna(0)
    ti_d = 0.5 * mu + 0.5 * n6s + 3 * n3s + n3n6
    df["TI"] = (c14 + c16 + c18s) / ti_d.replace(0, np.nan)

    return df


def generate_all_synthetic(
    df_dict: dict[int, pd.DataFrame],
    n_per_group: int = 1000,
    seed: int = 42,
    *,
    verbose: bool = False,
) -> dict[int, pd.DataFrame]:
    """Generate synthetic data for every source table in ``df_dict``.

    Parameters
    ----------
    df_dict : dict[int, pandas.DataFrame]
        Source tables keyed by 1..6.
    n_per_group : int, default 1000
        Samples drawn per group (AG/RG).
    seed : int, default 42
    verbose : bool, default False
        When ``True``, print per-table progress.

    Returns
    -------
    dict[int, pandas.DataFrame]
        Mapping ``{1: synthetic_df1, ..., 6: synthetic_df6}``.
    """
    rng = np.random.default_rng(seed)
    lit = get_correlations()
    stor = get_storage_correlations()
    out: dict[int, pd.DataFrame] = {}

    for key, df in df_dict.items():
        if verbose:
            print(f"\n{'=' * 50}\nTable {key}: {df.shape}")

        col0 = str(df.columns[0]).lower()

        if "fatty" in col0:
            ttype = "C"
        elif "storage" in col0:
            ttype = "B"
        else:
            ttype = "A"

        if ttype == "A":
            parsed = parse_type_a(df)
            cs = lit.get(key, {})
            frames: list[pd.DataFrame] = []
            for grp, sdf in parsed.items():
                feats = sdf["feature"].tolist()
                R = build_corr(feats, cs)
                syn = copula_generate(
                    feats,
                    sdf["mean"].to_numpy(),
                    sdf["sd"].to_numpy(),
                    sdf["min"].to_numpy(),
                    sdf["max"].to_numpy(),
                    R,
                    n_per_group,
                    int(rng.integers(0, 2**31)),
                )
                syn["group"] = grp
                if key == 3:
                    syn = calc_yields(syn)
                frames.append(syn)
                if verbose:
                    print(f"  {grp}: {len(feats)} features x {n_per_group}")
            if frames:
                out[key] = pd.concat(frames, ignore_index=True)

        elif ttype == "B":
            parsed = parse_type_b(df)
            frames = []
            for grp, sdf in parsed.items():
                days = sorted(sdf["day"].unique())
                for day in days:
                    ddf = sdf[sdf["day"] == day]
                    feats = ddf["feature"].tolist()
                    bases = ddf["feature_base"].tolist()
                    cspec: dict[tuple[str, str], float] = {}
                    for (a, b), r in stor.items():
                        for i, bi in enumerate(bases):
                            for j, bj in enumerate(bases):
                                if a in bi.lower() and b in bj.lower() and i != j:
                                    cspec[(feats[i], feats[j])] = r
                    R = build_corr(feats, cspec)
                    syn = copula_generate(
                        feats,
                        ddf["mean"].to_numpy(),
                        ddf["sd"].to_numpy(),
                        ddf["min"].to_numpy(),
                        ddf["max"].to_numpy(),
                        R,
                        n_per_group,
                        int(rng.integers(0, 2**31)),
                    )
                    syn["group"] = grp
                    syn["storage_day"] = day
                    syn.columns = [
                        c.rsplit("_day", 1)[0] if "_day" in c else c for c in syn.columns
                    ]
                    frames.append(syn)
                if verbose:
                    print(f"  {grp}: {len(days)} timepoints x {n_per_group}")
            if frames:
                out[key] = pd.concat(frames, ignore_index=True)

        elif ttype == "C":
            parsed = parse_type_c(df)
            cs = lit.get(key, {})
            frames = []
            for grp, sdf in parsed.items():
                base = sdf[~sdf["is_derived"]].copy()
                if base.empty:
                    continue
                feats = base["feature"].tolist()
                R = build_corr(feats, cs)
                syn = copula_generate(
                    feats,
                    base["mean"].to_numpy(),
                    base["sd"].to_numpy(),
                    base["min"].to_numpy(),
                    base["max"].to_numpy(),
                    R,
                    n_per_group,
                    int(rng.integers(0, 2**31)),
                )
                syn = calc_fa_indices(syn)
                syn["group"] = grp
                frames.append(syn)
                if verbose:
                    nb = len(feats)
                    nd = len(syn.columns) - nb - 1
                    print(f"  {grp}: {nb} base FAs + {nd} derived x {n_per_group}")
            if frames:
                out[key] = pd.concat(frames, ignore_index=True)

    if verbose:
        print(f"\n{'=' * 50}\nDONE!")
        for k, v in out.items():
            print(f"  Table {k}: {v.shape}")
    return out


def validate(synthetic_dict: dict[int, pd.DataFrame], table_key: int) -> pd.DataFrame:
    """Return a per-group ``describe`` summary for the requested synthetic table.

    Parameters
    ----------
    synthetic_dict : dict[int, pandas.DataFrame]
    table_key : int

    Returns
    -------
    pandas.DataFrame
        Concatenated summary statistics with a ``group`` column.
    """
    df = synthetic_dict.get(table_key)
    if df is None:
        raise KeyError(f"Table {table_key} not found in synthetic_dict")

    summaries: list[pd.DataFrame] = []
    for grp in df["group"].unique():
        g = df[df["group"] == grp]
        num = g.select_dtypes(include=[np.number]).drop(columns=["storage_day"], errors="ignore")
        desc = num.describe().round(4)
        desc["group"] = grp
        summaries.append(desc)
    return pd.concat(summaries) if summaries else pd.DataFrame()


def load_default_df_dict() -> dict[int, pd.DataFrame]:
    """Load the bundled six literature CSVs into a ``df_dict``.

    Returns
    -------
    dict[int, pandas.DataFrame]
        Mapping ``{1: data_1.csv, ..., 6: data_6.csv}``.
    """
    root = resources.files("caspian_fish_quality").joinpath("data").joinpath("literature")
    out: dict[int, pd.DataFrame] = {}
    for i in range(1, 7):
        with resources.as_file(root.joinpath(f"data_{i}.csv")) as path:
            out[i] = pd.read_csv(path)
    return out


__all__ = [
    "calc_fa_indices",
    "calc_yields",
    "find_cols",
    "fix_index_shift",
    "generate_all_synthetic",
    "get_correlations",
    "get_storage_correlations",
    "load_default_df_dict",
    "parse_mean_sem_str",
    "parse_type_a",
    "parse_type_b",
    "parse_type_c",
    "to_float",
    "validate",
]
