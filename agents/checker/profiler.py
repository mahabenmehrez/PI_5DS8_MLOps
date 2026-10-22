"""Builds a reference profile from any tabular DataFrame.

The profile is what "normal" looks like. Every incoming batch is compared to it,
so nothing here knows about a specific dataset: column kinds, ranges, null rates
and distribution bins are all learned from the reference data.
"""
import json

import numpy as np
import pandas as pd

N_BINS = 10
TOP_CATEGORIES = 30
OTHER = "__other__"


def column_kind(series: pd.Series, categorical: set) -> str:
    if series.name in categorical or pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def numeric_bins(values: np.ndarray) -> tuple[list, list]:
    """Interior bin edges from reference quantiles, and the share of rows per bin."""
    edges = np.unique(np.quantile(values, np.linspace(0, 1, N_BINS + 1)[1:-1]))
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return edges.tolist(), (counts / counts.sum()).tolist()


def category_shares(series: pd.Series, keep: list | None = None) -> dict:
    values = series.dropna().astype(str)
    shares = values.value_counts(normalize=True)
    if keep is None:
        keep = shares.index[:TOP_CATEGORIES].tolist()
    out = {k: float(shares.get(k, 0.0)) for k in keep}
    out[OTHER] = float(max(0.0, 1.0 - sum(out.values())))
    return out


def build_profile(df: pd.DataFrame, cfg: dict) -> dict:
    categorical = set(cfg.get("categorical", []))
    ignore = set(cfg.get("ignore_columns", []))
    columns = {}
    for name in df.columns:
        if name in ignore:
            continue
        s = df[name]
        kind = column_kind(s, categorical)
        col = {"kind": kind, "dtype": str(s.dtype), "null_rate": float(s.isna().mean())}
        non_null = s.dropna()
        if kind == "numeric" and len(non_null):
            v = non_null.to_numpy(dtype=float)
            p_lo, p_hi = np.quantile(v, [0.001, 0.999])
            margin = 0.1 * (p_hi - p_lo)
            edges, shares = numeric_bins(v)
            p25, p75 = np.quantile(v, [0.25, 0.75])
            sample = v if len(v) <= 500 else np.random.default_rng(0).choice(v, 500, replace=False)
            col.update(lo=float(p_lo - margin), hi=float(p_hi + margin),
                       median=float(np.median(v)), p25=float(p25), p75=float(p75),
                       edges=edges, shares=shares, sample=[float(x) for x in sample])
        elif kind == "categorical":
            col["shares"] = category_shares(s)
            mode = non_null.mode().iloc[0] if len(non_null) else None
            col["mode"] = mode.item() if hasattr(mode, "item") else mode
        columns[name] = col
    return {
        "dataset": cfg.get("dataset", "unknown"),
        "rows": int(len(df)),
        "duplicate_rate": float(df.duplicated().mean()),
        "columns": columns,
    }


def save_profile(profile: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


def load_profile(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
