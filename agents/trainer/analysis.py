"""Trainer statistics: no LLM here (roadmap 10.1, 10.4).

- psi / weighted_psi: input drift of a data window against the reference bins
- estimate_gain: v0 gain estimate that gates retraining
- choose_window: which clean batches to train and evaluate on
- compare / gate: overall and per-group MAE of a candidate vs the live model,
  with a paired bootstrap (500 resamples)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

GROUPS = ("alcohol_band", "acidity_band", "quality_band")
# The gate vetoes only on groups made from the inputs. quality_band is made from the label itself:
# when every score drops one point, wines move between quality bands, and the old model (which predicts
# near the mean) looks good on band 6 by chance. That group is reported, never a reason to reject.
VETO_GROUPS = ("alcohol_band", "acidity_band")
EPS = 1e-4


# --------------------------------------------------------------------------
# Drift
# --------------------------------------------------------------------------
def psi(values: np.ndarray, cuts: list[float], ref_share: list[float]) -> float:
    """Population stability index of values against reference bins (cut points + shares)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan")
    idx = np.searchsorted(np.asarray(cuts), values, side="left")
    cur = np.bincount(idx, minlength=len(cuts) + 1) / len(values)
    ref = np.clip(np.asarray(ref_share), EPS, None)
    cur = np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def drift_table(frame: pd.DataFrame, psi_bins: dict, importance: dict) -> dict:
    """PSI per feature and the importance-weighted PSI (sum(imp * psi) / sum(imp))."""
    per_feature = {}
    for feature, bins in psi_bins["features"].items():
        if feature in frame:
            per_feature[feature] = psi(frame[feature].to_numpy(), bins["cuts"], bins["ref_share"])
    total_imp = sum(importance.get(f, 0.0) for f in per_feature) or 1.0
    weighted = sum(importance.get(f, 0.0) * v for f, v in per_feature.items() if not math.isnan(v)) / total_imp
    return {"psi": per_feature, "weighted_psi": float(weighted)}


# --------------------------------------------------------------------------
# Gain estimate v0 (roadmap 10.4 W2 Wed): gain = a*weighted_psi + b*mae_rise + c*log(new_rows)
# --------------------------------------------------------------------------
DEFAULT_GAIN_WEIGHTS = {"a": 10.0, "b": 0.5, "c": 0.3}


def estimate_gain(weighted_psi: float, mae_rise_pct: float, new_rows: int, weights: dict | None = None) -> float:
    """Expected MAE improvement of a retrain, in percent.

    weighted_psi in PSI units, mae_rise_pct = how much worse the live model is
    than at training time (percent, negative clipped to 0), new_rows = clean
    labelled rows the live model has not seen. Hand-set weights, to be tuned.
    """
    w = {**DEFAULT_GAIN_WEIGHTS, **(weights or {})}
    return float(
        w["a"] * max(weighted_psi, 0.0)
        + w["b"] * max(mae_rise_pct, 0.0)
        + w["c"] * math.log1p(max(new_rows, 0) / 1000)
    )


def mae_rise(abs_errors: np.ndarray, mae_ref: float, *, resamples: int = 500, seed: int = 0) -> dict:
    """How much worse the live model is now than at training time, in percent.

    Returns the point estimate and the lower 95% bootstrap bound. The gain
    estimate uses the lower bound: on 500 rows the point estimate moves by
    about +-5% from sampling alone, which would otherwise trigger retrains on noise.
    """
    abs_errors = np.asarray(abs_errors, dtype=float)
    rng = np.random.default_rng(seed)
    boot = abs_errors[rng.integers(0, len(abs_errors), size=(resamples, len(abs_errors)))].mean(axis=1)
    to_pct = lambda m: float(100 * (m / mae_ref - 1))  # noqa: E731
    return {"mae_now": float(abs_errors.mean()), "rise_pct": to_pct(abs_errors.mean()),
            "rise_pct_low": to_pct(np.percentile(boot, 2.5))}


# --------------------------------------------------------------------------
# Window choice
# --------------------------------------------------------------------------
@dataclass
class Window:
    train_ids: list[str]
    eval_ids: list[str]
    excluded: dict[str, str] = field(default_factory=dict)  # batch_id -> reason

    def as_payload(self) -> dict:
        ids = self.train_ids + self.eval_ids
        return {
            "start": ids[0] if ids else None,
            "end": ids[-1] if ids else None,
            "batch_ids": self.train_ids,
            "eval_batch_ids": self.eval_ids,
            "excluded": self.excluded,
        }


def choose_window(
    batches: pd.DataFrame,
    *,
    size: int | None = None,
    eval_batches: int = 3,
    after: str | None = None,
    do_not_use: set[str] | None = None,
) -> Window:
    """Pick training and evaluation batches.

    batches: rows of the batches table (batch_id, status, labelled) in time order.
    Only passed / repaired batches count; do_not_use batches and anything up to
    ``after`` (the last batch before an expected change) are left out. The
    newest ``eval_batches`` labelled clean batches are held out for evaluation;
    the ``size`` batches before them (or all) are the training window.
    """
    excluded: dict[str, str] = {}
    keep = []
    seen_after = after is None
    for row in batches.itertuples(index=False):
        if not seen_after:
            excluded[row.batch_id] = "before_expected_change"
            seen_after = row.batch_id == after
            continue
        if row.status not in ("passed", "repaired"):
            excluded[row.batch_id] = row.status
        elif do_not_use and row.batch_id in do_not_use:
            excluded[row.batch_id] = "do_not_use"
        elif not row.labelled:
            excluded[row.batch_id] = "no_labels_yet"
        else:
            keep.append(row.batch_id)
    eval_ids = keep[-eval_batches:] if eval_batches else []
    train_ids = keep[: len(keep) - len(eval_ids)]
    if size:
        train_ids = train_ids[-size:]
    return Window(train_ids, eval_ids, excluded)


# --------------------------------------------------------------------------
# Comparison and gate
# --------------------------------------------------------------------------
def _paired_bootstrap(err_new: np.ndarray, err_live: np.ndarray, rng: np.random.Generator, n: int) -> tuple[float, float]:
    idx = rng.integers(0, len(err_new), size=(n, len(err_new)))
    diffs = err_new[idx].mean(axis=1) - err_live[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def compare(
    frame: pd.DataFrame,
    *,
    y: str = "quality",
    new: str = "pred_new",
    live: str = "pred_live",
    resamples: int = 500,
    min_group: int = 50,
    seed: int = 0,
) -> dict:
    """Overall and per-group MAE of new vs live on the same evaluation rows."""
    rng = np.random.default_rng(seed)
    err_new = (frame[y] - frame[new]).abs().to_numpy()
    err_live = (frame[y] - frame[live]).abs().to_numpy()

    def block(mask: np.ndarray) -> dict:
        en, el = err_new[mask], err_live[mask]
        lo, hi = _paired_bootstrap(en, el, rng, resamples)
        return {
            "n": int(mask.sum()),
            "mae_new": float(en.mean()),
            "mae_live": float(el.mean()),
            "delta": float(en.mean() - el.mean()),
            "delta_pct": float(100 * (en.mean() - el.mean()) / el.mean()) if el.mean() else 0.0,
            "delta_ci95": [lo, hi],
        }

    report = {"overall": block(np.ones(len(frame), dtype=bool)), "groups": {}, "resamples": resamples}
    for group in GROUPS:
        if group not in frame:
            continue
        report["groups"][group] = {}
        for band in sorted(frame[group].dropna().unique()):
            mask = (frame[group] == band).to_numpy()
            if mask.sum() < min_group:
                report["groups"][group][str(band)] = {"n": int(mask.sum()), "counts": False}
                continue
            report["groups"][group][str(band)] = {**block(mask), "counts": True}
    return report


def gate(report: dict, *, min_gain_pct: float = 2.0, max_group_worse_pct: float = 5.0) -> dict:
    """Pass when overall MAE improves by >= min_gain_pct and no counted group is clearly worse.

    A group fails when it is worse by more than max_group_worse_pct of its live
    MAE (point estimate) AND the paired bootstrap says the difference is real
    (lower 95% bound above 0). Only VETO_GROUPS can fail; a worse quality_band
    band is listed in ``label_groups_worse`` for the record.

    Calibration note: roadmap 10.1 starts from "upper 95% bound below +5%".
    Measured on 3 seeds x 3 window sizes, that rule flagged a "worse" group
    in 8 of 9 candidates trained on unchanged data (noise alone crosses +5%), and rejected
    the scenario D candidate for being 2-3% worse on low-alcohol wines while
    29% better on high-alcohol ones. This rule keeps F rejected (+82% on
    alcohol > 11.5) and lets D through.
    """
    overall = report["overall"]
    gain_pct = -overall["delta_pct"]
    failing, label_worse = [], []
    for group, bands in report["groups"].items():
        for band, stats in bands.items():
            if not stats.get("counts"):
                continue
            limit = max_group_worse_pct / 100 * stats["mae_live"]
            if stats["delta"] > limit and stats["delta_ci95"][0] > 0:
                (failing if group in VETO_GROUPS else label_worse).append({
                    "group": group, "band": band, "n": stats["n"],
                    "mae_live": round(stats["mae_live"], 4), "mae_candidate": round(stats["mae_new"], 4),
                    "delta_pct": round(stats["delta_pct"], 2),
                    "delta_ci95": [round(v, 4) for v in stats["delta_ci95"]], "limit": round(limit, 4),
                })
    return {
        "passed": gain_pct >= min_gain_pct and not failing,
        "overall_gain_pct": round(gain_pct, 3),
        "overall_ok": gain_pct >= min_gain_pct,
        "failing_groups": failing,
        "label_groups_worse": label_worse,
    }
