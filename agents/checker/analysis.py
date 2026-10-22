"""
Statistics and checks of the Data Checker (no LLM here, roadmap 5.2).

analyze_batch() turns one batch into a BatchReport: findings (schema, nulls,
ranges, unit changes, renames, duplicates, PSI/KS drift), the batch statistics
for the batch_stats table, an Isolation Forest score on those statistics, the
likely cause and the repair plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from agents.checker.checks import ALERT, DRIFT_CHECKS, WARN, Finding, Inspector
from agents.checker.repairs import apply_repairs, plan_repairs

# Errors in the feed that also reach the live API (the same values go to /predict):
# the incident is data-caused, so the Checker claims it and asks for a hold (scenario C).
LIVE_IMPACT_CHECKS = {"unit_change", "missing_column", "type_changed"}

CAUSES = {                       # finding -> likely_cause of the query.data_changed answer
    "unit_change": "unit_change",
    "missing_column": "schema_change", "renamed_column": "schema_change",
    "new_column": "schema_change", "type_changed": "schema_change",
    "schema_range": "schema_change", "out_of_range": "schema_change",
    "null_spike": "missing_values",
    "empty_batch": "late_batch",
    "drift": "valid_shift", "range_shift": "valid_shift", "batch_anomaly": "valid_shift",
}
CAUSE_PRIORITY = ["unit_change", "schema_change", "missing_values", "late_batch", "valid_shift"]


@dataclass
class BatchReport:
    batch_id: str
    rows: int
    findings: list[Finding]
    actionable: list[Finding]
    steps: list[dict]
    unfixable: list[Finding]
    stats: list[dict]
    psi: dict = field(default_factory=dict)
    isolation: Optional[dict] = None

    @property
    def drift_only(self) -> bool:
        return bool(self.actionable) and all(f.check in DRIFT_CHECKS for f in self.actionable)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.actionable if f.level >= ALERT and f.check not in DRIFT_CHECKS]

    @property
    def live_impact(self) -> list[Finding]:
        """Errors that also reach the live API: a unit or type change, a missing column, or a
        model feature invalid on most rows."""
        return [f for f in self.errors if f.check in LIVE_IMPACT_CHECKS
                or (f.check == "schema_range" and f.evidence.get("rate", 0) >= 0.5)]

    @property
    def cause(self) -> str:
        return likely_cause(self.actionable)

    def failed_checks(self) -> list[str]:
        return [f"{f.check}:{f.column}" if f.column else f.check for f in self.actionable]


def likely_cause(findings: list) -> str:
    """Findings or plain check names -> the most specific cause."""
    found = {CAUSES.get(getattr(f, "check", f)) for f in findings} - {None}
    return next((c for c in CAUSE_PRIORITY if c in found), "none")


def batch_stats(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    out = []
    for c in columns:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        v = s.dropna()
        q = v.quantile([0.01, 0.5, 0.99]) if len(v) else pd.Series([np.nan] * 3, index=[0.01, 0.5, 0.99])
        out.append({"column": c, "null_rate": float(s.isna().mean()),
                    "mean": float(v.mean()) if len(v) else None,
                    "median": float(q[0.5]), "p1": float(q[0.01]), "p99": float(q[0.99])})
    return out


class StatsAnomaly:
    """Isolation Forest on the statistics vector of a batch (median and null rate per feature).

    Trained once on the first N normal batches. A batch is an anomaly when it scores lower
    than every training batch, which keeps false alarms close to zero on normal days."""

    def __init__(self, columns: list[str], min_batches: int = 10):
        self.columns = columns
        self.min_batches = min_batches
        self.model: Optional[IsolationForest] = None
        self.floor = None

    def vector(self, stats: list[dict]) -> list[float]:
        by_col = {s["column"]: s for s in stats}
        return [by_col.get(c, {}).get(k) or 0.0 for c in self.columns for k in ("median", "null_rate")]

    def fit(self, stats_table: pd.DataFrame, normal_batches: list[str]) -> bool:
        if self.model is not None or len(normal_batches) < self.min_batches:
            return self.model is not None
        train = []
        for b in normal_batches[: self.min_batches]:
            rows = stats_table[stats_table.batch_id == b].rename(columns={"column_name": "column"})
            train.append(self.vector(rows.to_dict("records")))
        self.model = IsolationForest(n_estimators=200, random_state=0).fit(train)
        self.floor = float(self.model.score_samples(train).min())
        return True

    def score(self, stats: list[dict]) -> Optional[dict]:
        if self.model is None:
            return None
        s = float(self.model.score_samples([self.vector(stats)])[0])
        return {"score": round(s, 4), "floor": round(self.floor, 4), "anomaly": bool(s < self.floor - 0.02)}


def analyze_batch(batch_id: str, df: pd.DataFrame, inspector: Inspector, profile: dict,
                  bounds: dict, columns: list[str], min_score: float,
                  anomaly: Optional[StatsAnomaly] = None) -> BatchReport:
    findings = inspector.inspect(df)
    stats = batch_stats(df, columns)
    iso = anomaly.score(stats) if anomaly else None
    if iso and iso["anomaly"]:
        findings.append(Finding("batch_anomaly", None, WARN, 1.0, iso))
    # Importance decides how serious a finding is, but a significant drift is always worth a
    # warning, even on a feature the model barely uses (scenario E: warn, never quarantine).
    actionable = [f for f in findings if f.score >= min_score
                  or (f.check in DRIFT_CHECKS and f.level >= ALERT)]
    steps, unfixable = plan_repairs(actionable, profile, bounds)
    psi = {f.column: f.evidence["psi"] for f in findings if f.check == "drift"}
    return BatchReport(batch_id, len(df), findings, actionable, steps, unfixable, stats, psi, iso)


def repair_until_clean(df: pd.DataFrame, steps: list[dict], inspector: Inspector, profile: dict,
                       bounds: dict, min_score: float, max_rounds: int = 3):
    """Apply, re-inspect, plan again. Returns (df, steps applied, remaining blocking findings)."""
    applied, remaining = [], []
    for _ in range(max_rounds):
        df = apply_repairs(df, steps)
        applied += steps
        remaining = [f for f in inspector.inspect(df)
                     if f.score >= min_score and f.level >= ALERT and f.check not in DRIFT_CHECKS]
        if not remaining:
            return df, applied, []
        steps, unfixable = plan_repairs(remaining, profile, bounds)
        if unfixable or not steps:
            break
    return df, applied, remaining
