"""Compares one batch to the reference profile and lists what is wrong.

Every check is generic (works on any tabular dataset). Dataset knowledge comes
only from the settings: model features, feature importance, optional Pandera
schema and optional domain rules written as pandas expressions.

Weighting (roadmap 9.1, "severity depends on the model"):
- drift is a change in the world: weighted by the feature's importance only,
  so a big shift on a feature the model barely uses stays a warning (scenario E);
- a data error (wrong unit, missing column, nulls, broken rows) on a model
  feature always counts at least `error_floor`: it corrupts live predictions
  and every future training set, whatever today's model weights are.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from agents.checker.profiler import OTHER, category_shares

# Ratios that usually mean "same quantity, different unit":
# x10 / x100 / x1000 (mg<->g, cents), miles<->km, minutes<->hours.
UNIT_FACTORS = [10, 0.1, 100, 0.01, 1000, 0.001, 1.609, 1 / 1.609, 60, 1 / 60]

DEFAULT_THRESHOLDS = {
    "psi_warn": 0.10,
    "psi_alert": 0.25,
    "null_increase_warn": 0.02,
    "null_increase_alert": 0.05,
    "out_of_range_alert": 0.01,
    "unseen_category_alert": 0.01,
    "rule_violation_alert": 0.001,
    "duplicate_increase_alert": 0.05,
    "unit_ratio_tolerance": 0.05,
    "error_floor": 0.5,
    "ks_p_max": 0.05,
}

WARN, ALERT, CRITICAL = 1, 2, 3
LEVEL_NAMES = {WARN: "warn", ALERT: "alert", CRITICAL: "critical"}
# Changes of population, not broken data: weighted by importance, never repaired.
DRIFT_CHECKS = {"drift", "range_shift"}


@dataclass
class Finding:
    check: str
    column: str | None
    level: int
    weight: float          # how much the column matters for the model (0-1)
    evidence: dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.level * self.weight

    def to_dict(self) -> dict:
        return {"check": self.check, "column": self.column, "level": LEVEL_NAMES[self.level],
                "weight": round(self.weight, 3), "score": round(self.score, 3),
                "evidence": self.evidence}


def psi(expected: list, actual: list) -> float:
    e = np.clip(np.asarray(expected, dtype=float), 1e-4, None)
    a = np.clip(np.asarray(actual, dtype=float), 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def numeric_shares(values: np.ndarray, edges: list) -> list:
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return (counts / max(counts.sum(), 1)).tolist()


def column_psi(series: pd.Series, ref: dict) -> float | None:
    non_null = series.dropna()
    if not len(non_null):
        return None
    if ref["kind"] == "numeric":
        return psi(ref["shares"], numeric_shares(non_null.to_numpy(dtype=float), ref["edges"]))
    if ref["kind"] == "categorical":
        keep = [k for k in ref["shares"] if k != OTHER]
        actual = category_shares(series, keep)
        return psi([ref["shares"][k] for k in actual], list(actual.values()))
    return None


def rule_holds(df: pd.DataFrame, expr: str) -> pd.Series:
    """True where the rule holds. Rows with a null in a column of the rule count as holding:
    missing values are the null check's job, not the rule's."""
    ok = df.eval(expr).astype(bool)
    used = [c for c in df.columns if c in expr]
    return ok | df[used].isna().any(axis=1)


def match_unit_factor(ratio: float, tolerance: float, factors=UNIT_FACTORS) -> float | None:
    if not np.isfinite(ratio) or ratio <= 0 or abs(np.log(ratio)) < np.log(1.2):
        return None
    for factor in factors:
        if abs(ratio / factor - 1) <= tolerance:
            return factor
    return None


class Inspector:
    def __init__(self, profile: dict, cfg: dict, schema=None):
        self.profile = profile
        self.cfg = cfg
        self.schema = schema
        self.t = {**DEFAULT_THRESHOLDS, **cfg.get("thresholds", {})}
        self.importance = cfg.get("feature_importance", {})
        self.model_features = set(cfg.get("model_features", self.importance))
        self.default_weight = cfg.get("default_weight", 0.1)
        self.schema_columns = set(getattr(schema, "columns", {}) or {})

    def weight(self, column: str | None, check: str) -> float:
        if column is None:
            return 1.0
        importance = float(self.importance.get(column, self.default_weight))
        if check in DRIFT_CHECKS or column not in self.model_features:
            return importance
        return max(importance, self.t["error_floor"])

    def find(self, check, column, level, **evidence) -> Finding:
        return Finding(check, column, level, self.weight(column, check), evidence)

    # ------------------------------------------------------------------ checks
    def inspect(self, df: pd.DataFrame) -> list[Finding]:
        findings = []
        if df.empty:
            return [self.find("empty_batch", None, CRITICAL, rows=0)]
        ref_cols = self.profile["columns"]
        ignore = set(self.cfg.get("ignore_columns", []))
        missing = [c for c in ref_cols if c not in df.columns]
        new = [c for c in df.columns if c not in ref_cols and c not in ignore]

        renamed = self.detect_renames(df, missing, new)
        for old, new_name in renamed.items():
            findings.append(self.find("renamed_column", old, ALERT, new_name=new_name))
        for c in missing:
            if c not in renamed:
                findings.append(self.find("missing_column", c, CRITICAL))
        for c in new:
            if c not in renamed.values():
                findings.append(self.find("new_column", c, WARN, dtype=str(df[c].dtype)))

        for name, ref in ref_cols.items():
            if name in df.columns:
                findings += self.check_column(df[name], ref)

        findings += self.check_schema(df.rename(columns={v: k for k, v in renamed.items()}))
        findings += self.check_rules(df)

        # With a record id, a duplicate is the same record twice (normal rate 0). Without one,
        # compare identical rows with the reference's own rate (wines can be identical).
        id_col = self.cfg.get("id_column")
        if id_col and id_col in df.columns:
            dup_rate, reference = float(df.duplicated().mean()), 0.0
        else:
            dup_rate, reference = float(df.duplicated().mean()), self.profile["duplicate_rate"]
        if dup_rate - reference > self.t["duplicate_increase_alert"]:
            findings.append(self.find("duplicates", None, ALERT, rate=round(dup_rate, 4),
                                      reference=round(reference, 4), by_id=bool(id_col and id_col in df.columns)))
        return findings

    def check_column(self, s: pd.Series, ref: dict) -> list[Finding]:
        name, out = s.name, []
        kind = ref["kind"]

        if kind == "numeric" and not pd.api.types.is_numeric_dtype(s):
            converted = pd.to_numeric(s, errors="coerce")
            ok_share = float(converted.notna().sum() / max(s.notna().sum(), 1))
            out.append(self.find("type_changed", name, ALERT, expected="numeric",
                                 got=str(s.dtype), convertible=round(ok_share, 3)))
            if ok_share < 0.95:
                return out
            s = converted

        null_rate = float(s.isna().mean())
        increase = null_rate - ref["null_rate"]
        if increase > self.t["null_increase_alert"]:
            out.append(self.find("null_spike", name, ALERT, rate=round(null_rate, 4),
                                 reference=round(ref["null_rate"], 4)))
        elif increase > self.t["null_increase_warn"]:
            out.append(self.find("null_spike", name, WARN, rate=round(null_rate, 4),
                                 reference=round(ref["null_rate"], 4)))

        non_null = s.dropna()
        if kind == "numeric" and len(non_null):
            v = non_null.to_numpy(dtype=float)
            median = float(np.median(v))
            factor = None
            if ref["median"]:
                ratio = median / ref["median"]
                # the median of a 100-row batch moves by up to ~25%: widen the tolerance for small n
                tolerance = max(self.t["unit_ratio_tolerance"], 3 / np.sqrt(len(v)))
                factor = match_unit_factor(ratio, tolerance, self.t.get("unit_factors", UNIT_FACTORS))
                # A unit change scales the whole distribution: the quartiles must move by the
                # same factor. A population change moves them unevenly.
                if factor and not self.same_scaling(v, ref, factor):
                    factor = None
                if factor:
                    out.append(self.find("unit_change", name, ALERT, median_ratio=round(ratio, 4),
                                         factor=factor))
            outside = float(np.mean((v < ref["lo"]) | (v > ref["hi"])))
            # in a 100-row batch one rare but real value is already 1%: ask for 3+ rows
            if outside > max(self.t["out_of_range_alert"], 3 / len(v)) and not factor:
                # Outside the reference range but inside the schema's physical limits (white
                # wines' sugar, say) is a new population, not broken data: impossible values are
                # the schema's job. Without a schema for the column, it stays an error.
                check = "range_shift" if name in self.schema_columns else "out_of_range"
                out.append(self.find(check, name, ALERT, rate=round(outside, 4),
                                     lo=round(ref["lo"], 4), hi=round(ref["hi"], 4)))
        elif kind == "categorical" and len(non_null):
            known = set(ref["shares"]) - {OTHER}
            if ref["shares"].get(OTHER, 0) < 0.001:   # closed set of values, e.g. zone ids
                unseen = float((~non_null.astype(str).isin(known)).mean())
                if unseen > self.t["unseen_category_alert"]:
                    out.append(self.find("unseen_categories", name, ALERT, rate=round(unseen, 4)))

        if kind in ("numeric", "categorical") and not any(f.check == "unit_change" for f in out):
            value = column_psi(s, ref)
            if value is not None:
                # PSI of two samples of the SAME distribution is about (bins-1)(1/n + 1/n_ref),
                # which is large for 100-row batches. Only the excess above that noise is drift.
                bins = len(ref["shares"])
                noise = (bins - 1) * (1 / max(len(non_null), 1) + 1 / self.profile["rows"])
                excess = value - noise
                evidence = {"psi": round(value, 3), "noise": round(noise, 3)}
                significant = True
                if kind == "numeric" and ref.get("sample"):
                    evidence["ks_p"] = round(float(ks_2samp(non_null.to_numpy(dtype=float),
                                                            ref["sample"]).pvalue), 4)
                    # PSI is noisy on small batches: a drift must also pass the KS test
                    significant = evidence["ks_p"] < self.t["ks_p_max"]
                if not significant:
                    pass
                elif excess > self.t["psi_alert"]:
                    out.append(self.find("drift", name, ALERT, **evidence))
                elif excess > self.t["psi_warn"]:
                    out.append(self.find("drift", name, WARN, **evidence))
        return out

    def check_schema(self, df: pd.DataFrame) -> list[Finding]:
        """Pandera range checks. Missing columns, nulls and types are reported by the
        checks above already, so only value-range failures are kept here."""
        if self.schema is None:
            return []
        try:
            self.schema.validate(df, lazy=True)
            return []
        except Exception as e:                   # pandera SchemaErrors
            cases = getattr(e, "failure_cases", None)
            if cases is None:
                return []
        out = []
        ranged = cases[cases["check"].astype(str).str.contains("range|greater|less", regex=True)]
        for column, group in ranged.groupby("column"):
            rate = len(group) / len(df)
            out.append(self.find("schema_range", column, ALERT, rate=round(rate, 4),
                                 pandera_check=str(group["check"].iloc[0])))
        return out

    def same_scaling(self, v: np.ndarray, ref: dict, factor: float) -> bool:
        tolerance = max(2 * self.t["unit_ratio_tolerance"], 3 / np.sqrt(len(v)))
        q25, q75 = np.quantile(v, [0.25, 0.75])
        for now, before in ((q25, ref.get("p25")), (q75, ref.get("p75"))):
            if before and abs(now / before / factor - 1) > tolerance:
                return False
        return True

    def check_rules(self, df: pd.DataFrame) -> list[Finding]:
        out = []
        for rule in self.cfg.get("rules", []):
            try:
                ok = rule_holds(df, rule["expr"])
            except Exception as e:   # a column of the rule is missing: already reported
                out.append(self.find("rule_not_evaluable", None, WARN, rule=rule["name"], error=str(e)[:80]))
                continue
            rate = float((~ok).mean())
            if rate > self.t["rule_violation_alert"]:
                out.append(self.find("rule_violation", rule.get("column"), ALERT, rule=rule["name"],
                                     rate=round(rate, 4), expr=rule["expr"]))
        return out

    def detect_renames(self, df, missing, new) -> dict:
        """A missing column and a new column with the same distribution = a rename."""
        renames = {}
        for old in missing:
            ref = self.profile["columns"][old]
            for candidate in new:
                if candidate in renames.values():
                    continue
                try:
                    value = column_psi(df[candidate], ref)
                except (TypeError, ValueError):   # incompatible type, not the same column
                    continue
                if value is not None and value < self.t["psi_alert"] + 0.2:
                    renames[old] = candidate
                    break
        return renames
