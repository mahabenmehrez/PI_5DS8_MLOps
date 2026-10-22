"""Generic repairs. Each one is a pure function: DataFrame in, new DataFrame out.

plan_repairs() turns findings into a list of repair steps. Findings with no
safe repair (missing column, drift, big null spike...) are returned apart, so
the policy knows the batch cannot be fixed automatically.
"""
import json

import pandas as pd

from agents.checker.checks import rule_holds

MAX_IMPUTE_NULL_RATE = 0.20
MAX_DROP_ROWS_RATE = 0.05


def rename(df, old, new):
    return df.rename(columns={new: old})


def rescale(df, column, factor):
    out = df.copy()
    out[column] = out[column] / factor
    return out


def cast_numeric(df, column):
    out = df.copy()
    out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def impute(df, column, value):
    out = df.copy()
    out[column] = out[column].fillna(value)
    return out


def drop_duplicates(df):
    return df.drop_duplicates()


def drop_out_of_range(df, column, lo, hi):
    return df[df[column].isna() | df[column].between(lo, hi)]


def drop_rule_violations(df, expr):
    return df[rule_holds(df, expr)]


REPAIRS = {
    "rename": rename, "rescale": rescale, "cast_numeric": cast_numeric, "impute": impute,
    "drop_duplicates": drop_duplicates, "drop_out_of_range": drop_out_of_range,
    "drop_rule_violations": drop_rule_violations,
}


def plan_repairs(findings, profile, bounds: dict | None = None) -> tuple[list[dict], list]:
    """Returns (repair steps, findings that no repair can fix). Only alert-level findings are
    planned. `bounds` are the schema's physical ranges, {column: (lo, hi)}."""
    steps, unfixable = [], []
    bounds = bounds or {}
    cols = profile["columns"]
    for f in findings:
        if f.level < 2:
            continue
        e, c = f.evidence, f.column
        if f.check == "renamed_column":
            steps.append({"repair": "rename", "params": {"old": c, "new": e["new_name"]}})
        elif f.check == "unit_change":
            steps.append({"repair": "rescale", "params": {"column": c, "factor": e["factor"]}})
        elif f.check == "type_changed" and e.get("convertible", 0) >= 0.95:
            steps.append({"repair": "cast_numeric", "params": {"column": c}})
        elif f.check == "null_spike" and e["rate"] <= MAX_IMPUTE_NULL_RATE:
            ref = cols[c]
            value = ref.get("median") if ref["kind"] == "numeric" else ref.get("mode")
            steps.append({"repair": "impute", "params": {"column": c, "value": value}})
        elif f.check == "duplicates":
            steps.append({"repair": "drop_duplicates", "params": {}})
        elif f.check == "out_of_range" and e["rate"] <= MAX_DROP_ROWS_RATE:
            steps.append({"repair": "drop_out_of_range",
                          "params": {"column": c, "lo": cols[c]["lo"], "hi": cols[c]["hi"]}})
        elif f.check == "rule_violation" and e["rate"] <= MAX_DROP_ROWS_RATE:
            steps.append({"repair": "drop_rule_violations", "params": {"expr": e["expr"]}})
        elif f.check == "schema_range" and e["rate"] <= MAX_DROP_ROWS_RATE and c in bounds:
            lo, hi = bounds[c]
            steps.append({"repair": "drop_out_of_range", "params": {"column": c, "lo": lo, "hi": hi}})
        elif f.check in ("drift", "range_shift"):
            continue        # a real change in the world, not broken data: not ours to "fix"
        else:
            unfixable.append(f)
    # A wrong name, type or unit also breaks ranges and rules. Fix the column first and
    # let the next inspection round decide about rows, instead of dropping good rows now.
    if any(s["repair"] in STRUCTURAL for s in steps):
        steps = [s for s in steps if s["repair"] not in ROW_LEVEL]
        unfixable = [f for f in unfixable if f.check not in ROW_CHECKS]
    order = ["rename", "cast_numeric", "rescale", "drop_duplicates", "impute",
             "drop_out_of_range", "drop_rule_violations"]
    unique = {json.dumps(s, sort_keys=True): s for s in steps}
    steps = sorted(unique.values(), key=lambda s: order.index(s["repair"]))
    return steps, unfixable


STRUCTURAL = {"rename", "cast_numeric", "rescale"}
ROW_LEVEL = {"drop_out_of_range", "drop_rule_violations"}
ROW_CHECKS = {"out_of_range", "rule_violation", "schema_range"}


def apply_repairs(df, steps) -> pd.DataFrame:
    for step in steps:
        df = REPAIRS[step["repair"]](df, **step["params"])
    return df
