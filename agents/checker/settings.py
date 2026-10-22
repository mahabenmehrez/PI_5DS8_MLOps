"""
Data Checker settings: thresholds, paths, reference profile, feature importance.

Sources, in order of preference (roadmap 6.3, "every dependency has a stand-in"):
- thresholds: agents.yaml / samples/agents.yaml (agents.checker.thresholds), else defaults;
- reference data and feature_importance.json: samples/reference/ from the Trainer (P4),
  else a stand-in built here from data/winequality-red.csv (or the 500-row sample) with
  the baseline's own ElasticNet pipeline;
- schema: samples/schema.py (WineBatch, rebuilt from `bounds` after a unit change).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from agents.checker.profiler import build_profile
from samples.schema import FEATURES

ROOT = Path(__file__).resolve().parents[2]
FEATURE_COLUMNS = list(FEATURES)
LABEL = "quality"

DEFAULT_THRESHOLDS = {
    "psi_warn": 0.10,
    "psi_alert": 0.25,
    "error_floor": 0.5,          # minimum weight of a data error on a model feature
    "decision_min_score": 0.5,   # level x weight; below this a finding is only logged
    "isolation_min_batches": 10, # normal batches needed before the Isolation Forest is used
    "compare_last_batches": 5,   # query.data_changed: window vs the 5 batches before it
    # unit changes that make sense for lab measurements (mg <-> g, g <-> kg, x10 dilution);
    # no miles/km here, so natural batch-to-batch variation can never look like a unit change
    "unit_factors": [10, 0.1, 100, 0.01, 1000, 0.001],
}


def node(column: str) -> str:
    """CSV column name -> dependency-map feature node, e.g. 'total sulfur dioxide' -> feature:total_sulfur_dioxide."""
    return "feature:" + column.replace(" ", "_")


def column_of(feature: str) -> str:
    """'total_sulfur_dioxide' or 'feature:total_sulfur_dioxide' -> 'total sulfur dioxide'."""
    name = feature.split(":", 1)[-1]
    for c in FEATURE_COLUMNS:
        if c.replace(" ", "_") == name or c == name:
            return c
    return name


def load_thresholds() -> dict:
    for candidate in (ROOT / "agents.yaml", ROOT / "samples" / "agents.yaml"):
        if candidate.exists():
            import yaml
            cfg = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            found = (cfg.get("agents", {}).get("checker", {}) or {}).get("thresholds", {})
            return {**DEFAULT_THRESHOLDS, **found}
    return dict(DEFAULT_THRESHOLDS)


def reference_frame() -> pd.DataFrame:
    """The Trainer's reference rows, else the full red-wine data, else the 500-row sample.
    The Trainer writes snake_case columns (fixed_acidity); the batches keep the CSV names."""
    trainer_ref = ROOT / "samples" / "reference" / "reference.parquet"
    if trainer_ref.exists():
        ref = pd.read_parquet(trainer_ref)
        return ref.rename(columns={c: column_of(c) for c in ref.columns})
    red = ROOT / "data" / "winequality-red.csv"
    if red.exists():
        return pd.read_csv(red, sep=";")
    return pd.read_parquet(ROOT / "samples" / "feature_sample.parquet")


def feature_importance(reference: pd.DataFrame) -> dict:
    """|scaled ElasticNet coefficient| normalised to max 1 (roadmap 6.4)."""
    trainer_file = ROOT / "samples" / "reference" / "feature_importance.json"
    if trainer_file.exists():
        raw = json.loads(trainer_file.read_text(encoding="utf-8"))
    else:
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("model", ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42))])
        pipe.fit(reference[FEATURE_COLUMNS], reference[LABEL])
        raw = dict(zip(FEATURE_COLUMNS, abs(pipe[-1].coef_)))
    raw = {column_of(k): float(v) for k, v in raw.items()}
    top = max(raw.values()) or 1.0
    return {k: round(v / top, 4) for k, v in raw.items()}


@dataclass
class CheckerSettings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("CHECKER_DATA_DIR",
                                                                        ROOT / "data" / "checker")))
    thresholds: dict = field(default_factory=load_thresholds)
    bounds: dict = field(default_factory=lambda: dict(FEATURES))
    schema_version: int = 1

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        reference = reference_frame()
        self.importance = feature_importance(reference)
        self.inspector_cfg = {
            "dataset": "wine-quality",
            "model_features": FEATURE_COLUMNS,
            "feature_importance": self.importance,
            "default_weight": 0.1,
            "ignore_columns": ["wine_id", LABEL],   # labels are the Monitor's / Trainer's signal
            "id_column": "wine_id",
            "thresholds": self.thresholds,
        }
        self.profile = build_profile(reference[FEATURE_COLUMNS], self.inspector_cfg)

    @property
    def profile_path(self) -> Path:
        return self.data_dir / f"profile_v{self.schema_version}.json"

    def save_profile(self) -> Path:
        self.profile_path.write_text(json.dumps({"profile": self.profile, "bounds": self.bounds,
                                                 "cfg": self.inspector_cfg}), encoding="utf-8")
        return self.profile_path
