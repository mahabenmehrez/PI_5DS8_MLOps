"""Train the wine-quality model and produce the reference files.

One reproducible training function used by every system we compare
(roadmap 2.3 and 10.1): Pipeline(StandardScaler, ElasticNet), logged to
MLflow and registered as the model ``wine-quality``.

Usage (from the repo root):
    python baseline/train.py                         # train on data/winequality-red.csv, alias champion
    python baseline/train.py --alias candidate       # register without touching champion
    python baseline/train.py --tracking-uri file:./mlruns_local   # no MLflow server needed

With DRY_RUN=1 the run goes to the ``dry_run`` experiment and nothing is
registered or aliased.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]  # repo root (this file is baseline/train.py)
DEFAULT_DATA = ROOT / "data" / "winequality-red.csv"
DEFAULT_REFERENCE_DIR = ROOT / "samples" / "reference"

MODEL_NAME = "wine-quality"
EXPERIMENT = "wine-quality"
DRY_RUN_EXPERIMENT = "dry_run"

# Column names exactly as in the UCI CSV and in app/main.py, in this order.
RAW_FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
# The same features as named in agents.yaml and on the board (feature:<name>).
FEATURES = [c.replace(" ", "_") for c in RAW_FEATURES]
TARGET = "quality"
# Names used by the week-1 refactor (P2); kept so existing imports keep working.
FEATURE_COLUMNS = RAW_FEATURES
TARGET_COLUMN = TARGET

HOLDOUT_ROWS = 400
SEED = 42
PARAMS = {"alpha": 0.01, "l1_ratio": 0.5}
PSI_BINS = 10
MIN_GROUP_SIZE = 50


# --------------------------------------------------------------------------
# Features (kept separate from prepare_features_serve in app/main.py on
# purpose: the Code Reviewer compares the two)
# --------------------------------------------------------------------------
def prepare_features_train(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Turn a raw wine table into model inputs (11 columns, fixed order) and labels.

    Accepts either the CSV column names ("total sulfur dioxide") or the
    snake_case names used by agents ("total_sulfur_dioxide").
    Self-contained on purpose: the Reviewer's feature_diff job compiles this
    function on its own, without the rest of the module.
    """
    feature_columns = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]
    renamed = df.rename(columns={c.replace(" ", "_"): c for c in feature_columns})
    missing = [c for c in feature_columns if c not in renamed.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    X = renamed[feature_columns].astype(float)
    y = renamed["quality"].astype(float) if "quality" in renamed.columns else None
    return X, y


def load_wines(path: str | Path = DEFAULT_DATA) -> pd.DataFrame:
    """Read a UCI wine CSV (semicolon or comma separated) and add wine_id (row number)."""
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df = df.rename(columns=lambda c: c.strip().strip('"'))
    df.insert(0, "wine_id", np.arange(len(df)))
    return df


# --------------------------------------------------------------------------
# Groups for the fairness check (roadmap 10.1)
# --------------------------------------------------------------------------
def assign_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add alcohol_band, acidity_band and quality_band columns."""
    out = pd.DataFrame(index=df.index)
    alc = df["alcohol"]
    out["alcohol_band"] = np.select([alc < 10, alc <= 11.5], ["<10", "10-11.5"], ">11.5")
    acid = df["fixed acidity"] if "fixed acidity" in df else df["fixed_acidity"]
    out["acidity_band"] = np.select([acid < 7.5, acid <= 9.5], ["<7.5", "7.5-9.5"], ">9.5")
    if TARGET in df:
        q = df[TARGET]
        out["quality_band"] = np.select([q <= 5, q == 6], ["<=5", "6"], ">=7")
    return out


def group_metrics(frame: pd.DataFrame, y_true: str, y_pred: str) -> dict:
    """Overall and per-group MAE. Groups under MIN_GROUP_SIZE are reported but not counted."""
    err = (frame[y_true] - frame[y_pred]).abs()
    result = {"overall": {"n": int(len(frame)), "mae": float(err.mean())}, "groups": {}}
    for group in ("alcohol_band", "acidity_band", "quality_band"):
        if group not in frame:
            continue
        result["groups"][group] = {
            str(band): {
                "n": int(len(e)),
                "mae": float(e.mean()),
                "counts": bool(len(e) >= MIN_GROUP_SIZE),
            }
            for band, e in err.groupby(frame[group])
        }
    return result


# --------------------------------------------------------------------------
# Reference files (roadmap 6.4)
# --------------------------------------------------------------------------
def psi_bins(X_train: pd.DataFrame, n_bins: int = PSI_BINS) -> dict:
    """Decile cut points per feature, from the training data, with the reference share per bin.

    ``cuts`` are the inner edges; bin i holds values in (cuts[i-1], cuts[i]],
    with the first and last bins open-ended. Features with many ties
    (citric acid has many zeros) get fewer than n_bins bins.
    """
    features = {}
    for raw, snake in zip(RAW_FEATURES, FEATURES):
        values = X_train[raw].to_numpy()
        cuts = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))
        idx = np.searchsorted(cuts, values, side="left")
        counts = np.bincount(idx, minlength=len(cuts) + 1)
        features[snake] = {
            "cuts": [float(c) for c in cuts],
            "ref_share": [float(c) for c in counts / counts.sum()],
        }
    return {"n_bins": n_bins, "source": "training rows", "features": features}


def feature_importance(model: Pipeline) -> dict:
    """Absolute ElasticNet coefficients on the standardised inputs."""
    coefs = np.abs(model.named_steps["model"].coef_)
    return {snake: float(c) for snake, c in zip(FEATURES, coefs)}


def write_reference(out_dir: Path, holdout: pd.DataFrame, model: Pipeline, X_train: pd.DataFrame) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = holdout.rename(columns=dict(zip(RAW_FEATURES, FEATURES)))
    ref.to_parquet(out_dir / "reference.parquet", index=False)
    files = {
        "feature_importance.json": feature_importance(model),
        "psi_bins.json": psi_bins(X_train),
        "group_metrics.json": group_metrics(holdout, TARGET, "prediction"),
    }
    for name, content in files.items():
        (out_dir / name).write_text(json.dumps(content, indent=2) + "\n")
    return files["group_metrics.json"]


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
@dataclass
class TrainResult:
    run_id: str | None
    model_uri: str | None
    version: str | None
    alias: str | None
    mae: float
    train_rows: int
    holdout_rows: int
    reference_dir: str
    group_metrics: dict = field(default_factory=dict)


def build_model(params: dict | None = None) -> Pipeline:
    p = {**PARAMS, **(params or {})}
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"], random_state=SEED)),
        ]
    )


def build_pipeline(alpha: float = PARAMS["alpha"], l1_ratio: float = PARAMS["l1_ratio"]) -> Pipeline:
    """Week-1 name for build_model (P2's refactor)."""
    return build_model({"alpha": alpha, "l1_ratio": l1_ratio})


def split(df: pd.DataFrame, holdout_rows: int = HOLDOUT_ROWS, seed: int = SEED):
    return train_test_split(df, test_size=holdout_rows, random_state=seed)


def train(
    df: pd.DataFrame,
    *,
    holdout_rows: int = HOLDOUT_ROWS,
    params: dict | None = None,
    alias: str | None = "champion",
    reference_dir: str | Path = DEFAULT_REFERENCE_DIR,
    tracking_uri: str | None = None,
    register: bool = True,
) -> TrainResult:
    """Fit on df minus a holdout, log to MLflow, register and alias the model.

    Under DRY_RUN=1 the run is logged to the dry_run experiment and the model
    is neither registered nor aliased.
    """
    import mlflow

    dry = os.environ.get("DRY_RUN") == "1"
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(DRY_RUN_EXPERIMENT if dry else EXPERIMENT)

    train_df, holdout_df = split(df, holdout_rows)
    X_train, y_train = prepare_features_train(train_df)
    X_hold, y_hold = prepare_features_train(holdout_df)

    model = build_model(params).fit(X_train, y_train)
    pred = model.predict(X_hold)
    mae = float(mean_absolute_error(y_hold, pred))

    holdout = holdout_df[["wine_id", *RAW_FEATURES, TARGET]].copy()
    holdout["prediction"] = pred
    holdout = pd.concat([holdout, assign_groups(holdout)], axis=1).reset_index(drop=True)

    reference_dir = Path(reference_dir)
    with mlflow.start_run() as run:
        mlflow.log_params({**PARAMS, **(params or {}), "train_rows": len(train_df), "holdout_rows": len(holdout_df), "seed": SEED})
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", float(np.sqrt(np.mean((y_hold - pred) ** 2))))
        # No signature on purpose: app/main.py sends plain lists, and a strict
        # float schema would reject integer inputs such as citric acid = 0.
        mlflow.sklearn.log_model(model, "model")
        if dry:
            reference_dir = Path(tempfile.mkdtemp(prefix="reference_"))
        gm = write_reference(reference_dir, holdout, model, X_train)
        mlflow.log_artifacts(str(reference_dir), artifact_path="reference")
        run_id = run.info.run_id

    version = None
    if register and not dry:
        mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        version = str(mv.version)
        if alias:
            mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, alias, version)

    return TrainResult(
        run_id=run_id,
        model_uri=f"models:/{MODEL_NAME}/{version}" if version else f"runs:/{run_id}/model",
        version=version,
        alias=alias if version else None,
        mae=mae,
        train_rows=len(train_df),
        holdout_rows=len(holdout_df),
        reference_dir=str(reference_dir),
        group_metrics=gm,
    )


def train_window(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    live_model_uri: str | None = f"models:/{MODEL_NAME}@champion",
    params: dict | None = None,
    tracking_uri: str | None = None,
    tags: dict | None = None,
) -> dict:
    """Fit on a window of clean batches and compare with the live model on held-out batches.

    Used by the Trainer (dry_run_jobs/train_compare.py). Logs the model, the
    reference files (built from the evaluation rows) and comparison.json to
    MLflow; registers nothing. Returns run_id, mae values and the eval frame.
    """
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(DRY_RUN_EXPERIMENT if os.environ.get("DRY_RUN") == "1" else EXPERIMENT)

    to_raw = dict(zip(FEATURES, RAW_FEATURES))
    train_df, eval_df = train_df.rename(columns=to_raw), eval_df.rename(columns=to_raw)
    X_train, y_train = prepare_features_train(train_df)
    X_eval, y_eval = prepare_features_train(eval_df)
    model = build_model(params).fit(X_train, y_train)

    frame = eval_df[["wine_id", *RAW_FEATURES, TARGET]].copy() if "wine_id" in eval_df else eval_df[[*RAW_FEATURES, TARGET]].copy()
    frame["prediction"] = model.predict(X_eval)
    frame["pred_live"] = mlflow.pyfunc.load_model(live_model_uri).predict(X_eval) if live_model_uri else np.nan
    frame = pd.concat([frame.reset_index(drop=True), assign_groups(frame).reset_index(drop=True)], axis=1)

    with mlflow.start_run(tags=tags) as run:
        mlflow.log_params({**PARAMS, **(params or {}), "train_rows": len(train_df), "eval_rows": len(eval_df), "seed": SEED})
        mlflow.log_metric("mae", float(mean_absolute_error(y_eval, frame["prediction"])))
        mlflow.sklearn.log_model(model, "model")
        ref_dir = Path(tempfile.mkdtemp(prefix="reference_"))
        write_reference(ref_dir, frame.drop(columns=["pred_live"]), model, X_train)
        mlflow.log_artifacts(str(ref_dir), artifact_path="reference")
        run_id = run.info.run_id
    return {"run_id": run_id, "frame": frame}


def register_run(run_id: str, alias: str = "candidate", tracking_uri: str | None = None, tags: dict | None = None) -> str:
    """Register a logged run as a new wine-quality version and point ``alias`` at it."""
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME, tags=tags)
    mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, alias, str(mv.version))
    return str(mv.version)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--alias", default="champion", help="alias to set on the new version ('' for none)")
    ap.add_argument("--reference-dir", default=str(DEFAULT_REFERENCE_DIR))
    ap.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args(argv)

    result = train(
        load_wines(args.data),
        alias=args.alias or None,
        reference_dir=args.reference_dir,
        tracking_uri=args.tracking_uri,
        register=not args.no_register,
    )
    summary = asdict(result)
    summary.pop("group_metrics")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
