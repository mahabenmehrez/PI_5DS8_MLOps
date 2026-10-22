"""
Data-layer fault functions (author P3), referenced from harness/faults/data/*.yaml.

A fault is applied to the feed, so it reaches both the batch and the live
requests (roadmap 9.1). inject(df, **target) returns a broken copy; revert
functions exist for the file format but are no-ops, because every batch is a
fresh copy of the source data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _rng(seed: int = 7) -> np.random.Generator:
    return np.random.default_rng(seed)


def rescale_column(df: pd.DataFrame, column: str, factor: float) -> pd.DataFrame:
    out = df.copy()
    out[column] = out[column] * factor
    return out


def null_values(df: pd.DataFrame, column: str, rate: float) -> pd.DataFrame:
    out = df.copy()
    out.loc[_rng().random(len(out)) < rate, column] = np.nan
    return out


def rename_column(df: pd.DataFrame, column: str, new_name: str) -> pd.DataFrame:
    return df.rename(columns={column: new_name})


def duplicate_rows(df: pd.DataFrame, rate: float) -> pd.DataFrame:
    extra = df.sample(frac=rate, random_state=3)
    return pd.concat([df, extra], ignore_index=True)


def add_column(df: pd.DataFrame, column: str, value=1.0) -> pd.DataFrame:
    out = df.copy()
    out[column] = value
    return out


def out_of_range(df: pd.DataFrame, column: str, rate: float, value: float) -> pd.DataFrame:
    out = df.copy()
    out.loc[_rng().random(len(out)) < rate, column] = value
    return out


def white_wine(df: pd.DataFrame, path: str = "data/winequality-white.csv") -> pd.DataFrame:
    """A valid batch from another population: should be a warning, not a quarantine."""
    source = ROOT / path
    if not source.exists():
        raise FileNotFoundError(f"{source} is needed for the white-wine fault")
    white = pd.read_csv(source, sep=";").sample(len(df), replace=True, random_state=5)
    white = white.reset_index(drop=True)
    if "wine_id" in df.columns:
        white["wine_id"] = df["wine_id"].to_numpy()
    return white[df.columns]


def restore_column(df: pd.DataFrame, **_) -> pd.DataFrame:
    return df
