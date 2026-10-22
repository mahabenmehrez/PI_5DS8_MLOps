"""Model-layer faults (owner P4, roadmap 10.6 and 13.4).

Each inject function takes one replayed batch (wine_id, 11 features in
snake_case, quality), the index of the batch inside the fault's duration
(0-based) and the fault's ``target``; it returns the changed batch. They
change labels (what tasters say) or which rows reach a training window,
never the live inputs, so the Data Checker cannot see them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _mask(batch: pd.DataFrame, target: dict) -> pd.Series:
    mask = pd.Series(True, index=batch.index)
    if "alcohol_gt" in target:
        mask &= batch["alcohol"] > target["alcohol_gt"]
    if "alcohol_le" in target:
        mask &= batch["alcohol"] <= target["alcohol_le"]
    return mask


def shift_labels(batch: pd.DataFrame, step: int, target: dict) -> pd.DataFrame:
    """Concept drift: matching wines scored ``delta`` points differently.

    With ``ramp_batches`` the share of matching wines affected grows from
    1/ramp to 100% over that many batches (gradual drift).
    """
    out = batch.copy()
    mask = _mask(out, target)
    ramp = target.get("ramp_batches")
    if ramp:
        rng = np.random.default_rng(target.get("seed", 0) + step)
        mask &= rng.random(len(out)) < min(1.0, (step + 1) / ramp)
    out.loc[mask, "quality"] = out.loc[mask, "quality"] + target["delta"]
    return out


def drop_rows(batch: pd.DataFrame, step: int, target: dict) -> pd.DataFrame:
    """Group damage: matching wines never reach the batch (e.g. high-alcohol wines lost upstream)."""
    return batch[~_mask(batch, target)].copy()


def shuffle_labels(batch: pd.DataFrame, step: int, target: dict) -> pd.DataFrame:
    """Corrupted window: labels shuffled inside the batch (bad candidate if trained on it)."""
    out = batch.copy()
    rng = np.random.default_rng(target.get("seed", 0) + step)
    out["quality"] = rng.permutation(out["quality"].to_numpy())
    return out


def restore(batch: pd.DataFrame, step: int, target: dict) -> pd.DataFrame:
    """Revert: the replay writes clean batches again once the duration is over."""
    return batch
