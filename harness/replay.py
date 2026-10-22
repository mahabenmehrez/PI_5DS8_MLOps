"""
Replay script (owner P3, roadmap 9.1 and 13.6).

Draws batches of 100 wines with replacement from the red-wine data, one per
simulated hour. For each batch it
  1. saves the file and adds a `pending` row to `batches`,
  2. sends the same wines (features only, with wine_id) to the API as live requests,
  3. writes their true quality to `labels`, available after labels.delay.
A data fault is applied to the feed, so it reaches the batch and the live requests.

Source: data/winequality-red.csv if present, else samples/feature_sample.parquet.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from shared import clock

ROOT = Path(__file__).resolve().parents[1]
LABEL = "quality"


def load_source() -> pd.DataFrame:
    csv = ROOT / "data" / "winequality-red.csv"
    if csv.exists():
        return pd.read_csv(csv, sep=";")
    return pd.read_parquet(ROOT / "samples" / "feature_sample.parquet")


def http_sender(api_url: str) -> Callable[[dict], None]:
    """Sends one replayed wine to the baseline API (list of 11 numbers, as today)."""
    import httpx
    client = httpx.Client(base_url=api_url, timeout=5)

    def send(features: dict) -> None:
        values = [v for k, v in features.items() if k != "wine_id"]
        client.post("/predict", json={"data": values, "wine_id": features.get("wine_id")})
    return send


class Replay:
    def __init__(self, tables, batch_dir: Path, source: Optional[pd.DataFrame] = None,
                 batch_size: int = 100, seed: int = 0, label_delay: timedelta = timedelta(days=1),
                 batch_every: timedelta = timedelta(hours=1), fault_runner=None,
                 send_request: Optional[Callable[[dict], None]] = None):
        self.tables = tables
        self.batch_dir = Path(batch_dir)
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.source = (source if source is not None else load_source()).reset_index(drop=True)
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.label_delay = label_delay
        self.batch_every = batch_every
        self.fault_runner = fault_runner
        self.send_request = send_request
        self.index = 0
        self.next_wine_id = 1
        self.sent_requests = 0

    def next_batch(self) -> str:
        """Advances the clock one batch period and publishes the next batch."""
        if clock.mode() == "frozen":
            clock.advance(self.batch_every)
        self.index += 1
        batch_id = f"batch-{self.index:04d}"
        rows = self.source.iloc[self.rng.integers(0, len(self.source), self.batch_size)].copy()
        rows.insert(0, "wine_id", range(self.next_wine_id, self.next_wine_id + len(rows)))
        self.next_wine_id += len(rows)
        rows = rows.reset_index(drop=True)
        if self.fault_runner:      # model faults (concept drift): what tasters say changes, not the inputs
            rows = self.fault_runner.apply(self.index, rows, "labels", batch_id)

        batch = rows
        if self.fault_runner:
            batch = self.fault_runner.apply(self.index, rows, "batch", batch_id)
        path = self.batch_dir / f"{batch_id}.parquet"
        batch.to_parquet(path, index=False)
        self.tables.add_batch(batch_id, str(path), len(batch))

        requests = rows.drop(columns=[LABEL], errors="ignore")
        if self.fault_runner:
            requests = self.fault_runner.apply(self.index, requests, "requests", batch_id)
        if self.send_request:
            for record in requests.to_dict("records"):
                self.send_request(record)
        self.sent_requests += len(requests)

        available = (clock.now() + self.label_delay).isoformat().replace("+00:00", "Z")
        self.tables.add_labels([(w, q, available) for w, q in zip(rows["wine_id"], rows[LABEL])])
        return batch_id
