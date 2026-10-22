"""What the Trainer reads and writes in board.db (roadmap 5.11).

Reads   batches (batch_id, path, status, ...)   written by the replay script and the Data Checker
        labels  (wine_id, quality, available_at) written by the label feeder
Writes  trainer_runs                             the Trainer's own state (last training, decisions)

The table layout follows roadmap 5.11; create_tables() makes them when the
real shared/tables.py is not there yet (solo runs, tests).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY, path TEXT, status TEXT, checked_at TEXT, reason TEXT);
CREATE TABLE IF NOT EXISTS labels (
    wine_id INTEGER PRIMARY KEY, quality REAL, available_at TEXT);
CREATE TABLE IF NOT EXISTS trainer_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, run_id TEXT, incident_id TEXT,
    decision TEXT, version TEXT, train_ids TEXT, eval_ids TEXT, last_batch TEXT,
    expected_gain REAL, overall_gain_pct REAL, detail TEXT);
"""


def now_iso() -> str:
    try:
        from shared import clock  # owner P3

        return clock.now().isoformat()
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class TrainerStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._cache: dict[str, pd.DataFrame] = {}

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Commit and CLOSE on exit (sqlite3's own `with` only commits; on Windows an open file can't be deleted)."""
        con = sqlite3.connect(self.db_path, timeout=10)
        try:
            con.execute("PRAGMA journal_mode=WAL")
            with con:
                yield con
        finally:
            con.close()

    def create_tables(self) -> None:
        with self.connect() as con:
            con.executescript(SCHEMA)

    # ---- reads ----------------------------------------------------------
    def _batch_file(self, path: str) -> pd.DataFrame:
        if path not in self._cache:
            p = Path(path)
            frame = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            self._cache[path] = frame.rename(columns=lambda c: c.strip().replace(" ", "_"))
        return self._cache[path]

    def batches(self, now: str | None = None) -> pd.DataFrame:
        """All batches in arrival order, with a ``labelled`` flag (every wine has a label by ``now``)."""
        now = now or now_iso()
        with self.connect() as con:
            frame = pd.read_sql("SELECT batch_id, path, status, reason FROM batches ORDER BY batch_id", con)
            labels = pd.read_sql("SELECT wine_id FROM labels WHERE available_at <= ?", con, params=(now,))
        known = set(labels["wine_id"])
        frame["labelled"] = [set(self._batch_file(p)["wine_id"]) <= known for p in frame["path"]]
        return frame

    def rows(self, batch_ids: list[str], now: str | None = None) -> pd.DataFrame:
        """Feature rows of these batches joined with the labels available at ``now``."""
        now = now or now_iso()
        if not batch_ids:
            return pd.DataFrame()
        with self.connect() as con:
            marks = ",".join("?" * len(batch_ids))
            paths = con.execute(f"SELECT batch_id, path FROM batches WHERE batch_id IN ({marks})", batch_ids).fetchall()
            labels = pd.read_sql("SELECT wine_id, quality FROM labels WHERE available_at <= ?", con, params=(now,))
        parts = [self._batch_file(path).assign(batch_id=bid) for bid, path in sorted(paths)]
        frame = pd.concat(parts, ignore_index=True).drop(columns=["quality"], errors="ignore")
        return frame.merge(labels, on="wine_id", how="inner")

    def last_training(self) -> dict | None:
        with self.connect() as con:
            con.row_factory = sqlite3.Row
            row = con.execute(
                "SELECT * FROM trainer_runs WHERE decision IN ('refuse', 'retrain', 'forwarded', 'rejected') ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    # ---- writes ---------------------------------------------------------
    def record(self, **fields) -> None:
        from shared.dry_run import guard_real_write

        guard_real_write("trainer_runs")
        for key in ("train_ids", "eval_ids", "detail"):
            if key in fields and not isinstance(fields[key], str):
                fields[key] = json.dumps(fields[key], default=str)
        fields.setdefault("time", now_iso())
        cols = ",".join(fields)
        with self.connect() as con:
            con.execute(f"INSERT INTO trainer_runs ({cols}) VALUES ({','.join('?' * len(fields))})", list(fields.values()))
