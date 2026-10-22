"""
Shared tables in board.db (roadmap 5.11). Messages announce, tables hold state:
a restarted agent catches up by reading these tables.

This file holds the tables of the data path (owner P3): batches, batch_stats,
labels and faults. predictions and release_state (P5) can be added here the
same way.

When DRY_RUN=1 (inside a dry-run job) every write raises, so a dry run can
never change the real tables.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Iterable, Optional

import pandas as pd

from shared import clock

# `path` is the file to train on: the original batch, or the repaired copy once the Checker
# repaired it (the original then stays in `raw_path`). The Trainer reads `path` only.
BATCH_COLUMNS = {"batch_id": "TEXT PRIMARY KEY", "path": "TEXT", "status": "TEXT",
                 "rows": "INTEGER", "created_at": "TEXT", "checked_at": "TEXT", "reason": "TEXT",
                 "clean_path": "TEXT", "raw_path": "TEXT", "schema_version": "INTEGER",
                 "findings": "TEXT"}

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS batches (
    {", ".join(f"{name} {kind}" for name, kind in BATCH_COLUMNS.items())});
CREATE TABLE IF NOT EXISTS batch_stats (
    batch_id TEXT, column_name TEXT, null_rate REAL, mean REAL, median REAL,
    p1 REAL, p99 REAL, PRIMARY KEY (batch_id, column_name));
CREATE TABLE IF NOT EXISTS labels (
    wine_id INTEGER PRIMARY KEY, quality REAL, available_at TEXT);
CREATE TABLE IF NOT EXISTS claims (
    incident_id TEXT, node TEXT, agent TEXT, claimed_at TEXT);
CREATE TABLE IF NOT EXISTS faults (
    fault_id TEXT, layer TEXT, target TEXT, injected_at TEXT, first_batch TEXT,
    expected TEXT, run_id TEXT);
"""

STATUSES = ("pending", "passed", "repaired", "quarantined")


class DryRunWriteError(RuntimeError):
    pass


class Tables:
    def __init__(self, path: str = "board.db"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        if path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")   # six agents writing at once
        self.db.executescript(SCHEMA)
        self._add_missing_batch_columns()

    def _add_missing_batch_columns(self) -> None:
        """The Trainer creates a smaller batches table (agents/trainer/data.py) when it starts
        first; add our columns to it instead of failing on the first write."""
        existing = {row[1] for row in self.db.execute("PRAGMA table_info(batches)")}
        for name, kind in BATCH_COLUMNS.items():
            if name not in existing:
                self.db.execute(f"ALTER TABLE batches ADD COLUMN {name} {kind.split()[0]}")
        self.db.commit()

    def _write(self, sql: str, params: Iterable = ()) -> None:
        if os.environ.get("DRY_RUN") == "1":
            raise DryRunWriteError("tables are read-only inside a dry run")
        self.db.execute(sql, tuple(params))
        self.db.commit()

    # ---- batches ----------------------------------------------------------------
    def add_batch(self, batch_id: str, path: str, rows: int) -> None:
        self._write("INSERT OR REPLACE INTO batches (batch_id, path, status, rows, created_at) "
                    "VALUES (?, ?, 'pending', ?, ?)", (batch_id, path, rows, clock.iso()))

    def set_batch_status(self, batch_id: str, status: str, reason: str = "",
                         clean_path: Optional[str] = None, schema_version: int = 1,
                         findings: Optional[list] = None) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown batch status {status!r}")
        self._write("UPDATE batches SET status=?, checked_at=?, reason=?, clean_path=?, "
                    "schema_version=?, findings=? WHERE batch_id=?",
                    (status, clock.iso(), reason, clean_path, schema_version,
                     json.dumps(findings or []), batch_id))

    def use_repaired_file(self, batch_id: str, repaired_path: str) -> None:
        """Training must use the repaired copy: `path` points to it, the original is kept
        in `raw_path`."""
        self._write("UPDATE batches SET raw_path=COALESCE(raw_path, path), path=? WHERE batch_id=?",
                    (repaired_path, batch_id))

    def batch(self, batch_id: str) -> Optional[dict]:
        row = self.db.execute("SELECT * FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
        return dict(row) if row else None

    def batches(self, status: Optional[str] = None) -> list[dict]:
        if status:
            cur = self.db.execute("SELECT * FROM batches WHERE status=? ORDER BY batch_id", (status,))
        else:
            cur = self.db.execute("SELECT * FROM batches ORDER BY batch_id")
        return [dict(r) for r in cur]

    def pending_batches(self) -> list[dict]:
        return self.batches("pending")

    # ---- batch_stats ------------------------------------------------------------
    def add_batch_stats(self, batch_id: str, stats: list[dict]) -> None:
        for s in stats:
            self._write("INSERT OR REPLACE INTO batch_stats VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (batch_id, s["column"], s["null_rate"], s["mean"], s["median"],
                         s["p1"], s["p99"]))

    def batch_stats(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM batch_stats ORDER BY batch_id", self.db)

    # ---- labels -----------------------------------------------------------------
    def add_labels(self, rows: list[tuple[int, float, str]]) -> None:
        for wine_id, quality, available_at in rows:
            self._write("INSERT OR REPLACE INTO labels VALUES (?, ?, ?)",
                        (int(wine_id), float(quality), available_at))

    def labels_available(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM labels WHERE available_at <= ?", self.db,
                                 params=(clock.iso(),))

    # ---- claims: which agent owns which incident, kept across restarts --------
    def record_claim(self, incident_id: str, node: str, agent: str) -> None:
        self._write("INSERT INTO claims VALUES (?, ?, ?, ?)", (incident_id, node, agent, clock.iso()))

    def recent_claim(self, nodes: list[str], since: str) -> Optional[str]:
        """Incident id of a claim on one of these nodes made at or after `since`, else None."""
        if not nodes:
            return None
        marks = ",".join("?" * len(nodes))
        row = self.db.execute(f"SELECT incident_id FROM claims WHERE node IN ({marks}) AND claimed_at >= ? "
                              "ORDER BY claimed_at DESC LIMIT 1", (*nodes, since)).fetchone()
        return row[0] if row else None

    # ---- faults (harness only; agents never read this table) ------------------
    def record_fault(self, fault_id: str, layer: str, target: dict, first_batch: str,
                     expected: dict, run_id: str = "dev") -> None:
        self._write("INSERT INTO faults VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (fault_id, layer, json.dumps(target), clock.iso(), first_batch,
                     json.dumps(expected), run_id))

    def faults(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM faults")]
