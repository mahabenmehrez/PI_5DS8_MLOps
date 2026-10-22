"""Temporary stand-ins for shared pieces other owners deliver in week 1.

- InMemoryBoard: stands in for shared/board.py (owner P6) in unit tests
- SqliteBoard:   the same API on board.db, so several agents in several
                 processes can talk (solo and integration runs)
- BaseAgent:     stands in for shared/base_agent.py (owner P3); only the
                 message-handling half of the 7-step loop
- new_message:   envelope builder matching roadmap 5.5 until shared/messages.py lands

They follow the APIs in roadmap 5.7 and 5.8 so the Trainer can switch to the
real modules by changing imports only. Delete this file once those merge.
"""

from __future__ import annotations

import fnmatch
import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from typing import Any

REQUIRED = ("id", "time", "type", "kind", "sender", "to", "topic", "level", "summary", "evidence", "payload", "status", "run_id")
TYPES = {"finding", "question", "answer", "request", "claim"}


def _now() -> str:
    try:
        from shared import clock

        return clock.now().isoformat()
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_message(**fields: Any) -> dict:
    msg = {
        "id": "msg-" + uuid.uuid4().hex[:12],
        "time": _now(),
        "to": [],
        "incident_id": None,
        "reply_to": None,
        "level": 0,
        "confidence": None,
        "evidence": {},
        "payload": {},
        "status": "open",
        "expires_at": None,
        "run_id": "dev",
        **fields,
    }
    missing = [k for k in REQUIRED if k not in msg]
    if missing:
        raise ValueError(f"message is missing {missing}")
    if msg["type"] not in TYPES:
        raise ValueError(f"bad message type {msg['type']!r}")
    if len(msg["summary"]) > 160:
        raise ValueError("summary is longer than 160 characters")
    from shared.messages import Message  # the team's envelope: fail here, not on the real board

    Message(**{k: v for k, v in msg.items() if k in Message.model_fields})
    return msg


class InMemoryBoard:
    def __init__(self, subscriptions: dict[str, list[str]] | None = None):
        self.messages: list[dict] = []
        self.cursors: dict[str, int] = {}
        self.subscriptions = subscriptions or {}
        self.heartbeats: dict[str, str] = {}

    def post(self, msg: dict) -> str:
        msg = new_message(**msg)
        self.messages.append(msg)
        return msg["id"]

    def _for(self, agent: str, msg: dict) -> bool:
        if msg["sender"] == agent:
            return False
        if msg["to"]:
            return agent in msg["to"]
        return any(fnmatch.fnmatch(msg["topic"], pat) for pat in self.subscriptions.get(agent, []))

    def read(self, agent: str) -> list[dict]:
        start = self.cursors.get(agent, 0)
        self.cursors[agent] = len(self.messages)
        return [m for m in self.messages[start:] if self._for(agent, m)]

    def answer(self, original: dict, **fields: Any) -> str:
        original["status"] = "answered"
        return self.post(
            {
                "type": "answer",
                "kind": original["kind"],
                "to": [original["sender"]],
                "topic": original["topic"],
                "incident_id": original.get("incident_id"),
                "reply_to": original["id"],
                "run_id": original.get("run_id", "dev"),
                **fields,
            }
        )

    def thread(self, incident_id: str) -> list[dict]:
        return [m for m in self.messages if m.get("incident_id") == incident_id]

    def heartbeat(self, agent: str) -> None:
        self.heartbeats[agent] = _now()


# roadmap 5.13; agents.yaml agents.<name>.subscriptions overrides these
DEFAULT_SUBSCRIPTIONS = {
    "checker": ["dataset:*", "feature:*", "expected_change:*"],
    "trainer": ["dataset:*", "model:*", "expected_change:*"],
    "deployer": ["model:*", "deployment:*", "expected_change:*"],
    "monitor": ["feature:*", "model:*", "deployment:*", "expected_change:*"],
    "reviewer": ["code:*", "feature:*"],
    "setup": ["pipeline:*"],
}

BOARD_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, time TEXT, type TEXT, kind TEXT,
    sender TEXT, "to" TEXT, topic TEXT, incident_id TEXT, reply_to TEXT, level INTEGER, confidence REAL,
    summary TEXT, evidence TEXT, payload TEXT, status TEXT, expires_at TEXT, run_id TEXT);
CREATE TABLE IF NOT EXISTS cursors (agent TEXT PRIMARY KEY, last_seq INTEGER);
CREATE TABLE IF NOT EXISTS heartbeats (agent TEXT PRIMARY KEY, time TEXT);
"""
COLUMNS = ("id", "time", "type", "kind", "sender", "to", "topic", "incident_id", "reply_to", "level", "confidence",
           "summary", "evidence", "payload", "status", "expires_at", "run_id")
JSON_COLUMNS = ("to", "evidence", "payload")


def load_subscriptions(path=None) -> dict[str, list[str]]:
    """agents.<name>.subscriptions from agents.yaml, else the roadmap defaults."""
    import os
    from pathlib import Path

    subs = {k: list(v) for k, v in DEFAULT_SUBSCRIPTIONS.items()}
    path = Path(path or os.environ.get("AGENTS_YAML", Path(__file__).resolve().parents[1] / "agents.yaml"))
    if path.exists():
        import yaml

        for name, cfg in ((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("agents") or {}).items():
            if (cfg or {}).get("subscriptions"):
                subs[name] = list(cfg["subscriptions"])
    return subs


class SqliteBoard(InMemoryBoard):
    """InMemoryBoard's API on the messages, cursors and heartbeats tables of board.db.

    Every agent process opens the same file; read() keeps each agent's place in
    cursors, so a restarted agent carries on where it stopped.
    """

    def __init__(self, db_path, subscriptions: dict[str, list[str]] | None = None):
        self.db_path = str(db_path)
        self.subscriptions = subscriptions if subscriptions is not None else load_subscriptions()
        with self._connect() as con:
            con.executescript(BOARD_SCHEMA)

    def _connect(self):
        con = sqlite3.connect(self.db_path, timeout=10)
        con.execute("PRAGMA journal_mode=WAL")
        return _Closing(con)

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        msg = {k: row[k] for k in COLUMNS}
        for k in JSON_COLUMNS:
            msg[k] = json.loads(msg[k]) if msg[k] else ([] if k == "to" else {})
        return msg

    def insert(self, con: sqlite3.Connection, msg: dict) -> str:
        """Validate and write one message on an open connection (lets a caller post in its own transaction)."""
        msg = new_message(**msg)
        values = [json.dumps(msg[k]) if k in JSON_COLUMNS else msg[k] for k in COLUMNS]
        cols = ",".join(f'"{c}"' for c in COLUMNS)
        con.execute(f"INSERT INTO messages ({cols}) VALUES ({','.join('?' * len(COLUMNS))})", values)
        return msg["id"]

    def post(self, msg: dict) -> str:
        with self._connect() as con:
            return self.insert(con, msg)

    def _select(self, where: str = "", params: tuple = ()) -> list[sqlite3.Row]:
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            return con.execute(f"SELECT * FROM messages {where} ORDER BY seq", params).fetchall()

    @property
    def messages(self) -> list[dict]:
        return [self._row(r) for r in self._select()]

    def read(self, agent: str) -> list[dict]:
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            cur = con.execute("SELECT last_seq FROM cursors WHERE agent = ?", (agent,)).fetchone()
            rows = con.execute("SELECT * FROM messages WHERE seq > ? ORDER BY seq", (cur[0] if cur else 0,)).fetchall()
            if rows:
                con.execute("INSERT INTO cursors VALUES (?, ?) ON CONFLICT(agent) DO UPDATE SET last_seq = excluded.last_seq",
                            (agent, rows[-1]["seq"]))
        return [m for m in map(self._row, rows) if self._for(agent, m)]

    def answer(self, original: dict, **fields: Any) -> str:
        msg_id = super().answer(original, **fields)  # builds the answer and posts it through self.post
        with self._connect() as con:
            con.execute("UPDATE messages SET status = 'answered' WHERE id = ?", (original["id"],))
        return msg_id

    def ask(self, msg: dict, timeout_s: float = 60, poll_s: float = 0.5) -> dict | None:
        """Post a question or request and wait for its answer; None when nobody answers in time."""
        msg_id = self.post(msg)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rows = self._select("WHERE reply_to = ?", (msg_id,))
            if rows:
                return self._row(rows[0])
            time.sleep(poll_s)
        return None

    def thread(self, incident_id: str) -> list[dict]:
        return [self._row(r) for r in self._select("WHERE incident_id = ?", (incident_id,))]

    def heartbeat(self, agent: str) -> None:
        with self._connect() as con:
            con.execute("INSERT INTO heartbeats VALUES (?, ?) ON CONFLICT(agent) DO UPDATE SET time = excluded.time",
                        (agent, _now()))

    @property
    def heartbeats(self) -> dict[str, str]:
        with self._connect() as con:
            return dict(con.execute("SELECT agent, time FROM heartbeats").fetchall())


class _Closing:
    """`with` that commits (or rolls back) and then closes: on Windows an open board.db can't be deleted."""

    def __init__(self, con: sqlite3.Connection):
        self.con = con

    def __enter__(self) -> sqlite3.Connection:
        return self.con

    def __exit__(self, exc_type, *_):
        try:
            if exc_type is None:
                self.con.commit()
            else:
                self.con.rollback()
        finally:
            self.con.close()


def tail(db_path, follow: bool = True, poll_s: float = 1.0) -> None:
    """Print the board as it fills: python -m shared.standins tail board.db"""
    board, last = SqliteBoard(db_path, {}), 0
    while True:
        for row in board._select("WHERE seq > ?", (last,)):
            last = row["seq"]
            to = ",".join(json.loads(row["to"]) or ["*"])
            print(f"{row['time']}  {row['sender']:>8} -> {to:<14} {row['type']:8} {row['kind']:26} {row['summary']}", flush=True)
        if not follow:
            return
        time.sleep(poll_s)


class BaseAgent:
    """Message-handling part of the roadmap 5.8 loop. The full loop is P3's."""

    name = "base"
    menu: list = []

    def __init__(self, board: InMemoryBoard):
        self.board = board
        self.handled: set[str] = set()

    def handle(self, msg: dict) -> None:  # each agent overrides
        pass

    def safe_handle(self, msg: dict) -> None:
        if msg["id"] in self.handled:  # handlers must be idempotent
            return
        self.handled.add(msg["id"])
        try:
            self.handle(msg)
        except Exception as exc:  # never crash on a bad message
            print(f"[{self.name}] failed to handle {msg.get('kind')} {msg.get('id')}: {exc}")

    def step(self) -> None:
        for msg in self.board.read(self.name):
            self.safe_handle(msg)
        self.board.heartbeat(self.name)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Look at a board.db")
    ap.add_argument("command", choices=["tail"])
    ap.add_argument("db", nargs="?", default="board.db")
    ap.add_argument("--once", action="store_true", help="print what is there and stop")
    args = ap.parse_args()
    try:
        tail(args.db, follow=not args.once)
    except KeyboardInterrupt:
        pass
