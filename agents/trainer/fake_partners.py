"""Fake partners for the long-running Trainer (roadmap 5.4, solo mode on board.db).

Plays the four pieces the Trainer talks to, until the real ones are merged:
  replay + label feeder  write 40 batches of a scenario and their labels
  Data Checker           marks every batch passed
  Monitor                sends request.retrain and prints the answer
  Deployer               answers request.deploy_candidate with accepted

Start the agent in one terminal and this in another, on the same board.db:

    python -m agents.trainer --db runs/demo/board.db --mlflow http://localhost:5000
    python -m agents.trainer.fake_partners --db runs/demo/board.db --scenario D
"""

from __future__ import annotations

import argparse
import sqlite3
import tempfile
import time
from contextlib import closing
from pathlib import Path

from agents.trainer.solo import SCENARIOS, build_replay
from shared.standins import SqliteBoard

DONE = {"done", "refused", "failed"}


def load_scenario(db_path: str | Path, scenario: str, seed: int = 0) -> dict:
    """Build the scenario's batches next to db_path, then post them and request.retrain in one transaction.

    One transaction, so a Trainer that is already running sees the batches and the
    request together (not 40 batches first, which would start a scheduled retrain).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as con:
        has = con.execute("SELECT name FROM sqlite_master WHERE name = 'batches'").fetchone()
        if has and con.execute("SELECT COUNT(*) FROM batches").fetchone()[0]:
            raise SystemExit(f"{db_path} already has batches: use a new --db for each run")
    replay = build_replay(Path(tempfile.mkdtemp(dir=db_path.parent, prefix=f"replay_{scenario}_")), SCENARIOS[scenario],
                          seed=seed)
    board = SqliteBoard(db_path)
    request = {"type": "request", "kind": "request.retrain", "sender": "monitor", "to": ["trainer"],
               "topic": "model:wine-quality", "incident_id": "inc-" + scenario.lower().ljust(8, "0")[:8], "level": 1,
               "summary": f"Scenario {scenario}: live error (MAE) went up, please retrain",
               "payload": {"reason": "label_mae_rise", "window": {}, "affected_features": [], "expected_change_id": None}}
    with closing(sqlite3.connect(replay.db_path)) as src, closing(sqlite3.connect(db_path, timeout=10)) as con:
        from agents.trainer.data import SCHEMA

        con.executescript(SCHEMA)
        with con:
            con.executemany("INSERT INTO batches VALUES (?,?,?,?,?)", src.execute("SELECT * FROM batches").fetchall())
            con.executemany("INSERT INTO labels VALUES (?,?,?)", src.execute("SELECT * FROM labels").fetchall())
            request_id = board.insert(con, request)
    print(f"replay {scenario}: {replay.n_batches} batches + labels written, request.retrain {request_id} posted")
    return {"request_id": request_id, "replay": replay}


def play(db_path: str | Path, request_id: str, timeout_s: float = 600, poll_s: float = 1.0) -> dict | None:
    """Act as Monitor and Deployer until the retrain request is answered (or timeout). Returns that answer."""
    board = SqliteBoard(db_path)
    deadline, answer = time.monotonic() + timeout_s, None
    while time.monotonic() < deadline:
        for msg in board.read("deployer"):
            print(f"deployer <- {msg['kind']}: {msg['summary']}")
            if msg["kind"] == "request.deploy_candidate" and msg["type"] == "request":
                board.answer(msg, sender="deployer", level=1, summary="Candidate accepted for shadow testing (fake Deployer)",
                             payload={"decision": "accepted", "detail": "fake deployer: no traffic is moved"})
        for msg in board.read("monitor"):
            print(f"monitor  <- {msg['kind']}: {msg['summary']}")
            if msg["type"] == "answer" and msg["reply_to"] == request_id and msg["payload"].get("decision") in DONE:
                answer = msg
        if answer:
            return answer
        time.sleep(poll_s)
    print(f"no final answer from the trainer after {timeout_s:.0f} s: is `python -m agents.trainer` running on {db_path}?")
    return None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m agents.trainer.fake_partners", description=__doc__.splitlines()[0])
    ap.add_argument("--db", default="runs/demo/board.db", help="the board.db the Trainer runs on")
    ap.add_argument("--scenario", default="D", choices=sorted(SCENARIOS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=600, help="seconds to wait for the trainer")
    args = ap.parse_args(argv)
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    sent = load_scenario(args.db, args.scenario, args.seed)
    answer = play(args.db, sent["request_id"], timeout_s=args.timeout)
    if answer:
        print(f"retrain request {answer['payload']['decision']}: {answer['payload'].get('detail')}")


if __name__ == "__main__":
    main()
