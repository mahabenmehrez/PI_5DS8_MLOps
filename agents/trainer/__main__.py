"""Run the Trainer as a long-running agent on the team's board.db.

    python -m agents.trainer                          # board.db and MLflow from agents.yaml
    python -m agents.trainer --db runs/demo/board.db --mlflow http://localhost:5000
    python -m agents.trainer --once                   # one round, then exit (CI, debugging)

Every round (default every 2 s, roadmap 3.4): read new messages, heartbeat,
check whether a retrain is requested or due, decide, act. Ctrl+C stops it
cleanly. Uses shared.board.Board (owner P6) when it exists, otherwise the
SqliteBoard stand-in, which keeps the same API and tables.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from agents.trainer.agent import SUBSCRIPTIONS, TrainerAgent, load_settings
from agents.trainer.data import TrainerStore

log = logging.getLogger("trainer")


def make_board(db_path: str):
    """The team's board (P6) when merged, the stand-in until then. Returns (board, label)."""
    try:
        from shared.board import Board
    except ImportError:
        from shared.standins import SqliteBoard, load_subscriptions

        subs = load_subscriptions()
        subs.setdefault("trainer", SUBSCRIPTIONS)
        return SqliteBoard(db_path, subs), "stand-in board (shared/standins.py)"
    try:
        return Board(db_url=f"sqlite:///{db_path}"), "shared.board"
    except TypeError:
        return Board(), "shared.board"


def llm_config_for(choice: str, db_path: str) -> dict:
    """config = agents.yaml / .env (default); off = rules only; or a provider name."""
    from shared.llm import config_for, load_config

    cfg = load_config() if choice == "config" else config_for(choice)
    cfg["db_url"] = cfg.get("db_url") or f"sqlite:///{db_path}"  # llm_calls table next to the messages
    return cfg


def _watch_posts(board) -> None:
    """Log every message the Trainer posts (kind + summary), whatever board class it is."""
    post = board.post

    def logged(msg):
        msg_id = post(msg)
        log.info("posted  %-9s %-26s %s", msg.get("type"), msg.get("kind"), msg.get("summary"))
        return msg_id

    board.post = logged


class Runner:
    def __init__(self, agent: TrainerAgent, interval_s: float = 2.0):
        self.agent, self.interval_s = agent, interval_s
        self._last_error = None

    def round(self) -> None:
        try:
            self.agent.step()
        except Exception as exc:  # never die on one bad round: log it, try again next round
            self._error(f"step failed: {type(exc).__name__}: {exc}")
        for entry in self.agent.log:
            if "error" in entry:
                self._error(f"{entry.get('phase')}: {entry['error']}")
            else:
                log.info("decided %-18s by %-10s %s", entry.get("action"), entry.get("source"), entry.get("reason"))
        self.agent.log.clear()  # printed; trainer_runs keeps the record

    def _error(self, text: str) -> None:
        if text != self._last_error:  # same failure every 2 s: say it once
            log.warning(text)
        self._last_error = text

    def run(self, max_rounds: int | None = None) -> None:
        n = 0
        while max_rounds is None or n < max_rounds:
            started = time.monotonic()
            self.round()
            n += 1
            if max_rounds is not None and n >= max_rounds:
                break
            time.sleep(max(0.0, self.interval_s - (time.monotonic() - started)))


def check_mlflow(tracking_uri: str) -> None:
    from agents.trainer.solo import has_champion

    if has_champion(tracking_uri):
        log.info("MLflow %s: wine-quality@champion found", tracking_uri)
    else:
        log.warning("MLflow %s: no wine-quality@champion yet. Retrain requests will be answered 'failed' until "
                    "one exists (run python baseline/train.py)", tracking_uri)


def build(args) -> Runner:
    overrides = {}
    if args.db:
        overrides["db_path"] = args.db
    if args.mlflow:
        overrides["tracking_uri"] = args.mlflow
    settings = load_settings(overrides)
    db_path = str(Path(settings["db_path"]))
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    board, label = make_board(db_path)
    _watch_posts(board)
    store = TrainerStore(db_path)
    store.create_tables()  # IF NOT EXISTS: never touches tables other agents already made
    llm_cfg = llm_config_for(args.llm, db_path)
    agent = TrainerAgent(board, store=store, settings=overrides, llm_config=llm_cfg)
    log.info("trainer up: board %s (%s), every %.0f s", db_path, label, args.interval)
    if llm_cfg.get("provider") in ("off", None):
        log.info("LLM: off, rule decisions only")
    else:
        log.info("LLM: %s %s (rules if it fails)", llm_cfg["provider"], llm_cfg.get("model") or "")
    check_mlflow(agent.cfg["tracking_uri"])
    return Runner(agent, interval_s=args.interval)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m agents.trainer", description=__doc__.splitlines()[0])
    ap.add_argument("--db", help="board.db path (default: db_url in agents.yaml, else ./board.db)")
    ap.add_argument("--mlflow", metavar="URL", help="MLflow tracking URI (default: agents.yaml, MLFLOW_TRACKING_URI, "
                                                     "http://localhost:5000)")
    ap.add_argument("--llm", default="config",
                    help="config (agents.yaml / .env, default) | off (rules only) | ollama | groq | nvidia_nim | fake")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between rounds (default 2)")
    ap.add_argument("--once", action="store_true", help="run one round and exit")
    ap.add_argument("--rounds", type=int, help="run this many rounds and exit")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    runner = build(args)
    try:
        runner.run(1 if args.once else args.rounds)
    except KeyboardInterrupt:
        log.info("stopped")


if __name__ == "__main__":
    main()
