"""Run the Data Checker as a long-running agent on the team's board.db.

    python -m agents.checker                          # board.db and LLM from agents.yaml / .env
    python -m agents.checker --db runs/demo/board.db
    python -m agents.checker --llm off                # rules only (the traditional system)
    python -m agents.checker --once                   # one round, then exit

Same wiring as the Trainer (agents/trainer/__main__.py): shared.board.Board (P6) when it
exists, otherwise the SqliteBoard stand-in with the same API and tables, shared between
processes. The LLM is shared/llm.py (provider, model and key from agents.yaml and .env);
its calls are logged in the llm_calls table of the same board.db. Repairs are tried with
shared/dry_run.py in a subprocess.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from shared import clock
from shared.dry_run import dry_run
from shared.tables import Tables

from agents.checker.agent import CheckerAgent
from agents.checker.settings import ROOT

log = logging.getLogger("agents.checker")


def make_board(db_path: str):
    """The team's board (P6) when merged, the stand-in until then. Returns (board, label)."""
    try:
        from shared.board import Board
    except ImportError:
        from shared.standins import SqliteBoard, load_subscriptions
        subs = load_subscriptions()
        subs.setdefault("checker", CheckerAgent.subscriptions)
        return SqliteBoard(db_path, subs), "stand-in board (shared/standins.py)"
    try:
        return Board(db_url=f"sqlite:///{db_path}"), "shared.board"
    except TypeError:
        return Board(), "shared.board"


class ConfiguredLLM:
    """shared/llm.py with one config, so its llm_calls land in this agent's board.db."""

    def __init__(self, config: dict):
        self.config = config

    def decide(self, agent, context, menu, fallback, *, incident_id=None):
        from shared import llm
        return llm.decide(agent, context, menu, fallback, incident_id=incident_id, config=self.config)


def llm_config(choice: str, db_path: str) -> dict:
    """config = agents.yaml / .env; or a provider name (groq, nvidia_nim, ollama, fake)."""
    from shared.llm import config_for, load_config
    cfg = load_config() if choice == "config" else config_for(choice)
    cfg["db_url"] = f"sqlite:///{db_path}"
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", default=os.environ.get("BOARD_DB", str(ROOT / "board.db")))
    p.add_argument("--llm", default="config", help="config | off | groq | nvidia_nim | ollama | fake")
    p.add_argument("--poll", type=float, default=float(os.environ.get("POLL_S", 2)))
    p.add_argument("--once", action="store_true", help="one round, then exit")
    args = p.parse_args()

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    clock.configure()                                  # CLOCK_MODE / CLOCK_* set by the harness
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    board, label = make_board(args.db)
    rules_only = args.llm == "off" or os.environ.get("REASON") == "rules"
    agent = CheckerAgent(
        board=board,
        tables=Tables(args.db),
        llm=None if rules_only else ConfiguredLLM(llm_config(args.llm, args.db)),
        dry_run=dry_run,                               # subprocess, DRY_RUN=1
        reason="rules" if rules_only else "llm",
    )
    log.info("checker on %s (%s), decisions by %s", args.db, label,
             "the rules" if rules_only else f"the LLM ({args.llm})")
    if args.once:
        agent.step()
        return
    agent.run(poll_s=args.poll)


if __name__ == "__main__":
    main()
