"""Data Checker (P3) + Trainer (P4), both real, on one board.db (the SqliteBoard stand-in).

Replay: 10 clean batches, batch 11 with duplicated rows (the Checker repairs it), batches 12-13
with sulfur dioxide in g/L (the Checker quarantines them). Then the Trainer reads the board and
the shared batches table, as it does every round.
"""
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from shared import clock
from shared.base_agent import BaseAgent
from shared.standins import DEFAULT_SUBSCRIPTIONS, SqliteBoard, new_message
from shared.tables import Tables
from shared.testing import inprocess_dry_run
from agents.checker.agent import CheckerAgent
from agents.checker.settings import CheckerSettings
from agents.trainer.agent import TrainerAgent
from agents.trainer.analysis import choose_window
from agents.trainer.data import TrainerStore
from harness.fault_runner import FaultRunner, all_faults
from harness.replay import Replay

LATER = "2100-01-01T00:00:00Z"          # every label is available by then


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTS_YAML", str(tmp_path / "none.yaml"))   # the Trainer's defaults
    clock.configure(mode="frozen", start=datetime(2026, 3, 14, tzinfo=timezone.utc))
    db = str(tmp_path / "board.db")
    store = TrainerStore(db)
    store.create_tables()                     # the Trainer starts first: its smaller batches table
    tables = Tables(db)                       # ... gets the Checker's columns added
    board = SqliteBoard(db, {**DEFAULT_SUBSCRIPTIONS, "checker": CheckerAgent.subscriptions})
    faults = all_faults()
    replay = Replay(tables, tmp_path / "batches", fault_runner=FaultRunner([
        replace(faults["data.duplicated_rows"], after_batches=10, duration=1),
        replace(faults["data.unit_change.total_sulfur_dioxide"], after_batches=11, duration=2)]))
    checker = CheckerAgent(board=board, tables=tables, reason="rules", dry_run=inprocess_dry_run,
                           settings=CheckerSettings(data_dir=tmp_path / "checker"))
    for _ in range(13):
        replay.next_batch()
        checker.step()
    trainer = TrainerAgent(board, store=store, settings={"db_path": db},
                           llm_config={"provider": "off", "db_url": None})
    for msg in board.read("trainer"):         # what the Trainer's step() does with new messages
        trainer.safe_handle(msg)
    return {"db": db, "board": board, "tables": tables, "store": store, "trainer": trainer}


def test_the_trainer_runs_on_the_checkers_base_loop(run):
    assert isinstance(run["trainer"], BaseAgent)


def test_checker_decisions_are_in_the_shared_table(run):
    status = {b["batch_id"]: b["status"] for b in run["tables"].batches()}
    assert status["batch-0011"] == "repaired"
    assert status["batch-0012"] == status["batch-0013"] == "quarantined"
    assert all(status[f"batch-{i:04d}"] == "passed" for i in range(1, 11))


def test_the_trainer_excludes_quarantined_batches(run):
    trainer, board = run["trainer"], run["board"]
    assert {"batch-0012", "batch-0013"} <= trainer.do_not_use
    answers = [m for m in board.messages if m["sender"] == "trainer" and m["kind"] == "request.do_not_use"]
    assert answers and all(m["type"] == "answer" and m["payload"]["decision"] == "done" for m in answers)
    window = choose_window(run["store"].batches(LATER), do_not_use=trainer.do_not_use)
    used = set(window.train_ids + window.eval_ids)
    assert "batch-0011" in used and not used & {"batch-0012", "batch-0013"}


def test_the_trainer_trains_on_the_repaired_file(run):
    row = run["tables"].batch("batch-0011")
    assert row["path"] == row["clean_path"] and row["raw_path"] != row["path"]
    rows = run["store"].rows(["batch-0011"], LATER)
    assert len(rows) == rows["wine_id"].nunique() == 100     # the 20 duplicated rows are gone


def test_one_claim_per_incident_even_across_restarts(tmp_path):
    """A Checker restarted in the middle of the incident (new process: fresh memory) must not
    claim it again nor ask for a second hold; it keeps the same incident id."""
    clock.configure(mode="frozen", start=datetime(2026, 3, 14, tzinfo=timezone.utc))
    db = str(tmp_path / "board.db")
    tables, board = Tables(db), SqliteBoard(db, {"checker": CheckerAgent.subscriptions})
    replay = Replay(tables, tmp_path / "batches", fault_runner=FaultRunner(
        [all_faults()["data.unit_change.total_sulfur_dioxide"]]))
    for _ in range(13):
        replay.next_batch()
        CheckerAgent(board=board, tables=Tables(db), reason="rules", dry_run=inprocess_dry_run,
                     settings=CheckerSettings(data_dir=tmp_path / "checker")).step()   # a new agent each round
    mine = [m for m in board.messages if m["sender"] == "checker"]
    assert [m["kind"] for m in mine].count("incident.claim") == 1
    assert [m["kind"] for m in mine].count("request.hold") == 1
    assert len({m["incident_id"] for m in mine if m["kind"] == "request.do_not_use"}) == 1


def test_every_checker_message_is_valid_for_the_board(run):
    mine = [m for m in run["board"].messages if m["sender"] == "checker"]
    kinds = {m["kind"] for m in mine}
    assert {"data.batch_checked", "request.do_not_use", "incident.claim", "request.hold"} <= kinds
    for m in mine:
        new_message(**m)                      # raises on an invalid envelope
