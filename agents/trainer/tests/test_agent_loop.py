"""The Trainer as a long-running agent on board.db: the SqliteBoard stand-in, the runner, and one full round trip."""

from agents.trainer import fake_partners
from agents.trainer.__main__ import main as run_trainer
from agents.trainer.solo import setup_champion
from shared.standins import DEFAULT_SUBSCRIPTIONS, SqliteBoard


def ping(board, to="trainer"):
    return board.post({"type": "question", "kind": "test.ping", "sender": "harness", "to": [to],
                       "topic": "pipeline:setup", "summary": "hello board"})


def test_sqlite_board_api(tmp_path):
    db = tmp_path / "board.db"
    board = SqliteBoard(db, DEFAULT_SUBSCRIPTIONS)
    named = ping(board)
    board.post({"type": "finding", "kind": "data.batch_checked", "sender": "checker", "topic": "dataset:batch-0001",
                "incident_id": "inc-0000aaaa", "summary": "batch passed"})
    board.post({"type": "finding", "kind": "code.reviewed", "sender": "reviewer", "topic": "code:baseline/train.py",
                "summary": "not for the trainer"})
    got = board.read("trainer")
    assert [m["kind"] for m in got] == ["test.ping", "data.batch_checked"]  # named in `to`, or a subscribed topic
    assert board.read("trainer") == []
    # another process (a new board object on the same file) keeps the cursor
    again = SqliteBoard(db, DEFAULT_SUBSCRIPTIONS)
    assert again.read("trainer") == []
    again.answer(got[0], sender="trainer", summary="pong", payload={"ok": True})
    msgs = again.messages
    assert next(m for m in msgs if m["id"] == named)["status"] == "answered"
    reply = next(m for m in msgs if m["type"] == "answer")
    assert reply["reply_to"] == named and reply["to"] == ["harness"] and reply["kind"] == "test.ping"
    assert [m["kind"] for m in again.thread("inc-0000aaaa")] == ["data.batch_checked"]
    assert again.ask({"type": "question", "kind": "test.ping", "sender": "harness", "to": ["nobody"],
                      "topic": "pipeline:setup", "summary": "anyone?"}, timeout_s=0.2, poll_s=0.05) is None


def test_runner_answers_on_the_board(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTS_YAML", str(tmp_path / "none.yaml"))
    db = tmp_path / "board.db"
    board = SqliteBoard(db)
    asked = ping(board)
    run_trainer(["--db", str(db), "--mlflow", f"file:{tmp_path / 'mlruns'}", "--llm", "off", "--once"])
    answer = next(m for m in board.messages if m["type"] == "answer")
    assert answer["reply_to"] == asked and answer["sender"] == "trainer"
    assert "trainer" in board.heartbeats
    # restarting the agent does not answer the same question twice
    run_trainer(["--db", str(db), "--mlflow", f"file:{tmp_path / 'mlruns'}", "--llm", "off", "--rounds", "2",
                 "--interval", "0"])
    assert len([m for m in board.messages if m["type"] == "answer"]) == 1


def test_round_trip_with_fake_partners(tmp_path, monkeypatch):
    """Scenario D: the fake Monitor asks, the running Trainer retrains, the fake Deployer accepts."""
    monkeypatch.setenv("AGENTS_YAML", str(tmp_path / "none.yaml"))
    tracking = f"file:{tmp_path / 'mlruns'}"
    setup_champion(tracking)
    db = tmp_path / "board.db"
    sent = fake_partners.load_scenario(db, "D")
    run_trainer(["--db", str(db), "--mlflow", tracking, "--llm", "off", "--once"])
    answer = fake_partners.play(db, sent["request_id"], timeout_s=5, poll_s=0.1)
    assert answer["payload"]["decision"] == "done"
    kinds = [(m["sender"], m["kind"], m["type"]) for m in SqliteBoard(db).messages]
    assert ("trainer", "request.deploy_candidate", "request") in kinds
    assert ("deployer", "request.deploy_candidate", "answer") in kinds


def test_second_request_on_the_same_batches_is_not_retrained(tmp_path, monkeypatch):
    """A scheduled retrain can land just before the Monitor's request: the request gets that result, no second model."""
    monkeypatch.setenv("AGENTS_YAML", str(tmp_path / "none.yaml"))
    tracking = f"file:{tmp_path / 'mlruns'}"
    setup_champion(tracking)
    db = tmp_path / "board.db"
    first = fake_partners.load_scenario(db, "D")
    args = ["--db", str(db), "--mlflow", tracking, "--llm", "off", "--once"]
    run_trainer(args)
    again = SqliteBoard(db).post({"type": "request", "kind": "request.retrain", "sender": "monitor", "to": ["trainer"],
                                  "topic": "model:wine-quality", "level": 1, "summary": "still worse, retrain",
                                  "payload": {"reason": "label_mae_rise"}})
    run_trainer(args)
    answers = {m["reply_to"]: m["payload"] for m in SqliteBoard(db).messages if m["type"] == "answer" and m["sender"] == "trainer"}
    assert answers[first["request_id"]]["decision"] == "done"
    assert answers[again] == {"decision": "done", "detail": "candidate v2 was already sent to the Deployer", "version": "2"}
    assert len([m for m in SqliteBoard(db).messages if m["kind"] == "request.deploy_candidate"]) == 1


import pytest


@pytest.mark.parametrize("scenario, decision", [("D", "done"), ("E", "refused")])
def test_relative_db_and_one_decision_per_request(tmp_path, monkeypatch, scenario, decision):
    """As typed on Windows: a relative --db, and the agent left running for several rounds after it decided."""
    monkeypatch.setenv("AGENTS_YAML", str(tmp_path / "none.yaml"))
    monkeypatch.chdir(tmp_path)
    tracking = f"file:{tmp_path / 'mlruns'}"
    setup_champion(tracking)
    db = "runs/d1/board.db"
    sent = fake_partners.load_scenario(db, scenario)
    run_trainer(["--db", db, "--mlflow", tracking, "--llm", "off", "--rounds", "3", "--interval", "0"])
    msgs = SqliteBoard(db).messages
    answer = next(m for m in msgs if m["reply_to"] == sent["request_id"])
    assert answer["payload"]["decision"] == decision, answer["summary"]
    assert len([m for m in msgs if m["kind"] == "training.decision"]) == 1
