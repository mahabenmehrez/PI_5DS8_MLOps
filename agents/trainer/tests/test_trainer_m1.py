import pytest

from agents.trainer.agent import SUBSCRIPTIONS, TrainerAgent
from agents.trainer.data import TrainerStore
from agents.trainer.stub import TrainerStub
from shared.llm import MenuItem
from shared.standins import InMemoryBoard


def board():
    return InMemoryBoard({"trainer": SUBSCRIPTIONS})


def retrain_request(b):
    return b.post({"type": "request", "kind": "request.retrain", "sender": "monitor", "to": ["trainer"],
                   "topic": "model:wine-quality", "incident_id": "inc-0000beef", "level": 1,
                   "summary": "Live MAE up 18%", "expires_at": "2026-03-15T10:00:00Z",
                   "payload": {"reason": "label_mae_rise", "window": {}, "affected_features": [], "expected_change_id": None}})


def test_answers_ping_and_refuses_without_data(tmp_path):
    b = board()
    store = TrainerStore(tmp_path / "board.db")
    store.create_tables()
    agent = TrainerAgent(b, store=store, settings={"tracking_uri": f"file:{tmp_path / 'mlruns'}"},
                         llm_config={"provider": "off", "db_url": None}, now=lambda: "2026-03-15T10:00:00Z")
    ping = b.post({"type": "question", "kind": "test.ping", "sender": "harness", "to": ["trainer"],
                   "topic": "pipeline:setup", "summary": "hello board"})
    req = retrain_request(b)
    agent.step()
    agent.step()  # nothing new, nothing repeated
    answers = [m for m in b.messages if m["type"] == "answer"]
    assert {a["reply_to"] for a in answers} == {ping, req} and len(answers) == 2
    assert all(a["sender"] == "trainer" for a in answers)
    refused = next(a for a in answers if a["reply_to"] == req)
    assert refused["payload"]["decision"] == "refused"
    finding = next(m for m in b.messages if m["kind"] == "training.decision")
    assert finding["payload"]["reason_code"] == "not_enough_data"
    assert "trainer" in b.heartbeats


def test_analysis_error_answers_failed(tmp_path):
    b = board()
    agent = TrainerAgent(b, store=TrainerStore(tmp_path / "missing" / "board.db"),
                         llm_config={"provider": "off", "db_url": None})
    req = retrain_request(b)
    agent.step()
    answer = next(m for m in b.messages if m["type"] == "answer")
    assert answer["reply_to"] == req and answer["payload"]["decision"] == "failed"


def test_menu_is_valid_for_llm_helper():
    items = [MenuItem.coerce(m) for m in TrainerAgent.menu]
    assert {i.name for i in items} >= {"refuse", "retrain_in_dry_run", "reject_candidate"}
    assert next(i for i in items if i.name == "retrain_in_dry_run").level == 2


@pytest.mark.parametrize("fixture_set, kinds", [
    ("retrain", ["training.decision", "request.deploy_candidate"]),
    ("refuse", ["training.decision"]),
    ("reject", ["training.decision", "model.candidate_rejected"]),
])
def test_stub(fixture_set, kinds):
    b = board()
    stub = TrainerStub(b, fixture_set)
    retrain_request(b)
    stub.step()
    out = [m for m in b.messages if m["sender"] == "trainer"]
    assert out[0]["type"] == "answer" and out[0]["payload"]["decision"] == "accepted"
    assert [m["kind"] for m in out[1:]] == kinds
    assert all(m["incident_id"] == "inc-0000beef" for m in out)
