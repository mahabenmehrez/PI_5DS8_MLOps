import pytest

from agents.deployer import state
from agents.deployer.agent import DeployerAgent
from shared import memory
from shared.messages import Message, MessageType


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    db = str(tmp_path / "board.db")
    monkeypatch.setenv("BOARD_DB", db)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("DRY_RUN", raising=False)
    monkeypatch.setattr(memory, "DB_PATH", db)


def make(kind, mtype, payload=None, sender="trainer"):
    return Message(type=mtype, kind=kind, sender=sender, to=["deployer"],
                   topic="deployment:prod", level=1, summary="test",
                   payload=payload or {})


def rows(variant, n, start_id=0, latency=20.0, minute="00", prefix=None):
    p = prefix or variant[:4]
    return [{"request_id": f"{p}{minute}-{start_id + i}", "wine_id": start_id + i,
             "time": f"2099-01-01T00:{minute}:00+00:00", "model_version": "2",
             "variant": variant, "features": "[]",
             "prediction": 5.0 + ((start_id + i) % 7) * 0.1,
             "latency_ms": latency, "status_code": 200} for i in range(n)]


def add_labels(n):
    con = state.connect()
    con.executemany("INSERT OR REPLACE INTO labels VALUES (?,?,?)",
                    [(i, 5.0, "2000-01-01T00:00:00+00:00") for i in range(n)])
    con.commit()
    con.close()


def test_accepts_candidate_and_goes_shadow():
    agent = DeployerAgent()
    req = make("request.deploy_candidate", MessageType.REQUEST, {"version": "2"})
    out = agent.step([req])
    answers = [m for m in out if m.type == MessageType.ANSWER]
    assert answers[0].payload["decision"] == "accepted"
    assert answers[0].reply_to == req.id
    st = state.get_release_state()
    assert st["stage"] == "shadow" and st["candidate_version"] == "2"


def test_handler_is_idempotent():
    agent = DeployerAgent()
    req = make("request.deploy_candidate", MessageType.REQUEST, {"version": "2"})
    agent.step([req, req])
    assert len([m for m in agent.outbox if m.type == MessageType.ANSWER]) == 1


def test_hold_and_unknown_kind():
    agent = DeployerAgent()
    hold = make("request.hold", MessageType.REQUEST, {"reason": "unit change"}, "checker")
    agent.step([hold, make("data.batch_checked", MessageType.FINDING)])
    assert len(agent.holds) == 1


def test_menu_levels():
    levels = {a.name: a.level for a in DeployerAgent.menu}
    assert levels["promote_100"] == 3
    assert levels["rollback_emergency"] == 1


def test_shadow_gate_passes_then_advances_to_10():
    state.set_release_state(stage="shadow", candidate_version="2",
                            champion_version="1", shadow_on=1)
    state.log_predictions(rows("shadow", 320) + rows("champion", 320))
    add_labels(320)
    agent = DeployerAgent()
    agent.step([])
    assert state.get_release_state()["stage"] == "10%"
    assert agent.outbox[-1].kind == "deployment.stage_changed"


def test_shadow_waits_without_enough_data():
    state.set_release_state(stage="shadow", candidate_version="2", champion_version="1")
    state.log_predictions(rows("shadow", 20) + rows("champion", 20))
    DeployerAgent().step([])
    assert state.get_release_state()["stage"] == "shadow"


def test_emergency_rollback_after_two_slow_minutes():
    state.set_release_state(stage="10%", candidate_version="2",
                            champion_version="1", candidate_share=0.10)
    state.log_predictions(rows("candidate", 50, latency=420.0, minute="00")
                          + rows("candidate", 50, start_id=50, latency=410.0, minute="01")
                          + rows("champion", 100))
    agent = DeployerAgent()
    agent.step([])
    st = state.get_release_state()
    assert st["candidate_share"] == 0.0 and st["stage"] == "none"
    last = agent.outbox[-1]
    assert last.kind == "deployment.rolled_back" and last.payload["automatic"] is True


def test_promote_100_needs_human_approval(monkeypatch):
    state.set_release_state(stage="50%", candidate_version="2",
                            champion_version="1", candidate_share=0.5)
    state.log_predictions(rows("candidate", 200) + rows("champion", 200))
    add_labels(200)
    agent = DeployerAgent()
    monkeypatch.setattr(agent, "_minutes_in_stage", lambda st: 60.0)
    agent.step([])
    assert agent.outbox[-1].kind == "request.approval"
    assert state.get_release_state()["stage"] == "50%"      # rien ne change sans humain
    decision = Message(type=MessageType.ANSWER, kind="human.decision", sender="human",
                       to=["deployer"], topic="incident:x", level=3,
                       summary="ok", payload={"decision": "approve"})
    agent.step([decision])
    st = state.get_release_state()
    assert st["champion_version"] == "2" and st["stage"] == "none"