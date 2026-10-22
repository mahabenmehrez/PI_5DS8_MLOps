from datetime import datetime, timedelta, timezone

import pytest

from agents.deployer import state
from agents.deployer.agent import DeployerAgent
from agents.deployer.config import load_deployer_config
from shared import memory
from shared.messages import Message, MessageType

T0 = datetime(2099, 1, 1, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    db = str(tmp_path / "board.db")
    monkeypatch.setenv("BOARD_DB", db)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("DRY_RUN", raising=False)
    monkeypatch.setattr(memory, "DB_PATH", db)
    clock = {"now": T0}
    monkeypatch.setattr(state, "_now", lambda: clock["now"].isoformat())
    return clock


def rows(variant, n, start_id=0, latency=20.0, minute=0, version="2", status=200):
    p = variant[:4]
    t = (T0 + timedelta(minutes=minute)).isoformat()
    return [{"request_id": f"{p}{minute}-{version}-{status}-{start_id + i}", "wine_id": start_id + i,
             "time": t, "model_version": version, "variant": variant, "features": "[]",
             "prediction": 5.0 + ((start_id + i) % 7) * 0.1,
             "latency_ms": latency, "status_code": status} for i in range(n)]


def add_labels(n):
    con = state.connect()
    con.executemany("INSERT OR REPLACE INTO labels VALUES (?,?,?)",
                    [(i, 5.0, "2000-01-01T00:00:00+00:00") for i in range(n)])
    con.commit()
    con.close()


def healthy_10():
    state.set_release_state(stage="10%", candidate_version="2", champion_version="1",
                            candidate_share=0.10)
    state.log_predictions(rows("candidate", 200) + rows("champion", 200))
    add_labels(200)


def agent_with(monkeypatch, **kw):
    a = DeployerAgent(**kw)
    monkeypatch.setattr(a, "_minutes_in_stage", lambda st: 60.0)
    return a


def human(decision):
    return Message(type=MessageType.ANSWER, kind="human.decision", sender="human",
                   to=["deployer"], topic="incident:x", level=3, summary=decision,
                   payload={"decision": decision})


# ---------------------------------------------------------------- config
def test_config_reads_thresholds(tmp_path):
    f = tmp_path / "agents.yaml"
    f.write_text("agents:\n  deployer:\n    thresholds: {p95_ms: 250, error_rate: 0.03, alpha: 0.1}\n")
    cfg = load_deployer_config(str(f))
    assert cfg == {"limit_p95_ms": 250, "limit_error_rate": 0.03, "alpha": 0.1}


def test_config_missing_file_gives_defaults(tmp_path):
    assert load_deployer_config(str(tmp_path / "none.yaml")) == {}


def test_config_unknown_key_is_an_error(tmp_path):
    f = tmp_path / "agents.yaml"
    f.write_text("agents:\n  deployer:\n    thresholds: {p95: 250}\n")
    with pytest.raises(ValueError):
        load_deployer_config(str(f))


def test_threshold_from_yaml_changes_behaviour(tmp_path, monkeypatch):
    f = tmp_path / "agents.yaml"
    f.write_text("agents:\n  deployer:\n    thresholds: {p95_ms: 100}\n")
    state.set_release_state(stage="10%", candidate_version="2", champion_version="1",
                            candidate_share=0.10)
    state.log_predictions(rows("candidate", 50, latency=150.0, minute=0)
                          + rows("candidate", 50, start_id=50, latency=150.0, minute=1)
                          + rows("champion", 100))
    agent = DeployerAgent.from_yaml(str(f))
    agent.step([])
    assert agent.outbox[-1].kind == "deployment.rolled_back"     # 150 ms > limite de 100 ms


# ---------------------------------------------------------------- traffic_replay
class Replay:
    def __init__(self, ok):
        self.ok, self.calls = ok, 0

    def __call__(self, st, analysis):
        self.calls += 1
        return {"ok": self.ok, "p95_ms": 31.0, "detail": "p95 31 ms" if self.ok else "p95 900 ms"}


def test_replay_ok_allows_50_and_is_in_evidence(monkeypatch):
    healthy_10()
    rp = Replay(True)
    agent = agent_with(monkeypatch, replay=rp)
    agent.step([])
    assert rp.calls == 1 and state.get_release_state()["stage"] == "50%"
    assert agent.outbox[-1].evidence["traffic_replay"]["ok"] is True


def test_replay_failure_blocks_50_and_is_not_repeated(monkeypatch, env):
    healthy_10()
    rp = Replay(False)
    agent = agent_with(monkeypatch, replay=rp)
    agent.step([])
    assert state.get_release_state()["stage"] == "10%"
    assert "traffic_replay" in agent.last_hold
    agent.step([])
    assert rp.calls == 1                                    # pas de relance immediate
    env["now"] = T0 + timedelta(minutes=11)
    agent.step([])
    assert rp.calls == 2                                    # relance apres 10 minutes


def test_replay_crash_is_a_failure_not_an_exception(monkeypatch):
    healthy_10()

    def boom(st, analysis):
        raise RuntimeError("api injoignable")
    agent = agent_with(monkeypatch, replay=boom)
    agent.step([])
    assert state.get_release_state()["stage"] == "10%"


def test_replay_not_required_for_shadow_to_10(monkeypatch):
    state.set_release_state(stage="shadow", candidate_version="2", champion_version="1",
                            shadow_on=1)
    state.log_predictions(rows("shadow", 320) + rows("champion", 320))
    add_labels(320)
    rp = Replay(False)
    agent = agent_with(monkeypatch, replay=rp)
    agent.step([])
    assert rp.calls == 0 and state.get_release_state()["stage"] == "10%"


# ---------------------------------------------------------------- recuperation apres rollback
def slow_breach():
    state.set_release_state(stage="10%", candidate_version="2", champion_version="1",
                            candidate_share=0.10)
    state.log_predictions(rows("candidate", 50, latency=420.0, minute=0)
                          + rows("candidate", 50, start_id=50, latency=410.0, minute=1)
                          + rows("champion", 100, minute=1))


def test_rollback_is_verified_two_minutes_later(monkeypatch, env):
    slow_breach()
    env["now"] = T0 + timedelta(minutes=5)
    agent = DeployerAgent()
    agent.step([])
    assert agent.outbox[-1].kind == "deployment.rolled_back" and agent.watching()
    env["now"] = T0 + timedelta(minutes=8)
    state.log_predictions(rows("champion", 100, start_id=500, minute=7))
    agent.step([])
    closed = agent.outbox[-1]
    assert closed.kind == "incident.closed" and closed.payload["outcome"] == "recovered"
    assert closed.payload["actions"] == ["rollback_emergency"]
    assert not agent.watching() and closed.incident_id.startswith("inc-")


def test_rollback_not_recovered_when_errors_persist(monkeypatch, env):
    slow_breach()
    env["now"] = T0 + timedelta(minutes=5)
    agent = DeployerAgent()
    agent.step([])
    env["now"] = T0 + timedelta(minutes=8)
    state.log_predictions(rows("champion", 100, start_id=600, minute=7, status=500))
    agent.step([])
    assert agent.outbox[-1].payload["outcome"] == "not_recovered"


def test_recovery_waits_without_traffic(monkeypatch, env):
    slow_breach()
    env["now"] = T0 + timedelta(minutes=5)
    agent = DeployerAgent()
    agent.step([])
    env["now"] = T0 + timedelta(minutes=9)
    n = len(agent.outbox)
    agent.step([])
    assert len(agent.outbox) == n and agent.watching()


# ---------------------------------------------------------------- apres le 100 %
def promote(monkeypatch, env):
    healthy_10()
    state.set_release_state(stage="50%", candidate_share=0.5)
    agent = agent_with(monkeypatch)
    agent.step([])
    assert agent.pending is not None
    env["now"] = T0 + timedelta(minutes=10)
    agent.step([human("approve")])
    assert state.get_release_state()["champion_version"] == "2"
    return agent


def test_degraded_new_champion_is_restored_automatically(monkeypatch, env):
    agent = promote(monkeypatch, env)
    state.log_predictions(rows("champion", 60, start_id=1000, latency=420.0, minute=11)
                          + rows("champion", 60, start_id=1100, latency=430.0, minute=12))
    env["now"] = T0 + timedelta(minutes=13)
    agent.step([])
    st = state.get_release_state()
    assert st["champion_version"] == "1" and st["stage"] == "none"
    rolled = [m for m in agent.outbox if m.kind == "deployment.rolled_back"][-1]
    assert rolled.payload["from_version"] == "2" and rolled.payload["to_version"] == "1"
    assert rolled.payload["automatic"] is True


def test_healthy_new_champion_stays(monkeypatch, env):
    agent = promote(monkeypatch, env)
    state.log_predictions(rows("champion", 60, start_id=1000, minute=11)
                          + rows("champion", 60, start_id=1100, minute=12))
    env["now"] = T0 + timedelta(minutes=13)
    agent.step([])
    assert state.get_release_state()["champion_version"] == "2"


def test_post_promotion_watch_expires(monkeypatch, env):
    agent = promote(monkeypatch, env)
    env["now"] = T0 + timedelta(minutes=200)
    agent.step([])
    assert agent.post_promotion is None
