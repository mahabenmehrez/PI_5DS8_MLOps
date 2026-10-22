import json

import pytest

from agents.deployer import state
from agents.deployer.agent import DeployerAgent
from agents.deployer.llm_decision import DeployerLLM, LLMInvalid
from shared import memory

MENU = DeployerAgent.menu


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    db = str(tmp_path / "board.db")
    monkeypatch.setenv("BOARD_DB", db)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("DRY_RUN", raising=False)
    monkeypatch.setattr(memory, "DB_PATH", db)


def reply(action="hold", level=1, **over):
    body = {"action": action, "level": level, "params": {"reason": "test"},
            "reason": "chiffres cites", "confidence": 0.8, "success_check": "verifier"}
    body.update(over)
    return json.dumps(body)


def make(responses):
    it = iter(responses)

    def transport(system, user):
        return next(it), {"tokens": 10}
    return DeployerLLM(transport, "fake", "fake")


# ---------------------------------------------------------------- le wrapper LLM
def test_valid_answer_is_accepted():
    llm = make([reply()])
    assert llm({}, MENU)["action"] == "hold"
    assert llm.calls[0]["valid"] is True


def test_json_inside_code_fence_is_accepted():
    llm = make(["```json\n" + reply("rollback", 2) + "\n```"])
    assert llm({}, MENU)["action"] == "rollback"


def test_one_retry_then_success():
    llm = make(["pas du json", reply()])
    assert llm({}, MENU)["action"] == "hold"
    assert len(llm.calls) == 2 and llm.calls[0]["valid"] is False


def test_two_invalid_answers_raise():
    llm = make(["pas du json", "toujours pas"])
    with pytest.raises(LLMInvalid):
        llm({}, MENU)


@pytest.mark.parametrize("forbidden", ["advance_stage", "promote_100", "rollback_emergency", "delete_all"])
def test_forbidden_actions_are_rejected(forbidden):
    level = {"advance_stage": 2, "promote_100": 3, "rollback_emergency": 1}.get(forbidden, 1)
    llm = make([reply(forbidden, level), reply(forbidden, level)])
    with pytest.raises(LLMInvalid):
        llm({}, MENU)


def test_wrong_level_is_rejected():
    llm = make([reply("rollback", 1), reply("rollback", 1)])
    with pytest.raises(LLMInvalid):
        llm({}, MENU)


# ---------------------------------------------------------------- dans l'agent
def rows(variant, n, start_id=0, latency=20.0, minute="00"):
    p = variant[:4]
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


def grey_zone(cand_latency=60.0):
    """Candidat lent (60 ms contre 20 ms) mais sous la limite dure de 300 ms."""
    state.set_release_state(stage="10%", candidate_version="2",
                            champion_version="1", candidate_share=0.10)
    state.log_predictions(rows("candidate", 200, latency=cand_latency)
                          + rows("champion", 200, latency=20.0))
    add_labels(200)


class Spy:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def __call__(self, context, menu):
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def agent_with(llm, monkeypatch):
    agent = DeployerAgent(llm=llm)
    monkeypatch.setattr(agent, "_minutes_in_stage", lambda st: 60.0)
    return agent


def test_llm_can_choose_rollback_in_grey_zone(monkeypatch):
    grey_zone()
    spy = Spy({"action": "rollback", "level": 2, "params": {"reason": "lent"}})
    agent = agent_with(spy, monkeypatch)
    agent.step([])
    assert spy.calls == 1
    assert state.get_release_state()["stage"] == "none"
    assert agent.outbox[-1].kind == "deployment.rolled_back"
    assert agent.outbox[-1].payload["automatic"] is False


def test_llm_can_choose_hold_in_grey_zone(monkeypatch):
    grey_zone()
    spy = Spy({"action": "hold", "level": 1, "params": {"reason": "bruit possible"}})
    agent = agent_with(spy, monkeypatch)
    agent.step([])
    assert state.get_release_state()["stage"] == "10%"
    assert agent.last_hold == "bruit possible"


def test_guardrail_blocks_advance_from_llm(monkeypatch):
    grey_zone()
    spy = Spy({"action": "advance_stage", "level": 2, "params": {"to_stage": "50%"}})
    agent = agent_with(spy, monkeypatch)
    agent.step([])
    assert state.get_release_state()["stage"] == "10%"      # le candidat ne monte pas


def test_llm_failure_falls_back_to_rules(monkeypatch):
    grey_zone()
    agent = agent_with(Spy(RuntimeError("modele indisponible")), monkeypatch)
    agent.step([])
    assert state.get_release_state()["stage"] == "10%"      # repli : hold
    assert agent.last_hold and "p95" in agent.last_hold


def test_llm_not_called_when_gates_pass(monkeypatch):
    grey_zone(cand_latency=20.0)
    spy = Spy({"action": "rollback", "level": 2, "params": {"reason": "x"}})
    agent = agent_with(spy, monkeypatch)
    agent.step([])
    assert spy.calls == 0
    assert state.get_release_state()["stage"] == "50%"      # les regles avancent seules


def test_llm_never_called_on_hard_limit(monkeypatch):
    state.set_release_state(stage="10%", candidate_version="2",
                            champion_version="1", candidate_share=0.10)
    state.log_predictions(rows("candidate", 50, latency=420.0, minute="00")
                          + rows("candidate", 50, start_id=50, latency=410.0, minute="01")
                          + rows("champion", 100))
    spy = Spy({"action": "hold", "level": 1, "params": {"reason": "x"}})
    agent = agent_with(spy, monkeypatch)
    agent.step([])
    assert spy.calls == 0
    assert agent.outbox[-1].kind == "deployment.rolled_back"
    assert agent.outbox[-1].payload["automatic"] is True


def test_same_situation_reuses_decision(monkeypatch):
    grey_zone()
    spy = Spy({"action": "hold", "level": 1, "params": {"reason": "attente"}})
    agent = agent_with(spy, monkeypatch)
    agent.step([])
    agent.step([])
    assert spy.calls == 1                                    # pas de nouvel appel inutile
