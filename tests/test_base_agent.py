"""Unit tests for the 7-step loop in shared/base_agent.py, with a toy agent."""
from datetime import datetime, timezone

import pytest

from shared import clock
from shared.base_agent import Action, Analysis, BaseAgent, Decision, Result
from shared.messages import Message, MessageType
from shared.testing import DryRunResult, FakeLLM, InMemoryBoard


class ToyAgent(BaseAgent):
    name = "checker"
    subscriptions = ["dataset:*"]
    menu = [
        Action("pass", 1, success_check="row passed"),
        Action("quarantine", 1, success_check="batch absent", safest=True),
        Action("repair", 2, dry_run_kind="repair_check"),
        Action("promote", 3),
    ]

    def __init__(self, flag=True, rule="quarantine", **kw):
        super().__init__(**kw)
        self.flag, self.rule = flag, rule
        self.acted, self.handled = [], []

    def observe(self):
        return {"batch_id": "batch-0001"}

    def analyze(self, obs):
        return Analysis(flag=self.flag, nodes=["dataset:batch-0001"],
                        numbers={"psi": 0.9}, problem="unit change total_sulfur_dioxide")

    def context(self, analysis, inbox):
        return {"numbers": analysis.numbers}

    def rule_decision(self, analysis):
        return Decision(self.rule, self.action(self.rule).level)

    def act(self, decision):
        self.acted.append(decision.action)
        return Result(ok=True)

    def handle(self, msg):
        self.handled.append(msg.id)
        if msg.kind == "boom":
            raise RuntimeError("bad message")


@pytest.fixture(autouse=True)
def frozen_clock():
    clock.configure(mode="frozen", start=datetime(2026, 3, 14, tzinfo=timezone.utc))


def test_no_flag_only_heartbeat():
    agent = ToyAgent(flag=False)
    report = agent.step()
    assert not report.flagged and agent.acted == []
    assert "checker" in agent.board.last_seen


def test_flag_runs_decide_act_verify_remember():
    agent = ToyAgent()
    report = agent.step()
    assert report.decision.action == "quarantine" and report.verified
    assert report.incident_id.startswith("inc-")
    assert agent.memory.rows[0]["outcome"] == "worked"
    assert agent.action_log.rows[0]["action"] == "quarantine"


def test_valid_llm_decision_is_used():
    agent = ToyAgent(llm=FakeLLM([Decision("pass", 1, source="llm")]))
    assert agent.step().decision.source == "llm"
    assert agent.acted == ["pass"]


@pytest.mark.parametrize("bad", [Decision("delete_everything", 1), Decision("pass", 3), None])
def test_invalid_llm_decision_falls_back_to_rules(bad):
    agent = ToyAgent(llm=FakeLLM([bad]))
    report = agent.step()
    assert report.decision.action == "quarantine" and report.decision.source in ("fallback", "rules")


def test_rules_mode_never_calls_the_llm():
    llm = FakeLLM([Decision("pass", 1)])
    agent = ToyAgent(llm=llm, reason="rules")
    agent.step()
    assert llm.calls == [] and agent.acted == ["quarantine"]


def test_level_2_failed_dry_run_takes_the_safest_action():
    agent = ToyAgent(rule="repair", dry_run=lambda kind, inputs: DryRunResult(ok=False))
    report = agent.step()
    assert report.dry_run_ok is False
    assert agent.acted == ["quarantine"]


def test_level_2_passed_dry_run_acts():
    agent = ToyAgent(rule="repair")
    assert agent.step().dry_run_ok is True and agent.acted == ["repair"]


def test_level_3_asks_a_human_and_does_not_act():
    agent = ToyAgent(rule="promote")
    report = agent.step()
    assert report.waiting_for_human and agent.acted == []
    card = agent.board.by_kind("request.approval")[0]
    assert card.to == ["human"] and card.level == 3


def test_messages_are_handled_once_and_errors_do_not_crash():
    board = InMemoryBoard()
    agent = ToyAgent(board=board, flag=False)
    msg = Message(type=MessageType.QUESTION, kind="boom", sender="monitor", to=["checker"],
                  topic="feature:alcohol", level=1, summary="test")
    board.post(msg)
    agent.step()                                  # handler raises, loop survives
    agent.safe_handle(msg)                        # same id delivered again: skipped
    assert agent.handled == [msg.id]


def test_topic_subscription_delivers_broadcasts():
    board = InMemoryBoard()
    agent = ToyAgent(board=board, flag=False)
    board.post(Message(type=MessageType.FINDING, kind="data.batch_checked", sender="harness",
                       topic="dataset:batch-0002", level=0, summary="broadcast"))
    board.post(Message(type=MessageType.FINDING, kind="x", sender="harness",
                       topic="model:wine-quality@1", level=0, summary="not for checker"))
    assert len(agent.step().handled) == 1


def test_level_2_action_without_dry_run_kind_is_rejected():
    class Broken(ToyAgent):
        menu = [Action("repair", 2)]
    with pytest.raises(ValueError):
        Broken()


def test_posted_messages_use_the_shared_clock():
    agent = ToyAgent(rule="promote")
    agent.step()
    assert agent.board.by_kind("request.approval")[0].time == "2026-03-14T00:00:00Z"
