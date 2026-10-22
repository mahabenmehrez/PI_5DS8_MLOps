"""Data Checker (P3): unit, solo-scenario and migration tests (roadmap 9.6 and 13.1)."""
import contextlib
import io
import os
from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd
import pytest

from shared import clock
from shared.base_agent import Decision
from shared.messages import Message, MessageType
from shared.tables import DryRunWriteError, Tables
from shared.testing import DryRunResult, FakeLLM, InMemoryBoard, SimpleIncidents, inprocess_dry_run
from agents.checker.agent import CheckerAgent
from agents.checker.settings import CheckerSettings
from agents.checker.stub import CheckerStub
from harness.evaluate_checker import STATUS_OF, run
from harness.fault_runner import FaultRunner, all_faults
from harness.replay import ROOT, Replay
from harness.scenario_runner import run_scenario

DATA_FAULTS = [f for f in all_faults().values() if f.layer == "data"]


@pytest.fixture(autouse=True)
def frozen_clock():
    clock.configure(mode="frozen", start=datetime(2026, 3, 14, tzinfo=timezone.utc))


def make(tmp_path, fault=None, **kw):
    board, tables = InMemoryBoard(), Tables(":memory:")
    replay = Replay(tables, tmp_path / "batches", seed=kw.pop("seed", 0),
                    fault_runner=FaultRunner([fault], tables) if fault else None)
    agent = CheckerAgent(board=board, incidents=SimpleIncidents(), tables=tables,
                         settings=CheckerSettings(data_dir=tmp_path / "checker"),
                         dry_run=kw.pop("dry_run", inprocess_dry_run), **kw)
    return agent, replay, board, tables


def play(agent, replay, n):
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(n):
            replay.next_batch()
            agent.step()


def question(kind, payload, sender="monitor", topic="feature:total_sulfur_dioxide",
             mtype=MessageType.QUESTION):
    return Message(type=mtype, kind=kind, sender=sender, to=["checker"], topic=topic, level=1,
                   summary=kind, payload=payload, time=clock.iso())


@pytest.mark.parametrize("fault", DATA_FAULTS, ids=[f.id for f in DATA_FAULTS])
def test_each_data_fault_is_decided_correctly(fault, tmp_path):
    if "white_wine" in fault.inject and not (ROOT / "data" / "winequality-white.csv").exists():
        pytest.skip("needs data/winequality-white.csv")
    agent, replay, _, tables = make(tmp_path, fault)
    play(agent, replay, fault.after_batches + 1)
    row = tables.batch(f"batch-{fault.after_batches + 1:04d}")
    acceptable = fault.expected["acceptable_actions"]
    assert row["status"] in {STATUS_OF[a] for a in acceptable}
    if acceptable == ["pass_with_warning"]:
        assert row["reason"] != "pass"            # it must warn, not pass silently


def test_normal_batches_are_passed_without_alarm():
    tables = run(15, None, seed=7)
    assert [b["reason"] for b in tables.batches()] == ["pass"] * 15


def test_scenario_c_passes_solo():
    result = run_scenario("C")
    assert result.passed, result.summary()
    assert result.batches["batch-0011"] == "quarantined"
    assert result.batches["batch-0014"] == "passed"     # the feed is fixed again


def test_answers_no_change_when_data_is_clean(tmp_path):
    """Scenario D: concept drift changes labels, not inputs; the Checker must answer no."""
    agent, replay, board, _ = make(tmp_path)
    play(agent, replay, 6)
    q = question("query.data_changed", {"feature": "alcohol", "start": "", "end": ""},
                 topic="feature:alcohol")
    board.post(q)
    agent.step()
    answer = next(m for m in board.messages if m.reply_to == q.id)
    assert answer.payload == {"changed": False, "batch_ids": [], "likely_cause": "none"}
    assert not board.by_kind("incident.claim")


def test_clean_window_lists_only_clean_batches(tmp_path):
    fault = all_faults()["data.unit_change.total_sulfur_dioxide"]
    agent, replay, board, _ = make(tmp_path, fault)
    play(agent, replay, 12)
    q = question("request.clean_window", {"start": "", "end": "", "min_rows": 500},
                 sender="trainer", topic="dataset:window", mtype=MessageType.REQUEST)
    board.post(q)
    agent.step()
    answer = next(m for m in board.messages if m.reply_to == q.id)
    assert answer.payload["decision"] == "done"
    assert "batch-0011" not in answer.payload["batch_ids"]
    assert len(answer.payload["batch_ids"]) == 10 and answer.payload["rows"] == 1000


def test_expected_change_updates_the_schema(tmp_path):
    """Scenario B step: after an approved mg -> g migration, g/dm3 batches pass."""
    agent, replay, board, tables = make(tmp_path)
    play(agent, replay, 3)
    q = question("request.expected_change",
                 {"change_id": "chg-0001", "commit": "8f3e21c",
                  "changes": [{"name": "total_sulfur_dioxide", "old": "mg/dm3", "new": "g/dm3",
                               "factor": 0.001}],
                  "tasks": {"checker": "update schema"}},
                 sender="reviewer", topic="expected_change:chg-0001", mtype=MessageType.REQUEST)
    board.post(q)
    agent.step()
    answer = next(m for m in board.messages if m.reply_to == q.id)
    assert answer.payload["decision"] == "done" and answer.payload["task"] == "update schema"
    assert agent.settings.schema_version == 2

    # from batch 4 on, the feed really sends g/dm3, as planned by the migration
    grams = replace(all_faults()["data.unit_change.total_sulfur_dioxide"], after_batches=3, duration=100)
    replay.fault_runner = FaultRunner([grams])
    play(agent, replay, 2)
    assert tables.batch("batch-0004")["status"] == "passed"   # g/dm3 is the new normal
    assert not board.by_kind("incident.claim")


def test_valid_llm_decision_is_followed(tmp_path):
    fault = all_faults()["data.duplicated_rows"]
    llm = FakeLLM([Decision("quarantine", 1, {"reason": "llm prefers quarantine"}, source="llm")])
    agent, replay, _, tables = make(tmp_path, fault, llm=llm)
    play(agent, replay, 11)
    assert tables.batch("batch-0011")["status"] == "quarantined"
    assert llm.calls and "findings" in llm.calls[0]["context"]


def test_llm_context_never_contains_raw_rows(tmp_path):
    fault = all_faults()["data.duplicated_rows"]
    llm = FakeLLM()
    agent, replay, _, _ = make(tmp_path, fault, llm=llm)
    play(agent, replay, 11)
    context = llm.calls[0]["context"]
    assert not any(isinstance(v, pd.DataFrame) for v in context.values())
    assert len(str(context)) < 8000                  # ~2,000 tokens


def test_invalid_llm_decision_falls_back_to_rules(tmp_path):
    fault = all_faults()["data.duplicated_rows"]
    agent, replay, _, tables = make(tmp_path, fault, llm=FakeLLM([Decision("drop_table", 1)]))
    play(agent, replay, 11)
    assert tables.batch("batch-0011")["status"] == "repaired"


def test_failed_dry_run_falls_back_to_quarantine(tmp_path):
    fault = all_faults()["data.duplicated_rows"]
    agent, replay, _, tables = make(tmp_path, fault, dry_run=lambda kind, inputs: DryRunResult(ok=False))
    play(agent, replay, 11)
    assert tables.batch("batch-0011")["status"] == "quarantined"


def test_repair_check_dry_run_works_on_a_copy(tmp_path):
    fault = all_faults()["data.out_of_range.pH"]
    agent, replay, _, tables = make(tmp_path, fault)
    play(agent, replay, 10)
    replay.next_batch()
    agent.analyze(agent.observe())
    params = agent.repair_params(agent.report.steps)
    result = inprocess_dry_run("repair_check", {"action": "repair", "params": params})
    assert result.ok and result.metrics["rows_after"] < result.metrics["rows_before"]
    assert tables.batch("batch-0011")["status"] == "pending"   # nothing changed for real


def test_tables_refuse_writes_inside_a_dry_run():
    tables = Tables(":memory:")
    os.environ["DRY_RUN"] = "1"
    try:
        with pytest.raises(DryRunWriteError):
            tables.add_batch("batch-0001", "x.parquet", 100)
    finally:
        os.environ.pop("DRY_RUN")


def test_stub_passes_batches_and_answers_scenario_c(tmp_path):
    board, tables = InMemoryBoard(), Tables(":memory:")
    replay = Replay(tables, tmp_path / "batches")
    stub = CheckerStub(board, tables, mode="scenario_c")
    replay.next_batch()
    q = question("query.data_changed", {"feature": "total_sulfur_dioxide", "start": "", "end": ""})
    board.post(q)
    stub.step()
    assert tables.batch("batch-0001")["status"] == "passed"
    answer = next(m for m in board.messages if m.reply_to == q.id)
    assert answer.payload["likely_cause"] == "unit_change"
    assert CheckerStub.fixture("incident.claim")["kind"] == "incident.claim"
