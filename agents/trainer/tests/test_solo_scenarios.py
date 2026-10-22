"""Week 2 target (roadmap 4.2): scenarios D, E and F pass with the Trainer alone, against stubs.

Each scenario replays 40 batches of 100 wines into a fresh board.db, the
Monitor stub sends request.retrain and the real Trainer handles it. MLflow is
a local file store. About a minute in total.
"""

import pytest

from agents.trainer.solo import run_scenario
from shared.llm import FakeLLM


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("solo")


def trainer_kinds(out):
    return [m["kind"] for m in out["messages"] if m["sender"] == "trainer" and m["type"] != "answer"]


def answer(out):
    return next(m for m in out["messages"] if m["type"] == "answer")["payload"]["decision"]


def test_D_concept_drift_retrains_and_forwards(workdir):
    out = run_scenario("D", workdir)
    assert trainer_kinds(out) == ["training.decision", "request.deploy_candidate"]
    deploy = next(m for m in out["messages"] if m["kind"] == "request.deploy_candidate")
    assert deploy["to"] == ["deployer"] and deploy["payload"]["model_uri"] == "models:/wine-quality@candidate"
    assert deploy["payload"]["metrics"]["overall"]["delta_pct"] <= -2
    assert answer(out) == "done"


def test_E_drift_on_unused_feature_is_refused(workdir):
    out = run_scenario("E", workdir)
    assert trainer_kinds(out) == ["training.decision"]
    finding = next(m for m in out["messages"] if m["kind"] == "training.decision")
    assert finding["payload"]["decision"] == "refuse" and finding["payload"]["reason_code"] == "gain_below_threshold"
    assert answer(out) == "refused"


def test_F_candidate_worse_for_high_alcohol_is_rejected(workdir):
    out = run_scenario("F", workdir)
    assert trainer_kinds(out) == ["training.decision", "model.candidate_rejected"]
    rejected = next(m for m in out["messages"] if m["kind"] == "model.candidate_rejected")
    failing = {(g["group"], g["band"]) for g in rejected["payload"]["failing_groups"]}
    assert ("alcohol_band", ">11.5") in failing


def test_every_score_one_point_lower_is_forwarded(workdir):
    """All labels drop by 1: wines change quality band, so quality_band 6 looks worse; only input groups can veto."""
    out = run_scenario("sudden", workdir)
    assert trainer_kinds(out) == ["training.decision", "request.deploy_candidate"]
    assert answer(out) == "done"


def test_bad_candidate_is_never_forwarded(workdir):
    out = run_scenario("bad_candidate", workdir)
    assert "request.deploy_candidate" not in trainer_kinds(out)


def test_llm_cannot_forward_a_candidate_that_failed_the_gate(workdir):
    # retrain is the right call; then the LLM tries to forward F's bad candidate
    llm = FakeLLM([{"action": "retrain_in_dry_run", "level": 2, "params": {"batch_ids": []}, "reason": "drift", "confidence": 0.7},
                   {"action": "send_to_deployer", "level": 1, "params": {"run_id": "x"}, "reason": "looks good", "confidence": 0.9}])
    out = run_scenario("F", workdir, llm_client=llm, llm_config={"provider": "fake", "db_url": None})
    assert trainer_kinds(out) == ["training.decision", "model.candidate_rejected"]
    assert out["log"][-1]["source"] == "rules"


def test_llm_refusal_is_followed(workdir):
    llm = FakeLLM([{"action": "refuse", "level": 1, "params": {"expected_gain": 1.0, "threshold": 2.0},
                   "reason": "wait for more labels", "confidence": 0.6}])
    out = run_scenario("D", workdir, llm_client=llm, llm_config={"provider": "fake", "db_url": None})
    assert trainer_kinds(out) == ["training.decision"]
    assert out["log"][-1]["source"] == "llm" and out["log"][-1]["action"] == "refuse"
