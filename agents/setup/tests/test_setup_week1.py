"""Week-1 checks for the Setup Agent (P1): samples, stub, contract examples, baseline compose fix."""
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from agents.setup.agent import LEVELS, SetupAgent
from agents.setup.stub import SetupStub
from shared.messages import Message, MessageType

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "winequality-red.csv"
REQUIRED = {"pipeline.setup_completed": {"repo", "commit", "files", "agents_yaml", "schema_path", "depmap_seeded"},
            "pipeline.setup_failed": {"stage", "error_class", "attempts", "log_ref"}}


def rebuild_request(msg_id="msg-0123456789ab"):
    return Message(id=msg_id, type=MessageType.REQUEST, kind="request.rebuild", sender="reviewer", to=["setup"],
                   topic="pipeline:setup", level=1, summary="requirements.txt changed",
                   payload={"reason": "requirements.txt changed", "changed_paths": ["baseline/requirements.txt"]})


@pytest.mark.parametrize("mode,kind", [("default", "pipeline.setup_completed"), ("failed", "pipeline.setup_failed")])
def test_stub_posts_catalogue_kinds_with_required_fields(mode, kind):
    msg = SetupStub(mode=mode).emit()
    assert msg.kind == kind and msg.sender == "setup" and msg.topic == "pipeline:setup"
    assert REQUIRED[kind] <= set(msg.payload)
    assert len(msg.summary) <= 160


def test_stub_points_at_files_that_exist():
    p = SetupStub().emit().payload
    for path in [p["agents_yaml"], p["schema_path"], "samples/dependency_map.yaml"]:
        assert (ROOT / path).exists(), path


def test_stub_answers_rebuild_once():
    stub, req = SetupStub(), rebuild_request()
    answer, result = stub.handle(req)
    assert answer.type == MessageType.ANSWER and answer.reply_to == req.id and answer.to == ["reviewer"]
    assert answer.payload["status"] == "accepted" and result.kind == "pipeline.setup_completed"
    assert stub.handle(req) == []                       # idempotent: a repeat is skipped


@pytest.mark.parametrize("kind", sorted(REQUIRED))
def test_contract_examples_validate(kind):
    for f in (ROOT / "contracts" / "examples" / kind).glob("*.json"):
        msg = Message(**json.loads(f.read_text(encoding="utf-8")))
        assert msg.kind == kind and REQUIRED[kind] <= set(msg.payload)


def test_agent_skeleton_handles_rebuild_and_decides_by_rule():
    agent, req = SetupAgent(first_run=False), rebuild_request()
    assert agent.handle(req).payload["status"] == "accepted"
    assert agent.handle(Message(type=MessageType.FINDING, kind="data.batch_checked", sender="checker",
                                topic="dataset:batch-0001", level=0, summary="x")) is None   # unknown: ignored
    analysis = agent.analyze(agent.observe())
    assert analysis.flag and analysis.changed_paths == ["baseline/requirements.txt"]
    decision = agent.rule_decision(analysis)
    assert decision.action == "dry_run_build" and decision.level == LEVELS["dry_run_build"]
    agent.mark_handled(req)
    assert not agent.analyze(agent.observe()).flag      # handled: nothing left to do


def test_agents_yaml_sample_shape():
    cfg = yaml.safe_load((ROOT / "samples" / "agents.yaml").read_text(encoding="utf-8"))
    assert set(cfg["agents"]) == {"setup", "reviewer", "checker", "trainer", "deployer", "monitor"}
    assert cfg["agents"]["setup"]["subscriptions"] == ["pipeline:*"]
    assert len(cfg["features"]) == 11 and set(cfg["units"]) == set(cfg["features"])
    assert all(" " not in f for f in cfg["features"])


@pytest.mark.skipif(not DATA.exists(), reason="data/winequality-red.csv not downloaded")
def test_schema_accepts_real_batches_and_rejects_a_unit_change():
    import pandera
    import samples.schema as team_schema
    if not hasattr(team_schema, "validate"):
        pytest.skip("samples/schema.py is the Data Checker's now (no validate()); test_team checks make test with it")
    FEATURES, validate = team_schema.FEATURES, team_schema.validate
    df = pd.read_csv(DATA, sep=";")
    for start in range(0, len(df), 100):
        validate(df.iloc[start:start + 100])
    assert FEATURES == yaml.safe_load((ROOT / "samples" / "agents.yaml").read_text(encoding="utf-8"))["features"]
    bad = df.iloc[:100].copy()
    bad["total sulfur dioxide"] /= 1000               # mg/L -> g/L (scenario C)
    with pytest.raises(pandera.errors.SchemaErrors):
        validate(bad)


def test_baseline_db_has_a_healthcheck():
    compose = yaml.safe_load((ROOT / "baseline" / "docker-compose.yaml").read_text(encoding="utf-8"))
    # declared in the compose file, or the MySQL image's own HEALTHCHECK (mysql/mysql-server has one: the team's way)
    assert "healthcheck" in compose["services"]["db"] or "mysql/mysql-server" in compose["services"]["db"]["image"]
    waiting = [name for name, svc in compose["services"].items()
               if isinstance(svc.get("depends_on"), dict)
               and svc["depends_on"].get("db", {}).get("condition") == "service_healthy"]
    assert waiting == ["web"]                         # the MLflow server waits for a healthy db
