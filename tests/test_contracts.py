import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = sorted((ROOT / "contracts" / "examples").rglob("*.json"))
REQUIRED = {"id", "time", "type", "kind", "sender", "to", "topic", "level", "summary", "evidence", "payload", "status", "run_id"}
PAYLOAD = {  # roadmap 6.1 and 6.2
    ("finding", "training.decision"): {"decision", "expected_gain", "threshold", "data_window", "reason_code"},
    ("finding", "model.candidate_rejected"): {"version", "overall_delta", "failing_groups"},
    ("request", "request.deploy_candidate"): {"model_uri", "version", "champion_version", "metrics", "comparison_ref"},
    ("request", "request.clean_window"): {"start", "end", "min_rows"},
    ("answer", "request.retrain"): {"decision", "detail"},
    ("answer", "request.expected_change"): {"decision", "detail", "task", "done_at"},
}


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_example(path):
    """Every example in contracts/examples, whoever sent it, fits the shared envelope."""
    from shared.messages import Message

    msg = json.loads(path.read_text())
    Message(**msg)  # the team's pydantic envelope (roadmap 5.5)
    assert REQUIRED <= set(msg)
    assert msg["kind"] == path.parent.name
    assert msg["id"].startswith("msg-") and len(msg["id"]) == 16
    assert len(msg["summary"]) <= 160 and 0 <= msg["level"] <= 3
    if msg["type"] in ("question", "request"):
        assert msg["expires_at"]
    if msg["type"] == "answer":
        assert msg["reply_to"]


TRAINER_EXAMPLES = [p for p in EXAMPLES if json.loads(p.read_text())["sender"] == "trainer"]


@pytest.mark.parametrize("path", TRAINER_EXAMPLES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_trainer_payload(path):
    msg = json.loads(path.read_text())
    assert PAYLOAD[(msg["type"], msg["kind"])] <= set(msg["payload"])
