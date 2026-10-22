"""
Unit tests for ReviewerStub and message contracts.
"""
from shared.messages import Message, MessageType
from agents.reviewer.stub import ReviewerStub


def test_reviewer_stub_query_code_changed():
    stub = ReviewerStub(mode="scenario_a")
    ans = stub.answer_query_code_changed("total_sulfur_dioxide", "2026-10-13T09:00:00Z")
    assert len(ans["commits"]) == 1
    assert ans["commits"][0]["verdict"] == "ask_author"
    assert ans["commits"][0]["intended"] is False


def test_reviewer_stub_emit_messages():
    stub = ReviewerStub()
    msg_assessed = stub.emit_change_assessed("c123456", verdict="approve", risk_score=0.2)
    assert msg_assessed.kind == "code.change_assessed"
    assert msg_assessed.sender == "reviewer"
    assert msg_assessed.payload["commit"] == "c123456"

    msg_migration = stub.emit_migration_request("chg-1", "total_sulfur_dioxide")
    assert msg_migration.kind == "request.expected_change"
    assert msg_migration.level == 3
    assert "checker" in msg_migration.payload["tasks"]
