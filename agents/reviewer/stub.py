"""
Stub implementation of the Code Reviewer (Reviewer / P2) agent for teammates.
Provides canned responses and fixtures during Weeks 1 and 2.
"""
from typing import Any, Dict, List, Optional
from shared.messages import Message, MessageType


class ReviewerStub:
    """
    Fake Reviewer agent providing mock answers to teammate agents
    (Monitor, Setup Agent, Data Checker, Trainer, Deployer).
    """

    def __init__(self, mode: str = "default"):
        self.mode = mode
        self.published_messages: List[Message] = []

    def answer_query_code_changed(self, feature: str, since: str) -> Dict[str, Any]:
        """
        Answers query.code_changed questions from Monitor or Checker.
        """
        if self.mode == "scenario_a":
            return {
                "commits": [
                    {
                        "sha": "a1c2e3f",
                        "time": "2026-10-13T10:00:00Z",
                        "verdict": "ask_author",
                        "feature_changed": feature,
                        "intended": False,
                    }
                ]
            }
        elif self.mode == "scenario_b":
            return {
                "commits": [
                    {
                        "sha": "b2d4f6a",
                        "time": "2026-10-13T10:00:00Z",
                        "verdict": "approve_with_migration",
                        "feature_changed": feature,
                        "intended": True,
                    }
                ]
            }
        return {"commits": []}

    def emit_change_assessed(
        self, commit: str, verdict: str = "approve", risk_score: float = 0.1
    ) -> Message:
        """Publishes mock code.change_assessed message."""
        msg = Message(
            type=MessageType.FINDING,
            kind="code.change_assessed",
            sender="reviewer",
            to=["checker", "monitor", "trainer"],
            topic=f"commit:{commit}",
            level=1,
            summary=f"Stub assessed commit {commit} with verdict {verdict}",
            evidence={"risk_score": risk_score},
            payload={
                "commit": commit,
                "risk_score": risk_score,
                "verdict": verdict,
                "affected_features": [],
                "one_sided": False,
                "feature_diff": {},
            },
        )
        self.published_messages.append(msg)
        return msg

    def emit_migration_request(self, change_id: str, feature: str) -> Message:
        """Publishes mock request.expected_change message."""
        msg = Message(
            type=MessageType.REQUEST,
            kind="request.expected_change",
            sender="reviewer",
            to=["checker", "monitor", "trainer", "deployer"],
            topic=f"expected_change:{change_id}",
            level=3,
            summary=f"Stub migration plan for feature {feature}",
            payload={
                "change_id": change_id,
                "commit": "stub-commit",
                "changes": [
                    {"name": feature, "old": "mg/dm3", "new": "g/dm3", "factor": 0.001}
                ],
                "tasks": {
                    "checker": "update_schema",
                    "monitor": "mute_alerts",
                    "trainer": "retrain",
                    "deployer": "hold_releases",
                },
            },
        )
        self.published_messages.append(msg)
        return msg
