"""
Stub of the Deployer (P5) agent for teammates.
Posts the usual Deployer messages from fixtures and answers with canned replies.
"""
from typing import Any, Dict, List, Optional
from shared.messages import Message, MessageType

STAGES = [("shadow", 0.0), ("10%", 0.10), ("50%", 0.50)]


class DeployerStub:
    def __init__(self, mode: str = "default"):
        self.mode = mode  # "default" ou "scenario_g"
        self.published_messages: List[Message] = []

    def answer_deploy_candidate(self, request: Message) -> Message:
        """Repond accepted a request.deploy_candidate."""
        version = request.payload.get("version", "2")
        msg = Message(
            type=MessageType.ANSWER,
            kind="request.deploy_candidate",
            sender="deployer",
            to=[request.sender],
            topic=request.topic,
            reply_to=request.id,
            level=1,
            summary=f"Stub accepted candidate version {version}",
            payload={"decision": "accepted", "detail": "stub", "rollout_id": "rollout-0001"},
        )
        self.published_messages.append(msg)
        return msg

    def emit_stage_changed(self, model_version: str, from_stage: str,
                           to_stage: str, candidate_share: float) -> Message:
        """Poste deployment.stage_changed."""
        msg = Message(
            type=MessageType.FINDING,
            kind="deployment.stage_changed",
            sender="deployer",
            to=["monitor", "trainer"],
            topic="deployment:prod",
            level=2,
            summary=f"Stub moved model {model_version} from {from_stage} to {to_stage}",
            payload={
                "model_version": model_version,
                "from_stage": from_stage,
                "to_stage": to_stage,
                "candidate_share": candidate_share,
            },
        )
        self.published_messages.append(msg)
        return msg

    def run_rollout(self, model_version: str = "2") -> List[Message]:
        """Sequence shadow -> 10% -> 50%, ou rollback automatique en scenario G."""
        messages = []
        previous = "none"
        for stage, share in STAGES:
            messages.append(self.emit_stage_changed(model_version, previous, stage, share))
            previous = stage
            if self.mode == "scenario_g" and stage == "10%":
                messages.append(self.emit_rolled_back(model_version, "1"))
                break
        return messages

    def emit_rolled_back(self, from_version: str, to_version: str) -> Message:
        """Poste deployment.rolled_back (rollback automatique)."""
        msg = Message(
            type=MessageType.FINDING,
            kind="deployment.rolled_back",
            sender="deployer",
            to=[],
            topic="deployment:prod",
            level=1,
            summary=f"Stub rolled back from {from_version} to {to_version}",
            evidence={"p95_ms": 420.0},
            payload={
                "from_version": from_version,
                "to_version": to_version,
                "trigger": {"metric": "p95_ms", "value": 420.0, "limit": 300.0},
                "automatic": True,
            },
        )
        self.published_messages.append(msg)
        return msg

    def answer_query_deployment_changed(self, since: str) -> Dict[str, Any]:
        """Reponse a query.deployment_changed."""
        if self.mode == "scenario_g":
            return {"changes": [
                {"time": "2026-10-14T10:00:00Z", "from_stage": "shadow",
                 "to_stage": "10%", "model_version": "2"},
                {"time": "2026-10-14T10:40:00Z", "from_stage": "10%",
                 "to_stage": "rolled_back", "model_version": "1"},
            ]}
        return {"changes": []}