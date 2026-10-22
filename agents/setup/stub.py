"""
Stub (fake) Setup Agent (P1) for teammates, weeks 1 and 2 (roadmap 5.1 rule 5, 7.6).

Posts pipeline.setup_completed pointing at the hand-made sample files, and answers request.rebuild
with "accepted" then a new pipeline.setup_completed. No build, no LLM: canned messages from fixtures/.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.messages import Message, MessageType

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TOPIC = "pipeline:setup"


def _fixture(name: str) -> Dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class SetupStub:
    """Fake Setup Agent: modes "default" (setup succeeds) and "failed" (setup gives up)."""

    def __init__(self, mode: str = "default"):
        self.mode = mode
        self.published_messages: List[Message] = []
        self._handled: set = set()

    def _publish(self, msg: Message) -> Message:
        self.published_messages.append(msg)
        return msg

    def emit_setup_completed(self, commit: Optional[str] = None) -> Message:
        payload = _fixture("setup_completed.json")
        if commit:
            payload["commit"] = commit
        return self._publish(Message(
            type=MessageType.FINDING, kind="pipeline.setup_completed", sender="setup", to=[],
            topic=TOPIC, level=1, confidence=1.0,
            summary=f"Stub: pipeline ready for {payload['repo']} (sample files), commit {payload['commit'][:7]}",
            evidence={"stages_passed": 9, "fix_attempts": 0}, payload=payload,
        ))

    def emit_setup_failed(self) -> Message:
        payload = _fixture("setup_failed.json")
        return self._publish(Message(
            type=MessageType.FINDING, kind="pipeline.setup_failed", sender="setup", to=["human"],
            topic=TOPIC, level=1, confidence=1.0,
            summary=f"Stub: setup failed at {payload['stage']} ({payload['error_class']}) after {payload['attempts']} fixes",
            evidence={"attempts": payload["attempts"]}, payload=payload,
        ))

    def emit(self) -> Message:
        return self.emit_setup_failed() if self.mode == "failed" else self.emit_setup_completed()

    def handle(self, msg: Message) -> List[Message]:
        """request.rebuild -> "accepted" answer, then the setup result. Idempotent; other kinds ignored."""
        if msg.kind != "request.rebuild" or msg.id in self._handled:
            return []
        self._handled.add(msg.id)
        answer = self._publish(Message(
            type=MessageType.ANSWER, kind="request.rebuild", sender="setup", to=[msg.sender],
            topic=TOPIC, reply_to=msg.id, incident_id=msg.incident_id, level=1,
            summary="Stub: rebuild accepted",
            payload={"status": "accepted", "changed_paths": msg.payload.get("changed_paths", [])},
        ))
        return [answer, self.emit()]
