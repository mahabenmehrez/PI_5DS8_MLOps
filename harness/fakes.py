"""
Scripted fake agents for solo scenario runs (roadmap 5.1 rule 5).

A fake posts the messages its scenario script gives it at a given batch, and
answers every request or question addressed to it with a canned answer
(default: accepted). Times in payloads may be written "{now}" or "{now-3h}".
"""
from __future__ import annotations

import re
from datetime import timedelta

from shared import clock
from shared.messages import Message, MessageType

DEFAULT_ANSWERS = {"request.": {"decision": "accepted", "detail": "fake agent"},
                   "query.": {"changed": False, "batch_ids": [], "likely_cause": "none"}}


def render(value):
    """'{now-3h}' -> ISO time 3 simulated hours ago; recursive on dicts and lists."""
    if isinstance(value, dict):
        return {k: render(v) for k, v in value.items()}
    if isinstance(value, list):
        return [render(v) for v in value]
    if isinstance(value, str):
        m = re.fullmatch(r"\{now(?:-(\d+)h)?\}", value)
        if m:
            t = clock.now() - timedelta(hours=int(m.group(1) or 0))
            return t.isoformat().replace("+00:00", "Z")
    return value


class FakeAgent:
    def __init__(self, name: str, board, incidents, script: list[dict] | None = None,
                 answers: dict | None = None):
        self.name = name
        self.board = board
        self.incidents = incidents
        self.script = script or []
        self.answers = answers or {}
        self.received: list[Message] = []

    def post_scripted(self, batch_index: int) -> None:
        for item in self.script:
            if item.get("at_batch") != batch_index:
                continue
            topic = item["topic"]
            incident = self.incidents.open_or_join([topic], 3600 * 6)
            self.board.post(Message(
                type=MessageType(item.get("type", "question")), kind=item["kind"], sender=self.name,
                to=item.get("to", []), topic=topic, incident_id=incident, level=item.get("level", 1),
                summary=item.get("summary", f"{self.name} fake: {item['kind']}")[:160],
                payload=render(item.get("payload", {})), evidence=item.get("evidence", {}),
                time=clock.iso(), expires_at=render("{now}")))

    def respond(self) -> None:
        for msg in self.board.read(self.name):
            self.received.append(msg)
            if msg.type in (MessageType.REQUEST, MessageType.QUESTION) and self.name in msg.to:
                payload = self.answers.get(msg.kind) or next(
                    (v for k, v in DEFAULT_ANSWERS.items() if msg.kind.startswith(k)), {})
                self.board.answer(msg, sender=self.name, payload=dict(payload),
                                  summary=f"{self.name} fake answers {msg.kind}")
