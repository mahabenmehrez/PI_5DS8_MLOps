"""
In-memory stand-ins for the shared pieces other people own, with the same
calls as the roadmap (5.7 board, 5.9 LLM helper, 5.10 dry run and experience
log). Agents use them until the real ones are merged, and unit tests keep
using them afterwards.

    InMemoryBoard   -> shared/board.py      (P6); between processes: shared/standins.SqliteBoard
    FakeLLM         -> shared/llm.py        (P4)
    fake_dry_run    -> shared/dry_run.py    (P4); inprocess_dry_run runs the real jobs
    InMemoryMemory  -> shared/memory.py     (P5)
    SimpleIncidents -> shared/incidents.py
    ActionLog       -> results log          (P6)

This InMemoryBoard keeps Message objects so tests can read m.kind; it also accepts
the plain dicts that BaseAgent.post() sends.
"""
from __future__ import annotations

import fnmatch
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from shared import clock
from shared.messages import Message, MessageStatus, MessageType


class InMemoryBoard:
    def __init__(self):
        self.messages: list[Message] = []
        self.cursors: dict[str, int] = {}
        self.subscriptions: dict[str, list[str]] = {}
        self.last_seen: dict[str, datetime] = {}

    def subscribe(self, agent: str, patterns: list[str]) -> None:
        self.subscriptions[agent] = list(patterns)

    def post(self, msg) -> str:
        from shared.base_agent import as_message
        msg = as_message(msg)
        self.messages.append(msg)
        return msg.id

    def read(self, agent: str) -> list[Message]:
        """New messages named for this agent, or broadcast on a subscribed topic."""
        start = self.cursors.get(agent, 0)
        self.cursors[agent] = len(self.messages)
        patterns = self.subscriptions.get(agent, [])
        return [m for m in self.messages[start:] if m.sender != agent and (
            agent in m.to or (not m.to and any(fnmatch.fnmatch(m.topic, p) for p in patterns)))]

    def answer(self, original, **fields) -> str:
        if isinstance(original, dict):           # find the stored message to mark it answered
            original = next(m for m in self.messages if m.id == original["id"])
        original.status = MessageStatus.ANSWERED
        fields.setdefault("summary", f"answer to {original.kind}")
        fields.setdefault("level", original.level)
        fields.setdefault("time", clock.iso())
        msg = Message(type=MessageType.ANSWER, kind=original.kind, to=[original.sender],
                      topic=original.topic, reply_to=original.id,
                      incident_id=original.incident_id, **fields)
        return self.post(msg)

    def ask(self, msg: Message, timeout_s: float = 0) -> Optional[Message]:
        """Posts the question; returns an answer already on the board, else None."""
        self.post(msg)
        return next((m for m in self.messages if m.reply_to == msg.id), None)

    def thread(self, incident_id: str) -> list[Message]:
        return [m for m in self.messages if m.incident_id == incident_id]

    def heartbeat(self, agent: str) -> None:
        self.last_seen[agent] = clock.now()

    def by_kind(self, kind: str) -> list[Message]:
        return [m for m in self.messages if m.kind == kind]


class FakeLLM:
    """Returns scripted decisions in order, then the fallback. Records every call."""

    def __init__(self, script: Optional[list] = None):
        self.script = list(script or [])
        self.calls: list[dict] = []

    def decide(self, agent, context, menu, fallback):
        self.calls.append({"agent": agent.name, "context": context})
        return self.script.pop(0) if self.script else fallback


@dataclass
class DryRunResult:
    ok: bool
    outputs: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    log_tail: str = ""
    seconds: float = 0.0


def fake_dry_run(kind, inputs, timeout_s: int = 120) -> DryRunResult:
    """Always succeeds. Real version: dry_run_jobs/<kind>.py in a subprocess."""
    return DryRunResult(ok=True, outputs={"kind": kind})


def inprocess_dry_run(kind, inputs, timeout_s: int = 120):
    """The Trainer's runner (shared/dry_run.py) in its in-process mode: the real job,
    DRY_RUN=1, a temp workdir, no subprocess. Fast enough for tests and the harness."""
    from shared.dry_run import dry_run
    return dry_run(kind, inputs, timeout_s, mode="inprocess")


class InMemoryMemory:
    def __init__(self):
        self.rows: list[dict] = []

    def remember(self, agent, incident_id, problem, evidence, action, params, level,
                 outcome, time_to_recover_s) -> str:
        mem_id = f"mem-{len(self.rows) + 1:04d}"
        self.rows.append(dict(id=mem_id, agent=agent, incident_id=incident_id, problem=problem,
                              evidence=evidence, action=action, params=params, level=level,
                              outcome=outcome, time_to_recover_s=time_to_recover_s))
        return mem_id

    def similar(self, problem: str, agent: Optional[str] = None, k: int = 5) -> list[dict]:
        words = set(problem.lower().split())
        scored = [(len(words & set(r["problem"].lower().split())), r) for r in self.rows
                  if agent is None or r["agent"] == agent]
        return [r for s, r in sorted(scored, key=lambda x: -x[0]) if s > 0][:k]


class SimpleIncidents:
    """Joins an open incident that shares a node within the window, else opens one."""

    def __init__(self):
        self.open: list[dict] = []

    def open_or_join(self, nodes: list[str], window_s: float) -> str:
        now = clock.now()
        for inc in self.open:
            if set(nodes) & inc["nodes"] and now - inc["opened_at"] <= timedelta(seconds=window_s):
                inc["nodes"] |= set(nodes)
                return inc["id"]
        inc = {"id": "inc-" + uuid.uuid4().hex[:8], "nodes": set(nodes), "opened_at": now}
        self.open.append(inc)
        return inc["id"]


class ActionLog:
    def __init__(self):
        self.rows: list[dict] = []

    def action(self, agent, incident_id, decision, result, ok) -> None:
        self.rows.append(dict(agent=agent, incident_id=incident_id, action=decision.action,
                              level=decision.level, source=decision.source, ok=ok,
                              time=clock.iso()))
