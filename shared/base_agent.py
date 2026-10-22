"""
Base agent (owner P3): the 7-step loop every agent runs (roadmap 3.2 and 5.8).

    1 observe   -> 2 analyze -> 3 decide (LLM, rule fallback) -> 4 dry_run (level 2)
    -> 5 act (level 3 asks a human) -> 6 verify -> 7 remember

Each agent subclasses BaseAgent and writes six methods: observe, analyze,
context, rule_decision, act, handle. The loop, the action levels, the dry-run
gate, idempotent message handling, logging and memory are written once here,
so no agent can skip them.

The board (P6), LLM helper and dry-run runner (P4), experience log (P5) and
incidents are passed in. Until the real ones exist, the in-memory stand-ins
from shared/testing.py are used, so every agent can run alone.

Two message formats meet here: boards that store plain dicts (shared/standins.py
SqliteBoard, used between processes) and our in-memory board of Message objects.
step() hands our agents a Message, posts go out as dicts, and reply() answers a
message in the format its board gave it. Agents that keep their own loop (the
Trainer) can still subclass BaseAgent with dict menus and dict messages.
"""
from __future__ import annotations

import dataclasses
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from shared import clock
from shared.messages import Message, MessageType

log = logging.getLogger("agents")

LEVEL_INFO, LEVEL_SAFE, LEVEL_DRY_RUN, LEVEL_HUMAN = 0, 1, 2, 3

# shared/llm.py reports why it fell back; for us all of them are a fallback to the rules
FALLBACK_SOURCES = {"llm_invalid": "fallback", "llm_error": "fallback"}


def as_message(msg: Any) -> Message:
    """A board message, dict or Message, as a Message."""
    if isinstance(msg, Message):
        return msg
    return Message(**{k: v for k, v in msg.items() if k in Message.model_fields})


def message_id(msg: Any) -> str:
    return msg["id"] if isinstance(msg, dict) else msg.id


@dataclass(frozen=True)
class Action:
    """One entry of an agent's menu. The LLM may only pick actions from the menu."""
    name: str
    level: int                                   # 0 info, 1 safe, 2 dry run first, 3 human
    params_schema: dict = field(default_factory=dict)
    success_check: str = ""                      # written before acting, checked by verify()
    dry_run_kind: Optional[str] = None           # dry_run_jobs/<kind>.py, required for level 2
    safest: bool = False                         # the fallback when a dry run fails


@dataclass
class Analysis:
    """Output of step 2: plain numbers, a yes/no flag and the dependency-map nodes involved."""
    flag: bool
    nodes: list[str] = field(default_factory=list)
    numbers: dict = field(default_factory=dict)
    problem: str = ""                            # short text, used to find similar past cases


@dataclass
class Decision:
    action: str
    level: int
    params: dict = field(default_factory=dict)
    reason: str = ""
    confidence: Optional[float] = None
    success_check: str = ""
    memory_refs: list[str] = field(default_factory=list)
    source: str = "rules"                        # rules | llm | fallback


@dataclass
class Result:
    ok: bool
    outputs: dict = field(default_factory=dict)
    detail: str = ""


@dataclass
class StepReport:
    """What one round of the loop did. Returned by step(); useful in tests and logs."""
    handled: list[str] = field(default_factory=list)
    flagged: bool = False
    incident_id: Optional[str] = None
    decision: Optional[Decision] = None
    dry_run_ok: Optional[bool] = None
    waiting_for_human: bool = False
    result: Optional[Result] = None
    verified: Optional[bool] = None
    memory_id: Optional[str] = None


def to_decision(answer: Any) -> Optional[Decision]:
    """What an LLM helper returns (our Decision, a dict, or shared/llm.py's pydantic Decision)
    as our Decision; None stays None so valid() rejects it."""
    if answer is None or isinstance(answer, Decision):
        return answer
    data = answer.model_dump() if hasattr(answer, "model_dump") else dict(answer)
    known = {f.name for f in dataclasses.fields(Decision)}
    decision = Decision(**{k: v for k, v in data.items() if k in known})
    decision.source = FALLBACK_SOURCES.get(decision.source, decision.source)
    return decision


class BaseAgent:
    name: str = "base"
    menu: list[Action] = []
    subscriptions: list[str] = []                # topic globs, e.g. "dataset:*"

    def __init__(self, board=None, llm=None, dry_run: Optional[Callable] = None,
                 memory=None, incidents=None, action_log=None,
                 reason: str = "llm", window_s: float = 3600.0):
        from shared import testing                # stand-ins until the real pieces exist
        self.board = board or testing.InMemoryBoard()
        self.llm = llm or testing.FakeLLM()
        self.dry_run = dry_run or testing.fake_dry_run
        self.memory = memory or testing.InMemoryMemory()
        self.incidents = incidents or testing.SimpleIncidents()
        self.action_log = action_log or testing.ActionLog()
        self.reason = reason                     # "llm", or "rules" for the traditional system
        self.window_s = window_s
        self.inbox: list[Message] = []
        self.incident_id: Optional[str] = None   # incident of the current round, for act()
        self._handled_ids: set[str] = set()      # ids of the messages already handled
        self._raw: dict[str, Any] = {}           # message id -> the message as its board gave it
        for a in self.menu:
            if isinstance(a, Action) and a.level == LEVEL_DRY_RUN and not a.dry_run_kind:
                raise ValueError(f"{self.name}: level-2 action {a.name!r} needs a dry_run_kind")
        if hasattr(self.board, "subscribe"):
            self.board.subscribe(self.name, self.subscriptions)

    # ---- each agent writes these --------------------------------------------
    def observe(self) -> dict:
        """1. Look: read inputs (new batch, last hour of predictions, latest commit)."""
        raise NotImplementedError

    def analyze(self, obs: dict) -> Analysis:
        """2. Understand: statistics and checks, no LLM. No flag means nothing to do."""
        raise NotImplementedError

    def context(self, analysis: Analysis, inbox: list[Message]) -> dict:
        """What the LLM sees: numbers and memory matches, never raw rows."""
        raise NotImplementedError

    def rule_decision(self, analysis: Analysis) -> Decision:
        """Fallback when the LLM fails AND the decision of the 'rules only' system."""
        raise NotImplementedError

    def act(self, decision: Decision) -> Result:
        """5. Do: call the function of actions.py for this decision."""
        raise NotImplementedError

    def handle(self, msg: Message) -> None:
        """Questions and requests from other agents."""
        raise NotImplementedError

    def verify(self, decision: Decision, result: Result) -> bool:
        """6. Check. Override to test the action's success_check for real."""
        return result.ok

    # ---- given by the base class ----------------------------------------------
    def step(self) -> StepReport:
        report = StepReport()
        self.inbox = []
        self.incident_id = None
        for raw in self.board.read(self.name):
            try:
                msg = as_message(raw)
            except Exception:                    # a malformed message must not stop the loop
                log.exception("%s: unreadable message %r", self.name, raw)
                continue
            self._raw[msg.id] = raw
            if self.safe_handle(msg):
                report.handled.append(msg.id)

        analysis = self.analyze(self.observe())                          # 1 + 2
        if not analysis.flag:
            self.board.heartbeat(self.name)
            return report
        report.flagged = True
        incident = self.incidents.open_or_join(analysis.nodes, self.window_s)
        report.incident_id = self.incident_id = incident

        decision = self.decide(analysis)                                 # 3
        if decision.level == LEVEL_DRY_RUN:                              # 4
            action = self.action(decision.action)
            outcome = self.dry_run(action.dry_run_kind, {"action": decision.action,
                                                         "params": decision.params})
            report.dry_run_ok = bool(outcome.ok)
            if not outcome.ok:
                log.info("%s: dry run of %s failed, falling back to the safest action",
                         self.name, decision.action)
                decision = self.safest(analysis, reason=f"dry run of {decision.action} failed")
        report.decision = decision

        if decision.level == LEVEL_HUMAN:                                # 5 (level 3)
            self.ask_human(incident, decision)
            report.waiting_for_human = True
            return report

        started = time.monotonic()
        result = self.act(decision)                                      # 5
        ok = self.verify(decision, result)                               # 6
        report.result, report.verified = result, ok
        report.memory_id = self.memory.remember(                         # 7
            agent=self.name, incident_id=incident, problem=analysis.problem,
            evidence=analysis.numbers, action=decision.action, params=decision.params,
            level=decision.level, outcome="worked" if ok else "failed",
            time_to_recover_s=time.monotonic() - started)
        self.action_log.action(self.name, incident, decision, result, ok)
        return report

    def decide(self, analysis: Analysis) -> Decision:
        """3. The LLM picks one menu action; any problem -> the rule decision."""
        fallback = self.rule_decision(analysis)
        fallback.source = "rules"
        if self.reason == "rules":
            return fallback
        kwargs = {}
        if "incident_id" in inspect.signature(self.llm.decide).parameters:
            kwargs["incident_id"] = self.incident_id       # shared/llm.py logs it in llm_calls
        try:
            answer = self.llm.decide(self, self.context(analysis, self.inbox), self.menu,
                                     fallback=dataclasses.asdict(fallback), **kwargs)
            decision = to_decision(answer)
        except Exception:                        # the loop must never stop on the LLM
            log.exception("%s: llm.decide failed, using the rule decision", self.name)
            return fallback
        if not self.valid(decision):
            log.warning("%s: invalid LLM decision %r, using the rule decision", self.name, decision)
            fallback.source = "fallback"
            return fallback
        return decision

    def valid(self, decision: Optional[Decision]) -> bool:
        """The action must exist in the menu, with the menu's level."""
        if decision is None:
            return False
        try:
            return self.action(decision.action).level == decision.level
        except KeyError:
            return False

    def action(self, name: str) -> Action:
        for a in self.menu:
            if a.name == name:
                return a
        raise KeyError(f"{self.name} has no action {name!r} in its menu")

    def safest(self, analysis: Analysis, reason: str = "") -> Decision:
        """The menu action marked safest, else the lowest-level one."""
        candidates = [a for a in self.menu if a.safest] or sorted(self.menu, key=lambda a: a.level)
        a = candidates[0]
        return Decision(a.name, a.level, {}, reason=reason, success_check=a.success_check,
                        source="fallback")

    def safe_handle(self, msg: Any) -> bool:
        """Handles each message once, and never lets a bad message crash the loop.
        Takes a Message, or a dict for agents that keep their own loop (the Trainer)."""
        mid = message_id(msg)
        if mid in self._handled_ids:
            return False
        self._handled_ids.add(mid)
        if isinstance(msg, Message):
            self.inbox.append(msg)
        try:
            self.handle(msg)
        except Exception:
            kind = msg.get("kind") if isinstance(msg, dict) else msg.kind
            log.exception("%s: error while handling %s (%s)", self.name, mid, kind)
        return True

    def reply(self, msg: Message, **fields: Any) -> str:
        """Answers a message through the board, in the format the board gave it."""
        original = self._raw.get(msg.id, msg)
        fields.setdefault("sender", self.name)
        fields.setdefault("time", clock.iso())
        return self.board.answer(original, **fields)

    def ask_human(self, incident_id: Optional[str], decision: Decision) -> str:
        """Level 3: one request.approval card; the human answers through the CLI."""
        return self.post(
            type=MessageType.REQUEST, kind="request.approval", to=["human"],
            topic=f"incident:{incident_id}", incident_id=incident_id, level=LEVEL_HUMAN,
            summary=f"{self.name} asks approval for {decision.action}"[:160],
            evidence={"reason": decision.reason},
            payload={"question": f"Approve {decision.action}?", "options": ["approve", "reject"],
                     "default_if_no_answer": "reject", "card_ref": incident_id,
                     "action": decision.action, "params": decision.params})

    def post(self, *, type: MessageType, kind: str, topic: str, summary: str,
             to: Optional[list[str]] = None, level: int = LEVEL_INFO, payload: Optional[dict] = None,
             evidence: Optional[dict] = None, **fields: Any) -> str:
        """Builds a message from this agent with the shared clock's time, and posts it."""
        msg = Message(type=type, kind=kind, sender=self.name, to=to or [], topic=topic,
                      summary=summary, level=level, payload=payload or {},
                      evidence=evidence or {}, time=clock.iso(), **fields)
        return self.board.post(msg.model_dump(mode="json"))     # boards store plain dicts

    def run(self, poll_s: float = 2.0, max_steps: Optional[int] = None) -> None:
        """Main loop: one step every poll_s seconds (2 s in integration mode)."""
        n = 0
        while max_steps is None or n < max_steps:
            try:
                self.step()
            except Exception:                    # one bad round must not stop the agent
                log.exception("%s: step failed", self.name)
            n += 1
            time.sleep(poll_s)
