"""
Fake Data Checker for teammates (roadmap 9.6): marks every pending batch passed and
posts data.batch_checked; answers query.data_changed with a canned answer
("unit change on total_sulfur_dioxide since batch X" in mode scenario_c, else
"no change"); accepts request.clean_window and request.expected_change.

    from agents.checker.stub import CheckerStub
    stub = CheckerStub(board, tables, mode="scenario_c", changed_since="batch-0011")
    stub.step()                    # call it every round, like a real agent
"""
from __future__ import annotations

import json
from pathlib import Path

from shared import clock
from shared.base_agent import as_message
from shared.messages import Message, MessageType

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class CheckerStub:
    name = "checker"

    def __init__(self, board, tables=None, mode: str = "default", changed_since: str = "batch-0011"):
        self.board = board
        self.tables = tables
        self.mode = mode
        self.changed_since = changed_since
        if hasattr(board, "subscribe"):
            board.subscribe(self.name, ["dataset:*", "feature:*", "expected_change:*"])

    def step(self) -> None:
        for raw in self.board.read(self.name):     # dicts (SqliteBoard) or Messages
            self.handle(raw)
        if self.tables is None:
            return
        for batch in self.tables.pending_batches():
            self.tables.set_batch_status(batch["batch_id"], "passed", reason="stub",
                                         clean_path=batch["path"])
            self.board.post(Message(
                type=MessageType.FINDING, kind="data.batch_checked", sender=self.name,
                to=["trainer", "monitor"], topic=f"dataset:{batch['batch_id']}", level=1,
                summary=f"Batch {batch['batch_id']} passed (stub)", time=clock.iso(),
                payload={"batch_id": batch["batch_id"], "status": "passed", "rows": batch["rows"],
                         "failed_checks": [], "psi": {}, "repair": [], "clean_path": batch["path"]},
            ).model_dump(mode="json"))

    def answer(self, raw, **fields) -> None:
        self.board.answer(raw, sender=self.name, time=clock.iso(), **fields)

    def handle(self, raw) -> None:
        msg = as_message(raw)
        if msg.type == MessageType.ANSWER:
            return
        if msg.kind == "query.data_changed":
            if self.mode == "scenario_c":
                payload = {"changed": True, "batch_ids": [self.changed_since],
                           "likely_cause": "unit_change"}
                summary = f"Yes: unit change on total_sulfur_dioxide since {self.changed_since}"
            else:
                payload = {"changed": False, "batch_ids": [], "likely_cause": "none"}
                summary = "No data change"
            self.answer(raw, summary=summary, payload=payload, level=1)
        elif msg.kind == "request.clean_window":
            ids = [b["batch_id"] for b in (self.tables.batches("passed") if self.tables else [])]
            self.answer(raw, summary=f"{len(ids)} clean batches", level=1,
                              payload={"decision": "done", "detail": "stub", "batch_ids": ids,
                                       "rows": 100 * len(ids)})
        elif msg.kind == "request.expected_change":
            self.answer(raw, summary="Schema updated (stub)", level=2,
                              payload={"decision": "done", "detail": "stub",
                                       "task": msg.payload.get("tasks", {}).get(self.name, ""),
                                       "done_at": clock.iso()})

    @staticmethod
    def fixture(kind: str) -> dict:
        """Example outgoing message of the real Checker (agents/checker/fixtures/<kind>.json)."""
        return json.loads((FIXTURES / f"{kind}.json").read_text(encoding="utf-8"))
