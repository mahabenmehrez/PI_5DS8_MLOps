"""Fake Trainer (roadmap 5.1 rule 5 and 10.6) so the other five agents can test alone.

- answers request.retrain with accepted, then posts a training.decision
  (retrain or refuse, chosen by the fixture set);
- after a retrain, posts request.deploy_candidate for a hand-registered
  version (or model.candidate_rejected with the "reject" fixture set);
- answers request.expected_change with done.

Fixture set: STUB_FIXTURE=retrain (default, scenario D) | refuse (E) | reject (F).

    python -m agents.trainer.stub            # needs the real board (shared/board.py)
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE_SETS = {
    "retrain": ["training_decision_retrain.json", "deploy_candidate.json"],
    "refuse": ["training_decision_refuse.json"],
    "reject": ["training_decision_retrain.json", "candidate_rejected.json"],
}
ENVELOPE_KEYS = ("id", "time", "status", "reply_to")


def load_fixture(name: str) -> dict:
    msg = json.loads((FIXTURES / name).read_text())
    for key in ENVELOPE_KEYS:  # the board fills these in
        msg.pop(key, None)
    return msg


class TrainerStub:
    name = "trainer"

    def __init__(self, board, fixture_set: str | None = None):
        self.board = board
        self.fixture_set = fixture_set or os.environ.get("STUB_FIXTURE", "retrain")
        if self.fixture_set not in FIXTURE_SETS:
            raise ValueError(f"unknown fixture set {self.fixture_set!r}; use one of {sorted(FIXTURE_SETS)}")
        self.handled: set[str] = set()

    def handle(self, msg: dict) -> None:
        if msg["id"] in self.handled or msg["type"] == "answer":
            return
        self.handled.add(msg["id"])
        if msg["kind"] == "request.retrain":
            self.board.answer(msg, sender=self.name, level=1, summary="Retrain request accepted (stub)",
                              payload={"decision": "accepted", "detail": f"stub fixture set {self.fixture_set}"})
            for name in FIXTURE_SETS[self.fixture_set]:
                out = copy.deepcopy(load_fixture(name))
                out["incident_id"] = msg.get("incident_id")
                out["run_id"] = msg.get("run_id", "dev")
                self.board.post(out)
        elif msg["kind"] == "request.expected_change":
            self.board.answer(msg, sender=self.name, level=1, summary="Expected change handled (stub)",
                              payload={"decision": "done", "detail": "stub", "task": "retrain_on_post_change_data",
                                       "done_at": msg["time"]})

    def step(self) -> None:
        for msg in self.board.read(self.name):
            self.handle(msg)


def main() -> None:
    from shared.board import Board  # owner P6; not merged yet

    stub = TrainerStub(Board())
    while True:
        stub.step()
        time.sleep(2)


if __name__ == "__main__":
    main()
