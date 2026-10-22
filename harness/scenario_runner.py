"""
Scenario runner (owner P3, roadmap 13 and 14). Runs scenarios/<id>.yaml with the
real agents asked for and scripted fakes for the others, then checks the
expectations: messages that must appear (any order), alternative paths
("either"), messages that must not appear, human cards, end state.

    python -m harness.scenario_runner C                 # solo: real checker, fakes for the rest
    python -m harness.scenario_runner C --agents checker
"""
from __future__ import annotations

import argparse
import contextlib
import io
import logging
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from shared import clock
from shared.tables import Tables
from shared.testing import InMemoryBoard, SimpleIncidents, inprocess_dry_run
from harness.fakes import FakeAgent
from harness.fault_runner import FaultRunner, all_faults
from harness.replay import Replay

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "scenarios"
ALL_AGENTS = ["setup", "reviewer", "checker", "trainer", "deployer", "monitor"]


def real_agent(name: str, board, incidents, tables, data_dir: Path):
    if name == "checker":
        from agents.checker.agent import CheckerAgent
        from agents.checker.settings import CheckerSettings
        return CheckerAgent(board=board, incidents=incidents, tables=tables,
                            settings=CheckerSettings(data_dir=data_dir / "checker"),
                            dry_run=inprocess_dry_run)
    raise NotImplementedError(f"agent {name!r} is not merged yet: run it as a fake")


def matches(msg, spec: dict) -> bool:
    if msg.kind != spec["kind"]:
        return False
    if "sender" in spec and msg.sender != spec["sender"]:
        return False
    if "type" in spec and msg.type.value != spec["type"]:
        return False
    if "to" in spec and not set(spec["to"]) <= set(msg.to):
        return False
    return all(msg.payload.get(k) == v for k, v in spec.get("payload", {}).items())


@dataclass
class ScenarioResult:
    scenario: str
    passed: bool
    problems: list[str] = field(default_factory=list)
    messages: list = field(default_factory=list)
    batches: dict = field(default_factory=dict)

    def summary(self) -> str:
        head = f"scenario {self.scenario}: {'PASS' if self.passed else 'FAIL'}"
        return "\n".join([head] + [f"  - {p}" for p in self.problems])


def evaluate(scenario: dict, board, tables) -> list[str]:
    expect, problems = scenario.get("expect", {}), []
    msgs = board.messages
    for spec in expect.get("messages", []):
        if not any(matches(m, spec) for m in msgs):
            problems.append(f"missing message {spec}")
    either = expect.get("either")
    if either and not any(all(any(matches(m, s) for m in msgs) for s in group) for group in either):
        problems.append(f"none of the alternative paths happened: {either}")
    for spec in expect.get("must_not", []):
        bad = [m for m in msgs if matches(m, spec)]
        if bad:
            problems.append(f"forbidden message {spec} seen {len(bad)}x")
    cards = {m.incident_id for m in msgs if m.kind == "request.approval"}
    if len(cards) > expect.get("max_human_cards", 99):
        problems.append(f"{len(cards)} human cards > {expect['max_human_cards']}")
    for batch_id, status in scenario.get("end_state", {}).get("batches", {}).items():
        row = tables.batch(batch_id)
        if not row or row["status"] != status:
            problems.append(f"batch {batch_id} is {row and row['status']}, expected {status}")
    return problems


def run_scenario(scenario_id: str, real: list[str] | None = None, verbose: bool = False) -> ScenarioResult:
    scenario = yaml.safe_load((SCENARIO_DIR / f"{scenario_id}.yaml").read_text(encoding="utf-8"))
    real = real or [scenario.get("expect", {}).get("owner", "checker")]
    fault = all_faults()[scenario["fault"]]
    n_batches = scenario.get("run", {}).get("batches") or (
        fault.after_batches + int(scenario.get("deadline", {}).get("simulated_hours", 6)))

    clock.configure(mode="frozen", start=datetime(2026, 3, 14, tzinfo=timezone.utc))
    work = Path(tempfile.mkdtemp(prefix=f"scenario_{scenario_id}_"))
    board, incidents, tables = InMemoryBoard(), SimpleIncidents(), Tables(":memory:")
    replay = Replay(tables, work / "batches", seed=scenario.get("seed", 0),
                    fault_runner=FaultRunner([fault], tables, run_id=f"solo-{scenario_id}"))
    agents = [real_agent(n, board, incidents, tables, work) for n in real]
    stubs = scenario.get("stubs", {})
    fakes = [FakeAgent(n, board, incidents, stubs.get(n, {}).get("script"),
                       stubs.get(n, {}).get("answers")) for n in ALL_AGENTS if n not in real]
    for f in fakes:
        board.subscribe(f.name, stubs.get(f.name, {}).get("subscriptions", []))

    out = io.StringIO()
    with contextlib.redirect_stdout(out if not verbose else sys.stdout):
        for i in range(1, n_batches + 1):
            replay.next_batch()
            for f in fakes:
                f.post_scripted(i)
            for _ in range(2):                    # second round handles the answers
                for a in agents:
                    a.step()
                for f in fakes:
                    f.respond()

    problems = evaluate(scenario, board, tables)
    return ScenarioResult(scenario_id, not problems, problems, list(board.messages),
                          {b["batch_id"]: b["status"] for b in tables.batches()})


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("scenario")
    p.add_argument("--agents", nargs="*", help="real agents (default: the scenario owner)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    result = run_scenario(args.scenario, args.agents, args.verbose)
    for m in result.messages:
        print(f"{m.time}  {m.sender:9s} -> {','.join(m.to) or '*':28s} {m.kind:24s} {m.summary}")
    print(result.summary())
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
