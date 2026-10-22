"""
Measures the Data Checker against the targets of roadmap 9.6:
>= 90% of data faults decided correctly, <= 1 false alarm per 15 normal batches.

For each data fault: replay 10 clean batches, then the faulty batch 11, with the
real Checker (rules or LLM) and fake partners. A false alarm is a normal batch
that is not plainly passed (quarantined, repaired, or passed with a warning).

    python -m harness.evaluate_checker [--normal 45] [--seeds 3]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from shared import clock
from shared.tables import Tables
from shared.testing import InMemoryBoard, SimpleIncidents, inprocess_dry_run
from harness.fault_runner import FaultRunner, all_faults
from harness.replay import ROOT, Replay

STATUS_OF = {"quarantine": "quarantined", "repair": "repaired",
             "pass": "passed", "pass_with_warning": "passed"}


def make_checker(work: Path, board, tables, llm=None):
    from agents.checker.agent import CheckerAgent
    from agents.checker.settings import CheckerSettings
    return CheckerAgent(board=board, incidents=SimpleIncidents(), tables=tables, llm=llm,
                        settings=CheckerSettings(data_dir=work / "checker"), dry_run=inprocess_dry_run)


def run(n_batches: int, fault=None, seed: int = 0, llm=None):
    clock.configure(mode="frozen", start=datetime(2026, 3, 14, tzinfo=timezone.utc))
    work = Path(tempfile.mkdtemp(prefix="eval_checker_"))
    board, tables = InMemoryBoard(), Tables(":memory:")
    runner = FaultRunner([fault], tables) if fault else None
    replay = Replay(tables, work / "batches", seed=seed, fault_runner=runner)
    checker = make_checker(work, board, tables, llm)
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(n_batches):
            replay.next_batch()
            checker.step()
    return tables


def evaluate(normal: int = 45, seeds: int = 3, llm=None) -> dict:
    rows = []
    for fid, fault in all_faults().items():
        if fault.layer != "data":
            continue
        if "white_wine" in fault.inject and not (ROOT / "data" / "winequality-white.csv").exists():
            rows.append((fid, "skipped (needs data/winequality-white.csv)", "", None))
            continue
        for seed in range(seeds):
            tables = run(fault.after_batches + 1, fault, seed, llm)
            got = tables.batch(f"batch-{fault.after_batches + 1:04d}")
            wanted = {STATUS_OF[a] for a in fault.expected.get("acceptable_actions", [])}
            acceptable = fault.expected.get("acceptable_actions", [])
            if "pass" in acceptable:
                ok = got["status"] == "passed"
            elif "pass_with_warning" in acceptable:          # must pass AND warn
                ok = got["status"] == "passed" and got["reason"] != "pass"
            else:
                ok = got["status"] in wanted
            rows.append((fid, got["status"], got["reason"][:50], ok))

    false_alarms, warned = 0, 0
    for seed in range(seeds):
        tables = run(normal // seeds, None, 100 + seed, llm)
        for b in tables.batches():
            if b["status"] != "passed":
                false_alarms += 1
            elif b["reason"] != "pass":
                warned += 1
    graded = [r for r in rows if r[3] is not None]
    return {"faults": rows, "fault_accuracy": sum(r[3] for r in graded) / max(len(graded), 1),
            "normal_batches": (normal // seeds) * seeds, "false_alarms": false_alarms,
            "warnings_on_normal": warned}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--normal", type=int, default=45)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--llm", action="store_true",
                   help="decide with the real LLM of agents.yaml (needs .env) instead of the rules")
    a = p.parse_args()
    llm, name, first_call = None, "", 0
    if a.llm:
        from harness.demo_checker import real_llm
        llm, name = real_llm()
        first_call = len(llm.CALL_LOG)
    r = evaluate(a.normal, a.seeds, llm)
    for fid, status, reason, ok in r["faults"]:
        print(f"{'ok ' if ok else ('-- ' if ok is None else 'NO ')} {fid:42s} {status:12s} {reason}")
    print(f"\nfaults decided correctly: {r['fault_accuracy']:.0%}   (target >= 90%)")
    per15 = 15 * (r["false_alarms"] + r["warnings_on_normal"]) / r["normal_batches"]
    print(f"normal batches: {r['normal_batches']}, wrongly blocked/repaired: {r['false_alarms']}, "
          f"passed with a warning: {r['warnings_on_normal']}  -> {per15:.1f} alarms per 15 (target <= 1)")
    if llm is not None:
        report_llm(llm, name, first_call)


def report_llm(llm, name: str, first_call: int = 0) -> None:
    """LLM cost and reliability, from shared/llm.py's call log (one record per attempt).
    The same records are in the llm_calls table of the board.db named by agents.yaml."""
    calls = [c for c in llm.CALL_LOG[first_call:] if c.agent == "checker"]
    by = {o: sum(c.outcome == o for c in calls) for o in ("ok", "invalid", "error", "fallback")}
    print(f"\nLLM {name}: {len(calls)} calls, {by['ok']} valid decisions, {by['invalid']} invalid "
          f"answers, {by['error']} errors, {by['fallback']} fell back to the rules")
    if calls:
        lat = [c.latency_ms / 1000 for c in calls]
        tokens = [(c.tokens_in or 0) + (c.tokens_out or 0) for c in calls]
        print(f"latency mean {sum(lat) / len(lat):.2f}s, max {max(lat):.2f}s; "
              f"tokens mean {sum(tokens) / len(tokens):.0f}, total {sum(tokens)}")
    print(f"call log: table llm_calls in {llm.load_config().get('db_url')}")


if __name__ == "__main__":
    main()
