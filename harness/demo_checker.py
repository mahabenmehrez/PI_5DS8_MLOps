"""
Demo of the Data Checker for a presentation: three faulty batches, three different decisions.

    python -m harness.demo_checker          # decisions by the rules
    python -m harness.demo_checker --llm    # decisions by the LLM of agents.yaml (needs .env)

For each case: 10 clean batches, then the faulty batch 11. Prints what the agent found,
what it decided and why, the messages it sent, and the final state of the batch.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from shared import clock
from shared.tables import Tables
from shared.testing import InMemoryBoard, SimpleIncidents, inprocess_dry_run
from harness.fault_runner import FaultRunner, all_faults
from harness.replay import Replay

CASES = [
    ("data.duplicated_rows", "20 % de lignes en double dans le lot"),
    ("data.unit_change.total_sulfur_dioxide", "le soufre arrive en g/L au lieu de mg/L (÷1000)"),
    ("data.valid_shift.white_wine", "un lot de vins BLANCS (valide, mais différent)"),
]
VERDICT = {"repaired": "RÉPARÉ", "quarantined": "MIS EN QUARANTAINE", "passed": "ACCEPTÉ"}


def run_case(fault_id: str, llm=None):
    from agents.checker.agent import CheckerAgent
    from agents.checker.settings import CheckerSettings

    clock.configure(mode="frozen", start=datetime(2026, 3, 14, tzinfo=timezone.utc))
    work = Path(tempfile.mkdtemp(prefix="demo_checker_"))
    board, tables = InMemoryBoard(), Tables(":memory:")
    replay = Replay(tables, work / "batches", fault_runner=FaultRunner([all_faults()[fault_id]]))
    agent = CheckerAgent(board=board, incidents=SimpleIncidents(), tables=tables, llm=llm,
                         settings=CheckerSettings(data_dir=work / "checker"),
                         dry_run=inprocess_dry_run)
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(10):
            replay.next_batch()
            agent.step()
        board.messages.clear()
        replay.next_batch()
        report = agent.step()
    return agent, report, board, tables


SOURCE = {"llm": "le LLM", "rules": "les règles", "fallback": "les règles (repli)"}
EVIDENCE_KEYS = ("rate", "median_ratio", "factor", "psi", "ks_p")


def case_result(fault_id: str, label: str, llm=None) -> dict:
    """One case as plain data: used by the console demo and by the web demo (demo_ui.py)."""
    _, report, board, tables = run_case(fault_id, llm)
    batch = tables.batch("batch-0011")
    findings = [f for f in json.loads(batch["findings"]) if f["score"] >= 0.5]
    d = report.decision
    return {
        "fault_id": fault_id, "label": label, "status": batch["status"],
        "verdict": VERDICT[batch["status"]],
        "findings": [{"check": f["check"], "column": f["column"] or "",
                      "evidence": {k: v for k, v in f["evidence"].items() if k in EVIDENCE_KEYS}}
                     for f in findings[:5]],
        "decision": d.action if d else "pass",
        "source": SOURCE[d.source] if d else "aucune décision (rien d'important)",
        "reason": (d.reason or "") if d else "",
        "dry_run": report.dry_run_ok,
        "messages": [{"kind": m.kind, "to": m.to} for m in board.messages if m.sender == "checker"],
    }


def show(fault_id: str, label: str, llm=None) -> None:
    r = case_result(fault_id, label, llm)
    print("=" * 78)
    print(f"Lot batch-0011 : {label}")
    print("-" * 78)
    print("Ce que l'agent a trouvé :")
    for f in r["findings"][:4]:
        print(f"  - {f['check']:15s} {f['column']:22s} {f['evidence']}")
    print(f"Décision : {r['decision']}   (décidé par {r['source']})")
    print(f"Pourquoi : {r['reason'][:200]}")
    if r["dry_run"] is not None:
        print(f"Essai sur une copie (dry run) : {'réussi' if r['dry_run'] else 'échoué'}")
    print("Messages envoyés :")
    for m in r["messages"]:
        print(f"  - {m['kind']:20s} -> {', '.join(m['to'])}")
    print(f"RÉSULTAT : {r['verdict']}   (statut = {r['status']})")


def real_llm():
    """shared/llm.py with agents.yaml / .env. Returns (llm, "provider / model");
    raises if the provider needs a key that .env does not have."""
    import os
    from shared import llm
    cfg = llm.load_config()                     # also loads .env
    if cfg.get("provider") in ("nvidia_nim", "groq", "openai"):
        key_env = cfg.get("api_key_env") or "NVIDIA_API_KEY"
        if not os.environ.get(key_env):
            raise RuntimeError(f"{key_env} is not set: copy .env.example to .env and add your key")
    return llm, f"{cfg.get('provider')} / {cfg.get('model')}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", action="store_true", help="decide with the LLM of agents.yaml (.env)")
    args = p.parse_args()
    llm = None
    if args.llm:
        llm, name = real_llm()
        print(f"Décisions par le LLM : {name}\n")
    for fault_id, label in CASES:
        show(fault_id, label, llm)
    print("=" * 78)


if __name__ == "__main__":
    main()
