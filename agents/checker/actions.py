"""
One function per menu action of the Data Checker (roadmap 9.3).

Each action updates the batches table (state) and announces it on the board
(messages announce, tables hold state). They take the agent so they can use
its board, tables and settings, and return a Result for verify().
"""
from __future__ import annotations

import shutil
from datetime import timedelta
from pathlib import Path

import pandas as pd

from shared import clock
from shared.base_agent import Result
from shared.messages import MessageType

from agents.checker.analysis import BatchReport, repair_until_clean
from agents.checker.settings import node

ALL_AGENTS = ["setup", "reviewer", "trainer", "deployer", "monitor"]


def _expires(hours: float = 1.0) -> str:
    return (clock.now() + timedelta(hours=hours)).isoformat().replace("+00:00", "Z")


def announce(agent, report: BatchReport, status: str, path: str, repair: list | None = None) -> str:
    """data.batch_checked to the Trainer and the Monitor."""
    return agent.post(
        type=MessageType.FINDING, kind="data.batch_checked", to=["trainer", "monitor"],
        topic=f"dataset:{report.batch_id}", level=1, incident_id=agent.incident_id,
        summary=f"Batch {report.batch_id} {status}"
                + (f": {', '.join(report.failed_checks()[:3])}" if report.actionable else ""),
        evidence={"psi": report.psi, "isolation": report.isolation,
                  "findings": [f.to_dict() for f in report.actionable][:10]},
        payload={"batch_id": report.batch_id, "status": status, "rows": report.rows,
                 "failed_checks": report.failed_checks(), "psi": report.psi,
                 "repair": [s["repair"] for s in (repair or [])], "clean_path": path})


def pass_batch(agent, report: BatchReport, warning: bool = False) -> Result:
    batch = agent.tables.batch(report.batch_id)
    agent.tables.set_batch_status(report.batch_id, "passed",
                                  reason="pass_with_warning" if warning else "pass",
                                  clean_path=batch["path"], schema_version=agent.settings.schema_version,
                                  findings=[f.to_dict() for f in report.findings])
    agent.tables.add_batch_stats(report.batch_id, report.stats)
    announce(agent, report, "passed", batch["path"])
    return Result(ok=True, outputs={"status": "passed"})


def repair(agent, report: BatchReport, steps: list[dict]) -> Result:
    batch = agent.tables.batch(report.batch_id)
    s = agent.settings
    df = pd.read_parquet(batch["path"])
    repaired, applied, remaining = repair_until_clean(
        df, steps, agent.inspector, s.profile, s.bounds, s.thresholds["decision_min_score"])
    if remaining:                                 # the real run disagrees with the dry run
        reason = "repair failed: " + ", ".join(f"{f.check}:{f.column}" for f in remaining)
        return quarantine(agent, report, reason)
    clean = Path(s.data_dir) / "clean" / f"{report.batch_id}.parquet"
    clean.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_parquet(clean, index=False)
    agent.tables.use_repaired_file(report.batch_id, str(clean))    # the Trainer reads `path`
    agent.tables.set_batch_status(report.batch_id, "repaired", reason="repair", clean_path=str(clean),
                                  schema_version=s.schema_version,
                                  findings=[f.to_dict() for f in report.findings])
    agent.tables.add_batch_stats(report.batch_id, report.stats)
    announce(agent, report, "repaired", str(clean), applied)
    return Result(ok=True, outputs={"status": "repaired", "rows_before": len(df),
                                    "rows_after": len(repaired), "steps": applied})


def quarantine(agent, report: BatchReport, reason: str) -> Result:
    batch = agent.tables.batch(report.batch_id)
    target = Path(agent.settings.data_dir) / "quarantine" / Path(batch["path"]).name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(batch["path"], target)
    agent.tables.set_batch_status(report.batch_id, "quarantined", reason=reason,
                                  schema_version=agent.settings.schema_version,
                                  findings=[f.to_dict() for f in report.findings])
    announce(agent, report, "quarantined", str(target))
    do_not_use(agent, [report.batch_id], reason)
    if report.live_impact:
        claim_incident(agent, report.live_impact[0].column, [report.batch_id], report.cause)
    return Result(ok=True, outputs={"status": "quarantined", "reason": reason})


def do_not_use(agent, batch_ids: list[str], reason: str) -> str:
    return agent.post(type=MessageType.REQUEST, kind="request.do_not_use", to=["trainer"],
                      topic=f"dataset:{batch_ids[0]}", level=1, incident_id=agent.incident_id,
                      summary=f"Do not train on {', '.join(batch_ids)}"[:160],
                      payload={"batch_ids": batch_ids, "reason": reason[:300]},
                      expires_at=_expires())


def claim_incident(agent, column: str, batch_ids: list[str], cause: str) -> bool:
    """incident.claim + request.hold, once per incident: the Checker owns data-caused incidents.
    Claims are kept in the claims table, so a restarted Checker does not claim twice."""
    incident = agent.incident_id
    feature = node(column)
    since = (clock.now() - timedelta(seconds=agent.window_s)).isoformat().replace("+00:00", "Z")
    if incident is None or agent.tables.recent_claim([feature], since):
        return False
    agent.tables.record_claim(incident, feature, agent.name)
    agent.post(type=MessageType.CLAIM, kind="incident.claim", to=ALL_AGENTS,
               topic=f"incident:{incident}", incident_id=incident, level=1, confidence=0.9,
               summary=f"Data incident: {cause} on {feature.split(':')[1]} since {batch_ids[0]}"[:160],
               evidence={"batches": batch_ids, "cause": cause},
               payload={"root_cause_node": feature,
                        "evidence_refs": [f"dataset:{b}" for b in batch_ids],
                        "instructions": {"trainer": "do_not_use", "deployer": "hold",
                                         "monitor": "do not request a retrain"}})
    request_hold(agent, f"{cause} on {feature} in batch {batch_ids[0]}")
    return True


def request_hold(agent, reason: str) -> str:
    return agent.post(type=MessageType.REQUEST, kind="request.hold", to=["deployer"],
                      topic="deployment:prod", level=1, incident_id=agent.incident_id,
                      summary=f"Hold releases: {reason}"[:160],
                      payload={"reason": reason,
                               "release_condition": "new batches pass schema and unit checks"},
                      expires_at=_expires())
