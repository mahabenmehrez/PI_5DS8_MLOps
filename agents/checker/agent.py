"""
Data Checker (P3): pass, repair or quarantine every incoming batch (roadmap 9).

    observe  : next `pending` row of the batches table
    analyze  : schema, statistics, nulls, ranges, unit changes, renames, duplicates,
               PSI/KS drift, Isolation Forest (analysis.py, no LLM). A clean batch is
               passed right there; anything else raises the flag.
    decide   : LLM (prompt.md) with rule_decision() as fallback and as the 'rules' system
    dry_run  : `repair_check` for repairs and schema updates (level 2)
    act      : actions.py
    handle   : query.data_changed, request.clean_window, request.expected_change

    python -m agents.checker
"""
from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

from shared import clock
from shared.base_agent import Action, Analysis, BaseAgent, Decision, Result
from shared.messages import Message, MessageType
from shared.tables import Tables

from agents.checker import actions
from agents.checker.analysis import BatchReport, StatsAnomaly, analyze_batch, likely_cause
from agents.checker.checks import ALERT, DRIFT_CHECKS, Inspector, match_unit_factor
from agents.checker.repairs import REPAIRS
from agents.checker.settings import (FEATURE_COLUMNS, ROOT, CheckerSettings, column_of, node,
                                     reference_frame)
from samples.schema import make_schema

log = logging.getLogger("agents.checker")


class ClaimedIncidents:
    """Incidents that survive a restart: a problem on a feature the Checker already claimed
    (claims table in board.db) keeps that incident id; anything else asks the wrapped incidents."""

    def __init__(self, inner, tables: Tables):
        self.inner, self.tables = inner, tables

    def open_or_join(self, nodes: list[str], window_s: float) -> str:
        since = (clock.now() - timedelta(seconds=window_s)).isoformat().replace("+00:00", "Z")
        claimed = self.tables.recent_claim([n for n in nodes if n.startswith("feature:")], since)
        return claimed or self.inner.open_or_join(nodes, window_s)


class CheckerAgent(BaseAgent):
    name = "checker"
    subscriptions = ["dataset:*", "feature:*", "expected_change:*"]
    menu = [
        Action("pass", 1, success_check="registry row passed"),
        Action("pass_with_warning", 1, success_check="row passed, warning posted"),
        Action("quarantine", 1, success_check="batch absent from the next training window",
               safest=True),
        Action("repair", 2, dry_run_kind="repair_check",
               success_check="repaired copy passes all checks and PSI < 0.25 on the repaired feature"),
        Action("warn", 1, success_check="data.drift_trend or finding delivered"),
        Action("request_hold", 1, success_check="Deployer answers accepted (incident owner only)"),
        Action("update_schema", 2, dry_run_kind="repair_check",
               success_check="only after request.expected_change; next batch passes"),
    ]

    def __init__(self, tables: Optional[Tables] = None, settings: Optional[CheckerSettings] = None,
                 **kw):
        kw.setdefault("window_s", 6 * 3600)      # one incident per problem over 6 simulated hours
        super().__init__(**kw)
        self.tables = tables or Tables(str(ROOT / "board.db"))
        self.settings = settings or CheckerSettings()
        self.inspector = self._build_inspector()
        self.anomaly = StatsAnomaly(FEATURE_COLUMNS, self.settings.thresholds["isolation_min_batches"])
        self.report: Optional[BatchReport] = None
        self.incidents = ClaimedIncidents(self.incidents, self.tables)
        self.settings.save_profile()

    def _build_inspector(self) -> Inspector:
        s = self.settings
        return Inspector(s.profile, s.inspector_cfg, make_schema(s.bounds))

    # ---- 1. Look ---------------------------------------------------------------
    def observe(self) -> dict:
        pending = self.tables.pending_batches()
        if not pending:
            return {}
        batch = pending[0]
        return {"batch_id": batch["batch_id"], "path": batch["path"],
                "df": pd.read_parquet(batch["path"])}

    # ---- 2. Understand ---------------------------------------------------------
    def analyze(self, obs: dict) -> Analysis:
        self.report = None
        if not obs:
            return Analysis(flag=False)
        normal = [b["batch_id"] for b in self.tables.batches("passed") if b["reason"] == "pass"]
        self.anomaly.fit(self.tables.batch_stats(), normal)
        s = self.settings
        report = analyze_batch(obs["batch_id"], obs["df"], self.inspector, s.profile, s.bounds,
                               FEATURE_COLUMNS, s.thresholds["decision_min_score"], self.anomaly)
        self.report = report
        if not report.actionable:      # nothing that matters for the model: pass without the LLM
            actions.pass_batch(self, report)
            return Analysis(flag=False)
        nodes = [f"dataset:{report.batch_id}"] + sorted({node(f.column) for f in report.actionable
                                                         if f.column in FEATURE_COLUMNS})
        return Analysis(flag=True, nodes=nodes,
                        numbers={"rows": report.rows, "failed_checks": report.failed_checks(),
                                 "psi": report.psi, "isolation": report.isolation},
                        problem=f"{report.cause} " + " ".join(report.failed_checks()))

    # ---- 3. Decide ---------------------------------------------------------------
    def context(self, analysis: Analysis, inbox: list[Message]) -> dict:
        """What the LLM sees (prompt.md): numbers only, never rows."""
        r = self.report
        ranked = sorted(self.settings.importance.items(), key=lambda kv: -kv[1])
        return {
            "batch_id": r.batch_id, "rows": r.rows, "likely_cause": r.cause,
            "findings": [f.to_dict() for f in sorted(r.actionable, key=lambda f: -f.score)][:10],
            "psi_with_importance_rank": {c: {"psi": v, "rank": [k for k, _ in ranked].index(c) + 1}
                                         for c, v in r.psi.items() if c in self.settings.importance},
            "isolation_forest": r.isolation,
            "candidate_repairs": r.steps,
            "unfixable": [f.to_dict() for f in r.unfixable],
            "live_impact": [f"{f.check}:{f.column}" for f in r.live_impact],
            "similar_past_cases": self.memory.similar(analysis.problem, agent=self.name, k=3),
            "inbox": [{"kind": m.kind, "sender": m.sender, "summary": m.summary} for m in inbox][:5],
        }

    def rule_decision(self, analysis: Analysis) -> Decision:
        """Fallback when the LLM fails, and the whole decision in the 'rules' system.

        - an error that also reaches the live API (unit change, missing column, type) or that
          no repair can fix -> quarantine (+ claim and hold for the live ones);
        - other errors (duplicates, a few bad rows, a small null spike, rename) -> repair;
        - drift, anomalies, new columns, minor warnings -> pass with a warning."""
        r = self.report
        if r.unfixable or r.live_impact:
            blocking = r.live_impact or r.unfixable
            reason = f"{r.cause}: " + ", ".join(f"{f.check}:{f.column}" for f in blocking)
            return Decision("quarantine", 1, {"reason": reason}, reason=reason,
                            success_check="batch absent from the next training window")
        if r.errors:
            return Decision("repair", 2, self.repair_params(r.steps),
                            reason="repairable: " + ", ".join(s["repair"] for s in r.steps),
                            success_check="repaired copy passes all checks")
        if r.drift_only:
            return Decision("warn", 1, {}, reason="valid shift: " + ", ".join(r.failed_checks()),
                            success_check="data.drift_trend delivered")
        return Decision("pass_with_warning", 1, {}, reason="warnings only: " + ", ".join(r.failed_checks()))

    def repair_params(self, steps: list[dict]) -> dict:
        batch = self.tables.batch(self.report.batch_id)
        return {"batch_id": self.report.batch_id, "path": batch["path"], "steps": steps,
                "profile_path": str(self.settings.profile_path)}

    def decide(self, analysis: Analysis) -> Decision:
        """The LLM chooses the action; the agent fills in the technical parameters it cannot
        know (batch path, profile version). Repair steps come from the analysis unless the
        LLM gave valid ones."""
        decision = super().decide(analysis)
        if decision.action == "repair":
            steps = decision.params.get("steps")
            valid = isinstance(steps, list) and steps and all(
                isinstance(s, dict) and s.get("repair") in REPAIRS for s in steps)
            decision.params = self.repair_params(steps if valid else self.report.steps)
        return decision

    # ---- 5. Do -------------------------------------------------------------------
    def act(self, decision: Decision) -> Result:
        r, p = self.report, decision.params
        if decision.action == "pass":
            return actions.pass_batch(self, r)
        if decision.action == "pass_with_warning":
            return actions.pass_batch(self, r, warning=True)
        if decision.action == "quarantine":
            return actions.quarantine(self, r, p.get("reason") or decision.reason or r.cause)
        if decision.action == "repair":
            return actions.repair(self, r, p.get("steps") or r.steps)
        if decision.action == "warn":
            result = actions.pass_batch(self, r, warning=True)
            self.post_drift_trend(r)
            return result
        if decision.action == "request_hold":
            actions.request_hold(self, decision.reason or r.cause)
            return actions.quarantine(self, r, decision.reason or r.cause)
        return Result(ok=False, detail=f"{decision.action} is only used on request.expected_change")

    # ---- 6. Check ----------------------------------------------------------------
    def verify(self, decision: Decision, result: Result) -> bool:
        expected = {"quarantine": "quarantined", "request_hold": "quarantined",
                    "repair": "repaired"}.get(decision.action, "passed")
        row = self.tables.batch(self.report.batch_id)
        # a repair that failed for real ends as a quarantine: safe, but the action did not work
        return result.ok and row is not None and row["status"] == expected

    # ---- questions and requests from the others --------------------------------
    def handle(self, msg: Message) -> None:
        if msg.type.value == "answer":
            return
        handler = {"query.data_changed": self.answer_data_changed,
                   "request.clean_window": self.answer_clean_window,
                   "request.expected_change": self.answer_expected_change}.get(msg.kind)
        if handler:
            handler(msg)
        else:
            log.debug("checker ignores %s from %s", msg.kind, msg.sender)

    def answer_data_changed(self, msg: Message) -> None:
        """Compare the window's batches with the batches before it, for one feature."""
        column = column_of(msg.payload.get("feature", msg.topic))
        start, end = msg.payload.get("start") or "", msg.payload.get("end") or "￿"
        checked = [b for b in self.tables.batches() if b["status"] != "pending"]
        window = [b for b in checked if start <= (b["created_at"] or "") <= end] or checked[-1:]
        first = checked.index(window[0]) if window else 0
        before = checked[max(0, first - self.settings.thresholds["compare_last_batches"]):first]

        hits, checks = [], []
        for b in window:
            serious = [f["check"] for f in json.loads(b["findings"] or "[]")
                       if f["column"] == column and f["level"] != "warn"]
            if serious:
                hits.append(b["batch_id"])
                checks += serious
        ratio = self.median_ratio(column, [b["batch_id"] for b in window], [b["batch_id"] for b in before])
        cause = likely_cause(checks)
        if cause == "none" and ratio and match_unit_factor(ratio, 0.3, self.settings.thresholds["unit_factors"]):
            cause = "unit_change"
        changed = bool(hits) or cause != "none"

        if changed and cause in ("unit_change", "schema_change") and msg.incident_id:
            self.incident_id = msg.incident_id
            batch_ids = hits or [b["batch_id"] for b in window]
            if actions.claim_incident(self, column, batch_ids, cause):
                usable = [b for b in batch_ids if self.tables.batch(b)["status"] != "quarantined"]
                if usable:                        # quarantined batches were announced already
                    actions.do_not_use(self, usable, f"{cause} on {column}")
        self.reply(msg, level=1, confidence=0.9 if hits else 0.6,
                          summary=(f"Yes: {cause} on {column} in {', '.join(hits[:3])}" if changed
                                   else f"No change found on {column}")[:160],
                          evidence={"median_ratio_vs_before": ratio, "window_batches": len(window),
                                    "batches_before": len(before)},
                          payload={"changed": changed, "batch_ids": hits, "likely_cause": cause})

    def median_ratio(self, column: str, window: list[str], before: list[str]) -> Optional[float]:
        stats = self.tables.batch_stats()
        stats = stats[stats.column_name == column]
        now, old = stats[stats.batch_id.isin(window)]["median"], stats[stats.batch_id.isin(before)]["median"]
        if now.empty or old.empty or not old.median():
            return None
        return round(float(now.median() / old.median()), 4)

    def answer_clean_window(self, msg: Message) -> None:
        start, end = msg.payload.get("start") or "", msg.payload.get("end") or "￿"
        min_rows = int(msg.payload.get("min_rows", 0))
        clean = [b for b in self.tables.batches() if b["status"] in ("passed", "repaired")
                 and start <= (b["created_at"] or "") <= end]
        rows = sum(b["rows"] or 0 for b in clean)
        self.reply(msg, level=1,
                          summary=f"{len(clean)} clean batches, {rows} rows",
                          payload={"decision": "done" if rows >= min_rows else "failed",
                                   "detail": "" if rows >= min_rows else f"only {rows} rows < {min_rows}",
                                   "batch_ids": [b["batch_id"] for b in clean], "rows": rows})

    def answer_expected_change(self, msg: Message) -> None:
        """Migration plan from the Reviewer: new schema version for the changed feature,
        checked in a dry run first, then answer done (scenario B)."""
        tasks = msg.payload.get("tasks", {})
        if self.name not in tasks:
            return
        s = self.settings
        profile, bounds = s.profile, dict(s.bounds)
        for change in msg.payload.get("changes", []):
            profile, bounds = scale_column(profile, bounds, column_of(change["name"]),
                                           float(change["factor"]))
        ok, detail = self.check_new_schema(profile, bounds, msg.payload.get("changes", []))
        if ok:
            s.profile, s.bounds, s.schema_version = profile, bounds, s.schema_version + 1
            self.inspector = self._build_inspector()
            s.save_profile()
        self.reply(msg, level=2,
                          summary=(f"Schema v{s.schema_version} ready" if ok
                                   else f"Schema update failed: {detail}")[:160],
                          evidence={"schema_version": s.schema_version, "detail": detail},
                          payload={"decision": "done" if ok else "failed", "detail": detail,
                                   "task": tasks[self.name], "done_at": clock.iso()})

    def check_new_schema(self, profile, bounds, changes) -> tuple[bool, str]:
        """Dry run of update_schema: reference rows converted to the new unit must pass."""
        sample = reference_frame()[FEATURE_COLUMNS].copy()     # the reference, in the new unit
        for change in changes:
            c = column_of(change["name"])
            sample[c] = sample[c] * float(change["factor"])
        trial = Inspector(profile, self.settings.inspector_cfg, make_schema(bounds))
        blocking = [f for f in trial.inspect(sample) if f.level >= ALERT and f.check not in DRIFT_CHECKS]
        return (not blocking, "; ".join(f"{f.check}:{f.column}" for f in blocking))

    def post_drift_trend(self, r: BatchReport) -> None:
        stats = self.tables.batch_stats()
        for column, value in r.psi.items():
            history = stats[stats.column_name == column]["median"].tail(10).to_numpy()
            slope = float(np.polyfit(range(len(history)), history, 1)[0]) if len(history) > 2 else 0.0
            self.post(type=MessageType.FINDING,
                      kind="data.drift_trend", to=["monitor", "trainer"], topic=node(column), level=1,
                      summary=f"Drift on {column}: PSI {value}", evidence={"psi": value},
                      payload={"feature": node(column).split(":")[1], "statistic": "median",
                               "slope_per_batch": round(slope, 5), "batches_to_limit": None})


def scale_column(profile: dict, bounds: dict, column: str, factor: float):
    """New reference after an approved unit change: the same distribution, times factor."""
    import copy
    profile, bounds = copy.deepcopy(profile), dict(bounds)
    ref = profile["columns"][column]
    for key in ("lo", "hi", "median", "p25", "p75"):
        ref[key] = ref[key] * factor
    ref["edges"] = [e * factor for e in ref["edges"]]
    ref["sample"] = [v * factor for v in ref.get("sample", [])]
    if column in bounds:
        lo, hi = bounds[column]
        bounds[column] = (lo * factor, hi * factor)
    return profile, bounds

