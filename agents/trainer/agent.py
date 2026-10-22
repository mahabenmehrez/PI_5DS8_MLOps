"""Trainer agent (roadmap chapter 10, owner P4).

Retrains only when it is worth it, and forwards a new model only if it is
better overall and not worse for any group of wines.

One step follows the roadmap 5.8 loop:
  1. Look        read the inbox (request.retrain, request.do_not_use,
                 request.expected_change) and the batches table
  2. Understand  triggers, clean window, drift, live MAE, expected gain (analysis.py)
  3. Decide      llm.decide() between refuse and retrain_in_dry_run, with rule_decision() as fallback
  4. Try safely  train_compare dry run: fit on the window, compare with the live model per group
  5. Do          second decision on the comparison: send_to_deployer or reject_candidate
  6. Check       the gate is re-applied in code: a candidate that fails it is never forwarded
  7. Remember    every decision goes to trainer_runs (memory.py, owner P5, replaces this later)

    python -m agents.trainer        # long-running, on board.db (see INTEGRATION.md)
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

try:  # the real loop (owner P3) once merged
    from shared.base_agent import BaseAgent
except ImportError:  # stand-in until then
    from shared.standins import BaseAgent

from shared import llm
from shared.dry_run import dry_run

from agents.trainer import analysis
from agents.trainer.data import TrainerStore, now_iso

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "wine-quality"

MENU = [
    {"name": "refuse", "level": 1, "params_schema": ["expected_gain", "threshold"],
     "success_check": "training.decision posted with expected gain and threshold"},
    {"name": "request_clean_window", "level": 1, "params_schema": ["start", "end", "min_rows"],
     "success_check": "Data Checker answers with batch ids"},
    {"name": "retrain_in_dry_run", "level": 2, "params_schema": ["batch_ids"],
     "success_check": "run finished, metrics and comparison logged"},
    {"name": "register_candidate", "level": 1, "params_schema": ["run_id"],
     "success_check": "new version with alias candidate and all reference files"},
    {"name": "send_to_deployer", "level": 1, "params_schema": ["run_id"],
     "success_check": "Deployer answers accepted"},
    {"name": "reject_candidate", "level": 1, "params_schema": ["run_id"],
     "success_check": "model.candidate_rejected posted with failing groups"},
]
MENU_BY_NAME = {m["name"]: m for m in MENU}
RETRAIN_MENU = [MENU_BY_NAME["refuse"], MENU_BY_NAME["retrain_in_dry_run"]]
FORWARD_MENU = [MENU_BY_NAME["send_to_deployer"], MENU_BY_NAME["reject_candidate"]]
SUBSCRIPTIONS = ["dataset:*", "model:*", "expected_change:*"]

DEFAULTS = {
    "min_gain_pct": 2.0,
    "max_group_worse_pct": 10.0,   # roadmap starts at 5; see analysis.gate for why 10
    "retrain_every_batches": 10,
    "eval_batches": 5,           # 500 rows: every alcohol band has >= 50 wines
    "window_batches": None,      # None = all clean batches
    "drift_batches": 5,          # batches used for the drift check
    "rise_batches": 15,          # batches used to measure how much worse the live model got
    "min_train_rows": 300,
    "resamples": 500,
    "gain_weights": dict(analysis.DEFAULT_GAIN_WEIGHTS),
}


def load_settings(overrides: dict | None = None) -> dict:
    """agents.yaml (trainer thresholds, mlflow, db_url) + overrides."""
    cfg = {**DEFAULTS, "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
           "db_path": "board.db", "run_id": os.environ.get("RUN_ID", "dev")}
    path = Path(os.environ.get("AGENTS_YAML", ROOT / "agents.yaml"))
    if path.exists():
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg.update(((data.get("agents") or {}).get("trainer") or {}).get("thresholds") or {})
        cfg["tracking_uri"] = (data.get("mlflow") or {}).get("tracking_uri", cfg["tracking_uri"])
        db_url = data.get("db_url") or ""
        if db_url.startswith("sqlite:///"):
            cfg["db_path"] = db_url[len("sqlite:///"):]
        cfg["run_id"] = data.get("run_id", cfg["run_id"])
    cfg.update(overrides or {})
    return cfg


@dataclass
class Pending:
    """A retrain the Trainer owes someone, and the candidate it produced."""

    incident_id: str
    reason: str
    requests: list[dict] = field(default_factory=list)  # messages to answer when done
    candidate: dict | None = None                        # train_compare outputs
    window: dict | None = None
    expected_gain: float | None = None


class TrainerAgent(BaseAgent):
    name = "trainer"
    menu = MENU

    def __init__(self, board, *, store: TrainerStore | None = None, settings: dict | None = None,
                 llm_client=None, llm_config: dict | None = None, now=None):
        super().__init__(board)
        self.cfg = load_settings(settings)
        self.store = store or TrainerStore(self.cfg["db_path"])
        self.llm_client = llm_client
        self.llm_config = llm_config
        self._now = now  # callable for tests; shared.clock otherwise
        self.retrain_requests: list[dict] = []
        self.do_not_use: set[str] = set()
        self.expected_change: dict | None = None  # {"change_id", "after_batch", "message"}
        self.pending: Pending | None = None
        self.log: list[dict] = []  # decisions taken, for tests and the README

    # ---- helpers --------------------------------------------------------
    def now(self) -> str:
        return self._now() if self._now else now_iso()

    def post(self, **fields) -> str:
        fields.setdefault("run_id", self.cfg["run_id"])
        return self.board.post({"sender": self.name, **fields})

    def _mlflow(self):
        import mlflow

        mlflow.set_tracking_uri(self.cfg["tracking_uri"])
        return mlflow

    def champion(self) -> dict:
        """Live model version, its reference files and its reference MAE."""
        mlflow = self._mlflow()
        mv = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
        ref_dir = Path(mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="reference"))
        read = lambda name: json.loads((ref_dir / name).read_text())  # noqa: E731
        return {
            "version": str(mv.version),
            "uri": f"models:/{MODEL_NAME}@champion",
            "psi_bins": read("psi_bins.json"),
            "importance": read("feature_importance.json"),
            "mae_ref": read("group_metrics.json")["overall"]["mae"],
        }

    # ---- step 1: look ---------------------------------------------------
    def handle(self, msg: dict) -> None:
        if msg["type"] == "answer":
            return
        kind = msg["kind"]
        if kind == "request.retrain":
            self.retrain_requests.append(msg)
        elif kind == "request.do_not_use":
            self.do_not_use.update(msg["payload"].get("batch_ids", []))
            self.board.answer(msg, sender=self.name, level=1, summary="Batches excluded from training windows",
                              payload={"decision": "done", "detail": f"excluded {msg['payload'].get('batch_ids')}"})
        elif kind == "request.expected_change":
            batches = self.store.batches(self.now())
            after = batches["batch_id"].iloc[-1] if len(batches) else None
            self.expected_change = {"change_id": msg["payload"].get("change_id"), "after_batch": after, "message": msg}
            self.board.answer(msg, sender=self.name, level=1,
                              summary="Will retrain on data after the change only",
                              payload={"decision": "accepted", "detail": f"window starts after {after}",
                                       "task": "retrain_on_post_change_data", "done_at": None})
        elif kind == "test.ping":
            self.board.answer(msg, sender=self.name, level=0, summary="trainer: pong", payload={"ok": True})
        else:
            print(f"[trainer] ignoring {kind} {msg['id']}")

    def observe(self) -> dict:
        return {"now": self.now(), "batches": self.store.batches(self.now()), "last": self.store.last_training()}

    # ---- step 2: understand --------------------------------------------
    def analyze(self, obs: dict) -> dict:
        batches = obs["batches"]
        clean = batches[batches["status"].isin(["passed", "repaired"]) & ~batches["batch_id"].isin(self.do_not_use)]
        last_batch = (obs["last"] or {}).get("last_batch")
        new_clean = clean[clean["batch_id"] > last_batch] if last_batch else clean
        due = len(new_clean) >= self.cfg["retrain_every_batches"]
        requested = bool(self.retrain_requests)
        after = self.expected_change["after_batch"] if self.expected_change else None
        if not (requested or due):
            return {"flag": False}
        retry = '"dry_run_failed"' in ((obs["last"] or {}).get("detail") or "")  # a failed run may work now: try again
        if requested and obs["last"] and last_batch and new_clean.empty and not self.expected_change and not retry:
            return {"flag": True, "already_done": obs["last"]}

        window = analysis.choose_window(batches, size=self.cfg["window_batches"], eval_batches=self.cfg["eval_batches"],
                                        after=after, do_not_use=self.do_not_use)
        if not window.eval_ids:  # nothing clean and labelled yet: no need to load the live model
            return {"flag": True, "nodes": [f"model:{MODEL_NAME}"], "trigger": "request" if requested else "schedule",
                    "window": window, "no_data": True, "numbers": {
                        "champion_version": None, "mae_ref": None, "mae_now": None, "mae_rise_pct": 0.0,
                        "mae_rise_pct_low95": 0.0, "weighted_psi": 0.0, "top_psi": {}, "importance_of_top_psi": {},
                        "new_rows": 0, "train_rows": 0, "expected_gain": 0.0, "threshold": self.cfg["min_gain_pct"],
                        "train_batches": 0, "eval_batches": 0, "excluded": window.excluded,
                        "expected_change": self.expected_change["change_id"] if self.expected_change else None}}
        champ = self.champion()
        recent_ids = (window.train_ids + window.eval_ids)[-self.cfg["drift_batches"]:]
        recent = self.store.rows(recent_ids, obs["now"])
        drift = analysis.drift_table(recent, champ["psi_bins"], champ["importance"]) if len(recent) else {"psi": {}, "weighted_psi": 0.0}
        # live-model error is measured on the newest rise_batches clean batches (eval batches included):
        # 500 rows leave a +-9% bootstrap interval on the MAE rise, too wide to see scenario D
        rise_ids = (window.train_ids + window.eval_ids)[-self.cfg["rise_batches"]:]
        eval_rows = self.store.rows(rise_ids, obs["now"])
        rise = {"mae_now": None, "rise_pct": 0.0, "rise_pct_low": 0.0}
        if len(eval_rows):
            from baseline import train

            X, y = train.prepare_features_train(eval_rows)
            errors = (self._mlflow().pyfunc.load_model(champ["uri"]).predict(X) - y).abs().to_numpy()
            rise = analysis.mae_rise(errors, champ["mae_ref"])
        mae_now, mae_rise_pct = rise["mae_now"], rise["rise_pct"]
        train_rows = len(self.store.rows(window.train_ids, obs["now"])) if window.train_ids else 0
        labelled_new = new_clean[new_clean["labelled"]]["batch_id"].tolist()
        new_rows = len(self.store.rows(labelled_new, obs["now"])) if labelled_new else 0
        gain = analysis.estimate_gain(drift["weighted_psi"], rise["rise_pct_low"], new_rows, self.cfg["gain_weights"])
        top_psi = dict(sorted(drift["psi"].items(), key=lambda kv: -kv[1])[:3])
        return {
            "flag": True,
            "nodes": [f"model:{MODEL_NAME}"],
            "trigger": "request" if requested else "schedule",
            "window": window,
            "numbers": {
                "champion_version": champ["version"],
                "mae_ref": round(champ["mae_ref"], 4),
                "mae_now": round(mae_now, 4) if mae_now else None,
                "mae_rise_pct": round(mae_rise_pct, 2),
                "mae_rise_pct_low95": round(rise["rise_pct_low"], 2),
                "weighted_psi": round(drift["weighted_psi"], 4),
                "top_psi": {k: round(v, 3) for k, v in top_psi.items()},
                "importance_of_top_psi": {k: round(champ["importance"].get(k, 0.0), 4) for k in top_psi},
                "new_rows": new_rows,
                "train_rows": train_rows,
                "expected_gain": round(gain, 2),
                "threshold": self.cfg["min_gain_pct"],
                "train_batches": len(window.train_ids),
                "eval_batches": len(window.eval_ids),
                "excluded": window.excluded,
                "expected_change": self.expected_change["change_id"] if self.expected_change else None,
            },
        }

    # ---- step 3: decide -------------------------------------------------
    def context(self, analysis_: dict, inbox: list) -> dict:
        past = []
        last = self.store.last_training()
        if last:
            past.append({k: last[k] for k in ("decision", "version", "expected_gain", "overall_gain_pct")})
        return {"numbers": analysis_["numbers"], "trigger": analysis_["trigger"],
                "requests": [m["summary"] for m in self.retrain_requests], "past_retrains": past}

    def rule_decision(self, analysis_: dict) -> dict:
        n = analysis_["numbers"]
        if analysis_.get("no_data"):
            return {"action": "refuse", "level": 1, "params": {"expected_gain": 0.0, "threshold": n["threshold"]},
                    "reason": "no clean labelled batches to train or evaluate on yet", "confidence": 0.95}
        if n["expected_gain"] < n["threshold"]:
            return {"action": "refuse", "level": 1,
                    "params": {"expected_gain": n["expected_gain"], "threshold": n["threshold"]},
                    "reason": f"retraining would lower the error (MAE) by only {n['expected_gain']}%, less than the {n['threshold']}% needed",
                    "confidence": 0.8}
        if n["train_rows"] < self.cfg["min_train_rows"]:
            return {"action": "refuse", "level": 1,
                    "params": {"expected_gain": n["expected_gain"], "threshold": n["threshold"]},
                    "reason": f"only {n['train_rows']} clean labelled rows to train on", "confidence": 0.8}
        return {"action": "retrain_in_dry_run", "level": 2,
                "params": {"batch_ids": analysis_["window"].train_ids},
                "reason": f"live error (MAE) went up {n['mae_rise_pct']}%; retraining should lower it by {n['expected_gain']}% (need {n['threshold']}%)",
                "confidence": 0.7}

    def decide(self, ctx: dict, menu: list, fallback: dict, incident_id: str) -> llm.Decision:
        return llm.decide(self, ctx, menu, fallback, incident_id=incident_id,
                          client=self.llm_client, config=self.llm_config)

    # ---- the step -------------------------------------------------------
    def step(self) -> None:
        for msg in self.board.read(self.name):
            self.safe_handle(msg)
        self.board.heartbeat(self.name)
        if self.pending and self.pending.candidate:
            return self.finish_candidate()
        try:
            obs = self.observe()
            for msg in self.board.read(self.name):  # a request posted with the batches just seen (same transaction)
                self.safe_handle(msg)
            result = self.analyze(obs)
        except Exception as exc:  # MLflow or board.db unreachable: say so, try again next step
            if self.retrain_requests:
                self._answer_requests("failed", f"could not analyse: {type(exc).__name__}: {exc}"[:200])
            self.log.append({"phase": "analyze", "error": repr(exc)})
            return None
        if not result["flag"]:
            return None
        if result.get("already_done"):  # asked again, but nothing new since the last training
            last = result["already_done"]
            done = last["decision"] == "forwarded"
            detail = (f"candidate v{last['version']} was already sent to the Deployer" if done else
                      f"already decided '{last['decision']}' on these batches; no new clean batches since {last['last_batch']}")
            self.log.append({"phase": "retrain", "action": "already_done", "source": "rules", "reason": detail})
            return self._answer_requests("done" if done else "refused", detail,
                                         {"version": last.get("version")} if done else None)
        incident_id = next((m["incident_id"] for m in self.retrain_requests if m.get("incident_id")), None) \
            or "inc-" + uuid.uuid4().hex[:8]
        rule = self.rule_decision(result)
        if result.get("no_data"):  # nothing to weigh up: no LLM call
            decision = llm.Decision(**rule, source="rules")
        else:
            decision = self.decide(self.context(result, []), RETRAIN_MENU, rule, incident_id)
        self.log.append({"phase": "retrain", **decision.model_dump()})
        if decision.action == "retrain_in_dry_run":
            return self.retrain(result, decision, incident_id)
        return self.refuse(result, decision, incident_id,
                           reason_code="not_enough_data" if result.get("no_data") else "gain_below_threshold")

    # ---- step 4-5: act --------------------------------------------------
    def _answer_requests(self, decision: str, detail: str, extra: dict | None = None) -> None:
        for req in self.retrain_requests:
            self.board.answer(req, sender=self.name, level=1, summary=f"Retrain request {decision}: {detail}"[:160],
                              payload={"decision": decision, "detail": detail, **(extra or {})})
        self.retrain_requests = []

    def refuse(self, result: dict, decision: llm.Decision, incident_id: str, reason_code: str = "gain_below_threshold") -> None:
        n = result["numbers"]
        window = result["window"].as_payload()
        self.post(type="finding", kind="training.decision", to=["monitor"], topic=f"model:{MODEL_NAME}",
                  incident_id=incident_id, level=1, confidence=decision.confidence,
                  summary=f"Retrain refused: {decision.reason}"[:160],
                  evidence={k: n[k] for k in ("weighted_psi", "mae_rise_pct", "mae_rise_pct_low95", "new_rows", "top_psi", "importance_of_top_psi")},
                  payload={"decision": "refuse", "expected_gain": n["expected_gain"], "threshold": n["threshold"],
                           "data_window": window, "reason_code": reason_code, "decided_by": decision.source})
        self._answer_requests("refused", decision.reason)
        last_batch = window["batch_ids"][-1] if window["batch_ids"] else None
        self.store.record(incident_id=incident_id, decision="refuse", expected_gain=n["expected_gain"],
                          train_ids=window["batch_ids"], eval_ids=window["eval_batch_ids"],
                          last_batch=(window["eval_batch_ids"] or [last_batch])[-1],
                          detail={"reason": decision.reason, "reason_code": reason_code})

    def retrain(self, result: dict, decision: llm.Decision, incident_id: str) -> None:
        n, window = result["numbers"], result["window"]
        train_ids = [b for b in decision.params.get("batch_ids", window.train_ids) if b in window.train_ids] or window.train_ids
        res = dry_run("train_compare", {
            "db_path": str(Path(self.store.db_path).resolve()), "train_ids": train_ids, "eval_ids": window.eval_ids,
            "now": self.now(), "tracking_uri": self.cfg["tracking_uri"],
            "live_model_uri": f"models:/{MODEL_NAME}@champion", "min_rows": self.cfg["min_train_rows"],
            "min_gain_pct": self.cfg["min_gain_pct"], "max_group_worse_pct": self.cfg["max_group_worse_pct"],
            "resamples": self.cfg["resamples"],
        }, timeout_s=300)
        if not res.ok:  # dry run failed: fall back to the safest action
            lines = [ln.strip() for ln in res.log_tail.splitlines() if ln.strip() and not ln.startswith("exit code")]
            safe = llm.Decision(action="refuse", level=1, params={}, source="rules",
                                reason=f"training dry run failed: {(lines or ['no output'])[-1][-150:]}")
            return self.refuse(result, safe, incident_id, reason_code="dry_run_failed")
        self.post(type="finding", kind="training.decision", to=["monitor"], topic=f"model:{MODEL_NAME}",
                  incident_id=incident_id, level=2, confidence=decision.confidence,
                  summary=f"Retrained on {len(train_ids)} clean batches: {decision.reason}"[:160],
                  evidence={k: n[k] for k in ("weighted_psi", "mae_rise_pct", "mae_rise_pct_low95", "new_rows")},
                  payload={"decision": "retrain", "expected_gain": n["expected_gain"], "threshold": n["threshold"],
                           "data_window": {**window.as_payload(), "batch_ids": train_ids},
                           "reason_code": "requested" if result["trigger"] == "request" else "schedule",
                           "decided_by": decision.source, "candidate_run_id": res.outputs["run_id"]})
        self.pending = Pending(incident_id=incident_id, reason=decision.reason, requests=self.retrain_requests,
                               candidate=res.outputs, window={"train_ids": train_ids, "eval_ids": window.eval_ids},
                               expected_gain=n["expected_gain"])
        self.retrain_requests = []
        self.finish_candidate()

    def finish_candidate(self) -> None:
        p = self.pending
        cand, verdict = p.candidate, p.candidate["gate"]
        report = cand["report"]
        ctx = {"gate": verdict, "overall": report["overall"], "expected_gain": p.expected_gain,
               "min_gain_pct": self.cfg["min_gain_pct"], "max_group_worse_pct": self.cfg["max_group_worse_pct"]}
        rule = ({"action": "send_to_deployer", "level": 1, "params": {"run_id": cand["run_id"]},
                 "reason": f"error (MAE) {verdict['overall_gain_pct']:.1f}% lower, no alcohol or acidity group has a higher error", "confidence": 0.85}
                if verdict["passed"] else
                {"action": "reject_candidate", "level": 1, "params": {"run_id": cand["run_id"]},
                 "reason": self._reject_reason(verdict), "confidence": 0.9})
        decision = self.decide(ctx, FORWARD_MENU, rule, p.incident_id)
        if decision.action == "send_to_deployer" and not verdict["passed"]:
            # 6. Check: the gate is code, not LLM judgement
            decision = llm.Decision(**{**rule, "source": "rules"})
        self.log.append({"phase": "forward", **decision.model_dump()})
        self.retrain_requests = p.requests
        if decision.action == "send_to_deployer":
            self.send_to_deployer(decision)
        else:
            self.reject(decision)
        self.pending = None

    def _reject_reason(self, verdict: dict) -> str:
        g = verdict["overall_gain_pct"]
        overall = f"error (MAE) {g:.1f}% lower overall" if g >= 0 else f"error (MAE) {-g:.1f}% higher overall"
        if verdict["failing_groups"]:
            groups = ", ".join(f"{f['group']} {f['band']}" for f in verdict["failing_groups"])
            return f"{overall} and higher for {groups}" if g < 0 else f"{overall} but higher for {groups}"
        if g < 0:
            return overall
        return f"{overall}, less than the {self.cfg['min_gain_pct']}% needed"

    def send_to_deployer(self, decision: llm.Decision) -> None:
        from baseline import train

        p, cand = self.pending, self.pending.candidate
        report = cand["report"]
        champ_version = self.champion()["version"]
        version = train.register_run(cand["run_id"], alias="candidate", tracking_uri=self.cfg["tracking_uri"],
                                     tags={"incident_id": p.incident_id, "train_batches": ",".join(p.window["train_ids"])})
        by_group = {g: {b: round(s["delta_pct"], 2) for b, s in bands.items() if s.get("counts")}
                    for g, bands in report["groups"].items()}
        self.post(type="request", kind="request.deploy_candidate", to=["deployer"], topic=f"model:{MODEL_NAME}@{version}",
                  incident_id=p.incident_id, level=2, confidence=decision.confidence,
                  summary=(f"Candidate v{version} passes the gate: error (MAE) {report['overall']['mae_new']:.3f} vs "
                           f"{report['overall']['mae_live']:.3f} live, {-report['overall']['delta_pct']:.1f}% lower")[:160],
                  evidence={"eval_rows": report["overall"]["n"], "bootstrap_resamples": report["resamples"]},
                  expires_at=None,
                  payload={"model_uri": f"models:/{MODEL_NAME}@candidate", "version": version,
                           "champion_version": champ_version,
                           "metrics": {"overall": {"mae_candidate": round(report["overall"]["mae_new"], 4),
                                                   "mae_champion": round(report["overall"]["mae_live"], 4),
                                                   "delta_pct": round(report["overall"]["delta_pct"], 2)},
                                       "by_group": by_group},
                           "comparison_ref": cand["comparison_ref"]})
        self._answer_requests("done", f"candidate v{version} sent to the Deployer")
        self._finish_expected_change(version)
        self.store.record(run_id=cand["run_id"], incident_id=p.incident_id, decision="forwarded", version=version,
                          train_ids=p.window["train_ids"], eval_ids=p.window["eval_ids"],
                          last_batch=p.window["eval_ids"][-1], expected_gain=p.expected_gain,
                          overall_gain_pct=cand["gate"]["overall_gain_pct"], detail={"reason": decision.reason})

    def reject(self, decision: llm.Decision) -> None:
        p, cand = self.pending, self.pending.candidate
        verdict = cand["gate"]
        self.post(type="finding", kind="model.candidate_rejected", to=["monitor", "human"],
                  topic=f"model:{MODEL_NAME}", incident_id=p.incident_id, level=1, confidence=decision.confidence,
                  summary=f"Candidate rejected: {decision.reason}"[:160],
                  evidence={"mae_live": cand["report"]["overall"]["mae_live"], "mae_candidate": cand["report"]["overall"]["mae_new"],
                            "bootstrap_resamples": cand["report"]["resamples"]},
                  payload={"version": f"run:{cand['run_id']}", "overall_delta": round(-verdict["overall_gain_pct"] / 100, 4),
                           "failing_groups": verdict["failing_groups"], "comparison_ref": cand["comparison_ref"]})
        self._answer_requests("failed", decision.reason)
        self.store.record(run_id=cand["run_id"], incident_id=p.incident_id, decision="rejected",
                          train_ids=p.window["train_ids"], eval_ids=p.window["eval_ids"],
                          last_batch=p.window["eval_ids"][-1], expected_gain=p.expected_gain,
                          overall_gain_pct=verdict["overall_gain_pct"], detail={"reason": decision.reason})

    def _finish_expected_change(self, version: str) -> None:
        if not self.expected_change:
            return
        msg = self.expected_change["message"]
        self.board.answer(msg, sender=self.name, level=1, summary=f"Retrained on post-change data; candidate v{version} sent",
                          payload={"decision": "done", "detail": f"candidate v{version}",
                                   "task": "retrain_on_post_change_data", "done_at": self.now()})
        self.expected_change = None
