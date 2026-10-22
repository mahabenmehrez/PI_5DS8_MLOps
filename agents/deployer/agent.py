"""
Deployer agent (P5). Decides how much live traffic a candidate model gets.
Loop: observe -> analyze -> decide -> (dry run) -> act -> verify -> remember.
The emergency rollback is plain code (no LLM, no approval).
"""
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from shared.messages import Message, MessageType
from agents.deployer import analysis as rules
from agents.deployer import state

try:  # boucle commune de P3
    from shared.base_agent import BaseAgent, Action
except ImportError:  # boucle provisoire
    @dataclass
    class Action:
        name: str
        level: int
        params_schema: dict = field(default_factory=dict)
        success_check: str = ""

    class BaseAgent:
        name: str = "base"
        menu: List[Action] = []

        def __init__(self):
            self.handled_ids = set()
            self.outbox: List[Message] = []

        def post(self, msg: Message) -> Message:
            self.outbox.append(msg)
            return msg

        def safe_handle(self, msg: Message) -> None:
            if msg.id in self.handled_ids:  # handlers idempotents
                return
            self.handled_ids.add(msg.id)
            try:
                self.handle(msg)
            except Exception as exc:
                print(f"[{self.name}] handler error: {exc}")


@dataclass
class Analysis:
    flag: bool = False
    nodes: List[str] = field(default_factory=list)
    numbers: Dict[str, Any] = field(default_factory=dict)


class DeployerAgent(BaseAgent):
    name = "deployer"

    menu = [
        Action("hold", 1, {"reason": "str"}, "share unchanged; reason posted"),
        Action("rollback_emergency", 1, {"reason": "str"},
               "candidate share 0; error rate back to champion level"),
        Action("rollback", 2, {"reason": "str"}, "same, decided by the LLM",
               dry_run_kind="traffic_replay"),
        Action("advance_stage", 2, {"to_stage": "str"},
               "dry-run replay passed; no hard-limit breach for 10 min",
               dry_run_kind="traffic_replay"),
        Action("promote_100", 3, {"reason": "str"},
               "human approved; alias champion moved"),
    ]

    POST_PROMOTION_MINUTES = 60     # surveillance du nouveau champion apres le 100 %
    RECOVERY_MINUTES = 2            # delai pour verifier le retour a la normale
    REPLAY_RETRY_MINUTES = 10       # delai avant de relancer un traffic_replay echoue

    def __init__(self, cfg: Optional[dict] = None, llm=None, replay=None):
        super().__init__()
        self.outbox: List[Message] = []   # messages postes (tests + compat BaseAgent)
        self.cfg = cfg or {}
        self.llm = llm            # fonction(context, menu) -> dict, ou None
        self.replay = replay      # fonction(release_state, analysis) -> dict(ok, ...), ou None
        self.replay_result: Optional[dict] = None
        self._replay_retry_at: Optional[datetime] = None
        self.post_promotion: Optional[dict] = None   # surveillance apres promotion
        self.recovery: Optional[dict] = None         # verification apres un rollback
        self.recovered: List[dict] = []
        self.holds: Dict[str, str] = {}
        self.history: List[dict] = []
        self.pending: Optional[dict] = None   # approbation humaine en attente
        self.previous_champion: Optional[str] = None
        self.last_hold: Optional[str] = None
        self._breach: Optional[dict] = None
        self.last_llm: Optional[dict] = None   # derniere decision validee du LLM
        self._llm_key = None                   # cache : meme situation = meme decision
        self._llm_choice: Optional[dict] = None
        self._llm_at: Optional[datetime] = None

    # ---- utilitaires ------------------------------------------------
    def _minutes_in_stage(self, st: dict) -> float:
        try:
            start = datetime.fromisoformat(st["updated_at"])
            now = datetime.fromisoformat(state._now())
            return max(0.0, (now - start).total_seconds() / 60)
        except Exception:
            return 0.0

    @classmethod
    def from_yaml(cls, path: str = "agents.yaml", llm=None, replay=None):
        """Cree l'agent avec les seuils de agents.yaml (section agents.deployer)."""
        from agents.deployer.config import load_deployer_config
        return cls(cfg=load_deployer_config(path), llm=llm, replay=replay)

    def _now_dt(self) -> datetime:
        return datetime.fromisoformat(state._now())

    def _minutes_since(self, iso: str) -> float:
        try:
            return (self._now_dt() - datetime.fromisoformat(iso)).total_seconds() / 60
        except Exception:
            return 0.0

    def dry_run_ok(self, decision: dict, analysis: Optional[Analysis] = None) -> bool:
        """Dry run traffic_replay a 3x la vitesse, exige avant le passage a 50 % (roadmap 11.2)."""
        if decision["action"] != "advance_stage" or decision["params"].get("to_stage") != "50%":
            return True
        if self._replay_retry_at and self._now_dt() < self._replay_retry_at:
            self.last_hold = "traffic_replay 3x failed recently: retry later"
            return False
        if self.replay is None:
            self.replay_result = {"ok": True, "skipped": True,
                                  "detail": "traffic_replay not configured"}
            return True
        try:
            res = self.replay(state.get_release_state(), analysis)
        except Exception as exc:
            res = {"ok": False, "detail": f"replay error: {exc}"}
        self.replay_result = res
        if not res.get("ok"):
            self.last_hold = f"traffic_replay 3x failed: {res.get('detail', '')}"
            self._replay_retry_at = self._now_dt() + timedelta(minutes=self.REPLAY_RETRY_MINUTES)
        return bool(res.get("ok"))

    def _move_alias(self, version: str) -> None:
        """Deplace l'alias champion dans MLflow (ignore si MLflow n'est pas configure)."""
        if not os.getenv("MLFLOW_TRACKING_URI"):
            return
        try:
            from mlflow.tracking import MlflowClient
            MlflowClient().set_registered_model_alias("wine-quality", "champion", str(version))
        except Exception as exc:
            print(f"[deployer] alias not moved: {exc}")

    def post(self, msg: Message) -> Message:
        """Compat tests + board: accepte un Message (style Deployer d'origine)."""
        self.outbox.append(msg)
        try:
            data = msg.model_dump(mode="json") if hasattr(msg, "model_dump") else msg
            self.board.post(data)
        except Exception:
            pass
        return msg

    def _post_stage(self, from_stage: str, to_stage: str, share: float, version,
                    evidence: Optional[dict] = None) -> Message:
        self.history.append({"time": state._now(), "from_stage": from_stage,
                             "to_stage": to_stage, "model_version": str(version)})
        return self.post(Message(
            type=MessageType.FINDING, kind="deployment.stage_changed", sender=self.name,
            to=["monitor", "trainer"], topic="deployment:prod", level=2,
            summary=f"Model {version}: {from_stage} -> {to_stage} ({share:.0%})",
            evidence=evidence or {},
            payload={"model_version": str(version), "from_stage": from_stage,
                     "to_stage": to_stage, "candidate_share": share}))

    def _post_rollback(self, st: dict, trigger: dict, automatic: bool, summary: str) -> Message:
        self.history.append({"time": state._now(), "from_stage": st["stage"],
                             "to_stage": "rolled_back",
                             "model_version": str(st["champion_version"])})
        return self.post(Message(
            type=MessageType.FINDING, kind="deployment.rolled_back", sender=self.name,
            to=[], topic="deployment:prod", level=1, summary=summary,
            evidence=trigger,
            payload={"from_version": str(st["candidate_version"]),
                     "to_version": str(st["champion_version"]),
                     "trigger": trigger, "automatic": automatic}))

    # ---- 1. Look ----------------------------------------------------
    def observe(self) -> dict:
        st = state.get_release_state()
        if st["stage"] in ("none", "100%"):
            return {"state": st}
        variant = "shadow" if st["stage"] == "shadow" else "candidate"
        since = st["updated_at"]
        return {"state": st, "cand_variant": variant,
                "cand_rows": state.read_predictions(variant, since),
                "champ_rows": state.read_predictions("champion", since)}

    # ---- 2. Understand ----------------------------------------------
    def analyze(self, obs: dict) -> Analysis:
        st = obs["state"]
        self._breach = None
        if st["stage"] in ("none", "100%"):
            return Analysis(flag=False)
        cand = rules.summarize(obs["cand_rows"])
        champ = rules.summarize(obs["champ_rows"])
        cand_abs = state.abs_errors(obs["cand_variant"], st["updated_at"])
        champ_abs = state.abs_errors("champion", st["updated_at"])
        minutes = self._minutes_in_stage(st)
        per_minute: Dict[str, list] = {}
        for r in obs["cand_rows"]:
            per_minute.setdefault(r["time"][:16], []).append(r)
        windows = [rules.summarize(per_minute[k]) for k in sorted(per_minute)][-5:]
        if st["stage"] in ("10%", "50%"):
            self._breach = rules.hard_limit_breached(windows, self.cfg)
        gate = rules.evaluate_gate(st["stage"], cand, champ, cand_abs, champ_abs,
                                   minutes, self.cfg)
        return Analysis(True, ["deployment:prod"], {
            "stage": st["stage"], "candidate": cand, "champion": champ,
            "gate": gate, "breach": self._breach, "minutes_in_stage": minutes})

    # ---- 3. Decide --------------------------------------------------
    def rule_decision(self, analysis: Analysis) -> dict:
        n = analysis.numbers
        return rules.rule_decision(n["stage"], n["gate"], n["breach"],
                                   list(self.holds.values()))

    LLM_ALLOWED = ("hold", "rollback")     # le LLM ne peut jamais avancer ni promouvoir
    LLM_REUSE_MINUTES = 10                 # duree de reutilisation d'une decision identique

    def in_grey_zone(self, analysis: Analysis) -> bool:
        """Zone grise = une porte est franchement negative, mais aucune limite dure n'est depassee."""
        n = analysis.numbers
        return bool(n["gate"]["bad"]) and not n["breach"]

    def _llm_context(self, analysis: Analysis, fallback: dict) -> dict:
        """Ce que voit le LLM : des chiffres et des cas passes, jamais de lignes brutes."""
        n = analysis.numbers
        past: list = []
        try:
            from shared import memory
            query = {"error_class": "rollback", "feature": "deployment",
                     "text": "; ".join(n["gate"]["bad"])}
            past = [{"action": r["action"], "outcome": r["outcome"], "reason": r["problem_text"]}
                    for r in memory.similar(query, self.name, k=3)]
        except Exception:
            past = []
        return {
            "stage": n["stage"], "minutes_in_stage": round(n["minutes_in_stage"], 1),
            "candidate": n["candidate"], "champion": n["champion"],
            "gate_bad": n["gate"]["bad"], "gate_waiting": n["gate"]["waiting"],
            "holds": list(self.holds.values()),
            "hard_limits": {"p95_ms": rules.DEFAULTS["limit_p95_ms"],
                            "error_rate": rules.DEFAULTS["limit_error_rate"],
                            "minutes": rules.DEFAULTS["limit_minutes"]},
            "past_cases": past,
            "rule_decision": {"action": fallback["action"],
                              "reason": fallback["params"].get("reason", "")},
        }

    def _llm_cache_valid(self, key) -> bool:
        if self._llm_key != key or self._llm_choice is None or self._llm_at is None:
            return False
        try:
            age = (datetime.fromisoformat(state._now()) - self._llm_at).total_seconds() / 60
        except Exception:
            return False
        return age < self.LLM_REUSE_MINUTES

    def decide(self, analysis: Analysis) -> dict:
        fallback = self.rule_decision(analysis)
        if self.llm is None or not self.in_grey_zone(analysis):
            return fallback        # limite dure, cas clair ou attente : jamais de LLM
        n = analysis.numbers
        key = (n["stage"], tuple(sorted(b.split()[0] for b in n["gate"]["bad"])))
        if self._llm_cache_valid(key):
            return self._llm_choice
        try:
            choice = self.llm(self._llm_context(analysis, fallback), self.menu)
            spec = {a.name: a.level for a in self.menu}
            name = choice.get("action")
            if name in self.LLM_ALLOWED and choice.get("level") == spec[name]:
                choice.setdefault("params", {})
                choice["params"].setdefault("reason", choice.get("reason", ""))
                self.last_llm = choice
                self._llm_key, self._llm_choice = key, choice
                self._llm_at = datetime.fromisoformat(state._now())
                return choice
            print(f"[deployer] llm action refused by guardrail: {name!r}")
        except Exception as exc:
            print(f"[deployer] llm_invalid -> rule fallback: {exc}")
        return fallback

    # ---- 5. Do ------------------------------------------------------
    def act(self, decision: dict, analysis: Analysis):
        action = decision["action"]
        st = state.get_release_state()
        if action in ("rollback_emergency", "rollback"):
            state.set_release_state(stage="none", candidate_share=0.0,
                                    shadow_on=0, candidate_version=None)
            self._post_rollback(
                st, self._breach or {"metric": "llm", "value": 0, "limit": 0},
                action == "rollback_emergency",
                f"Rolled back {st['candidate_version']} -> {st['champion_version']}")
            baseline = analysis.numbers["champion"]["error_rate"] if analysis else 0.0
            self._start_recovery(action, baseline)
        elif action == "advance_stage":
            to = decision["params"]["to_stage"]
            if not self.dry_run_ok(decision, analysis):
                return None
            state.set_release_state(stage=to, candidate_share=rules.SHARES[to],
                                    shadow_on=1 if to == "shadow" else 0)
            evidence = {"traffic_replay": self.replay_result} if (
                to == "50%" and self.replay_result) else None
            self._post_stage(st["stage"], to, rules.SHARES[to], st["candidate_version"],
                             evidence)
        elif action == "promote_100" and self.pending is None:
            self.pending = {"version": st["candidate_version"]}
            self.post(Message(
                type=MessageType.REQUEST, kind="request.approval", sender=self.name,
                to=["human"], topic="deployment:prod", level=3,
                summary=f"Promote model {st['candidate_version']} to 100%?",
                evidence={"candidate": analysis.numbers["candidate"],
                          "champion": analysis.numbers["champion"]},
                payload={"question": f"Promote {st['candidate_version']} to 100% traffic?",
                         "options": ["approve", "reject"],
                         "default_if_no_answer": "reject", "card_ref": "deploy-card"}))
        return decision

    # ---- 6. Check ---------------------------------------------------
    def verify(self, decision: dict) -> bool:
        st = state.get_release_state()
        action = decision["action"]
        if action in ("rollback_emergency", "rollback"):
            return st["candidate_share"] == 0.0
        if action == "advance_stage":
            return st["stage"] == decision["params"]["to_stage"]
        if action == "promote_100":
            return self.pending is not None or st["stage"] == "none"
        return True

    # ---- 7. Remember ------------------------------------------------
    def _remember(self, analysis: Analysis, decision: dict, ok: bool) -> None:
        try:
            from shared import memory
            st = state.get_release_state()
            memory.remember(
                self.name, f"deploy-{st['candidate_version']}",
                {"error_class": decision["action"], "feature": "deployment",
                 "text": decision["params"].get("reason", "")},
                {"stage": analysis.numbers["stage"]}, decision["action"],
                decision["params"], decision["level"], "success" if ok else "failure")
        except Exception as exc:
            print(f"[deployer] memory error: {exc}")

    # ---- Suivi apres rollback ou promotion ---------------------------
    def watching(self) -> bool:
        """Vrai tant que l'agent doit encore observer du trafic (recuperation ou promotion)."""
        return bool(self.recovery or self.post_promotion)

    def _start_recovery(self, action: str, baseline_error: float) -> None:
        self.recovery = {"incident_id": "inc-" + uuid.uuid4().hex[:8], "at": state._now(),
                         "baseline_error": float(baseline_error), "action": action}

    def _check_recovery(self) -> None:
        """Succes check du rollback : le taux d'erreurs est revenu au niveau du champion."""
        rec = self.recovery
        if not rec or self._minutes_since(rec["at"]) < self.RECOVERY_MINUTES:
            return
        rows = [r for r in state.read_predictions(None, rec["at"]) if r["variant"] != "shadow"]
        if not rows:
            return                                   # pas de trafic : on attend
        summary = rules.summarize(rows)
        ok = summary["error_rate"] <= rec["baseline_error"] + 0.01
        elapsed = self._minutes_since(rec["at"]) * 60
        self.recovered.append({"incident_id": rec["incident_id"], "ok": ok,
                               "error_rate": summary["error_rate"]})
        self.post(Message(
            type=MessageType.FINDING, kind="incident.closed", sender=self.name, to=[],
            topic=f"incident:{rec['incident_id']}", incident_id=rec["incident_id"], level=0,
            summary=("Rollback verified: error rate back to champion level" if ok
                     else "Rollback done but error rate still high"),
            evidence={"error_rate": summary["error_rate"], "baseline": rec["baseline_error"]},
            payload={"outcome": "recovered" if ok else "not_recovered",
                     "root_cause_node": "deployment:prod", "actions": [rec["action"]],
                     "time_to_recover_s": round(elapsed)}))
        self.recovery = None

    def _check_post_promotion(self) -> None:
        """Apres un 100 %, les limites dures protegent le nouveau champion pendant une heure :
        s'il se degrade, l'ancien champion (deja approuve) est restaure sans LLM."""
        pp = self.post_promotion
        if not pp:
            return
        st = state.get_release_state()
        if st["stage"] != "none" or str(st["champion_version"]) != pp["version"] \
                or self._minutes_since(pp["since"]) > self.POST_PROMOTION_MINUTES:
            self.post_promotion = None
            return
        rows = [r for r in state.read_predictions("champion", pp["since"])
                if str(r["model_version"]) == pp["version"]]
        per_minute: Dict[str, list] = {}
        for r in rows:
            per_minute.setdefault(r["time"][:16], []).append(r)
        windows = [rules.summarize(per_minute[k]) for k in sorted(per_minute)][-5:]
        breach = rules.hard_limit_breached(windows, self.cfg)
        if not breach:
            return
        previous = pp["previous"]
        state.set_release_state(champion_version=previous, candidate_version=None,
                                stage="none", candidate_share=0.0, shadow_on=0)
        self._move_alias(previous)
        self._post_rollback(
            {"stage": "100%", "candidate_version": pp["version"], "champion_version": previous},
            breach, True, f"New champion {pp['version']} degraded: restored {previous}")
        self._start_recovery("rollback_champion", 0.0)
        try:
            from shared import memory
            memory.remember(self.name, f"deploy-{pp['version']}",
                            {"error_class": "rollback_champion", "feature": "deployment",
                             "text": f"{breach['metric']} {breach['value']:.3f} > {breach['limit']}"},
                            {"stage": "100%"}, "rollback_champion", {"to": previous}, 1, "success")
        except Exception as exc:
            print(f"[deployer] memory error: {exc}")
        self.post_promotion = None

    # ---- Messages recus ---------------------------------------------
    def handle(self, msg: Message) -> Optional[Message]:
        if msg.kind == "request.deploy_candidate" and msg.type == MessageType.REQUEST:
            return self._on_deploy_candidate(msg)
        if msg.kind == "request.hold" and msg.type == MessageType.REQUEST:
            return self._on_hold(msg)
        if msg.kind == "incident.closed":
            self.holds.pop(msg.incident_id or "", None)
            return None
        if msg.kind == "human.decision":
            return self._on_human_decision(msg)
        if msg.kind == "query.deployment_changed" and msg.type == MessageType.QUESTION:
            return self._on_query_changed(msg)
        print(f"[deployer] ignoring unknown kind: {msg.kind}")
        return None

    def _answer(self, original: Message, payload: dict, summary: str) -> Message:
        return self.post(Message(
            type=MessageType.ANSWER, kind=original.kind, sender=self.name,
            to=[original.sender], topic=original.topic, reply_to=original.id,
            level=1, summary=summary, payload=payload))

    def _on_deploy_candidate(self, msg: Message) -> Message:
        version = str(msg.payload.get("version", "?"))
        st = state.get_release_state()
        champion = str(msg.payload.get("champion_version") or st["champion_version"])
        self._llm_key, self._llm_choice, self.last_llm = None, None, None
        state.set_release_state(champion_version=champion, candidate_version=version,
                                stage="shadow", candidate_share=0.0, shadow_on=1)
        answer = self._answer(msg, {"decision": "accepted", "detail": "candidate received",
                                    "rollout_id": f"rollout-{version}"},
                              f"Accepted candidate version {version}")
        self._post_stage("none", "shadow", 0.0, version)
        return answer

    def _on_hold(self, msg: Message) -> Message:
        reason = msg.payload.get("reason", "")
        self.holds[msg.incident_id or msg.id] = reason
        return self._answer(msg, {"decision": "accepted", "detail": "release on hold"},
                            f"Hold accepted: {reason[:80]}")

    def _on_human_decision(self, msg: Message) -> None:
        if not self.pending:
            return None
        st = state.get_release_state()
        version = self.pending["version"]
        if msg.payload.get("decision") == "approve":
            self.previous_champion = st["champion_version"]   # gardee pour un rollback
            state.set_release_state(champion_version=version, candidate_version=None,
                                    stage="none", candidate_share=0.0, shadow_on=0)
            self._move_alias(version)
            self._post_stage("50%", "100%", 1.0, version)
            self.post_promotion = {"version": str(version), "previous": str(st["champion_version"]),
                                   "since": state._now()}
        else:
            # refus humain : retour propre au champion (pas de hold bloquant)
            state.set_release_state(stage="none", candidate_share=0.0,
                                    shadow_on=0, candidate_version=None)
            self._post_rollback(
                st, {"metric": "human", "value": 0, "limit": 0}, False,
                f"Human rejected {version}; back to {st['champion_version']}")
        self.pending = None
        return None

    def _on_query_changed(self, msg: Message) -> Message:
        since = msg.payload.get("since", "")
        changes = [h for h in self.history if h["time"] >= since]
        return self._answer(msg, {"changes": changes}, f"{len(changes)} deployment change(s)")

    # ---- Boucle -----------------------------------------------------
    def step(self, inbox: List[Message]) -> List[Message]:
        for msg in inbox:
            self.safe_handle(msg)
        self._check_recovery()
        self._check_post_promotion()
        analysis = self.analyze(self.observe())
        if not analysis.flag:
            return self.outbox
        decision = self.decide(analysis)
        if decision["action"] == "hold":
            self.last_hold = decision["params"].get("reason")
            return self.outbox
        result = self.act(decision, analysis)
        if result is not None:
            self._remember(analysis, decision, self.verify(decision))
        return self.outbox


if __name__ == "__main__":
    agent = DeployerAgent()
    test = Message(
        type=MessageType.REQUEST, kind="request.deploy_candidate",
        sender="trainer", to=["deployer"], topic="model:wine-quality@2",
        level=1, summary="Test candidate", payload={"version": "2"})
    out = agent.step([test])
    print("messages:", [(m.kind, m.payload.get("decision", m.payload.get("to_stage"))) for m in out])