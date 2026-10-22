"""
deployer_live.py - fait tourner TON VRAI DeployerAgent (agents/deployer/agent.py).

Ce qui est reel :
  - la classe DeployerAgent, ses portes (analysis.py), sa machine d'etats
  - les tables release_state, predictions et labels (state.py), dans une base SQLite TEMPORAIRE
  - les messages (shared.messages) envoyes et recus par l'agent
  - le journal de souvenirs (memory.py), dans la meme base temporaire

Ce qui est simule :
  - le trafic (requetes, latences, erreurs, predictions)  -> ce fichier le fabrique
  - l'horloge : 1 tick = 1 minute simulee (state._now est remplace pendant la demo)
  - les autres agents : Trainer, Monitor, humain = messages envoyes a la main

A placer a la RACINE du depot (a cote de agents/ et shared/).
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from shared import memory                                  # noqa: E402
from shared.messages import Message, MessageType          # noqa: E402
from agents.deployer import analysis as rules             # noqa: E402
from agents.deployer import state                         # noqa: E402
from agents.deployer.agent import DeployerAgent           # noqa: E402

SCENARIOS = {
    "Candidat sain": dict(extra_ms=0, err=0.002, sigma=0.65, bad=False),
    "Candidat lent (scenario G)": dict(extra_ms=400, err=0.002, sigma=0.65, bad=True),
    "Candidat en erreur (10 %)": dict(extra_ms=0, err=0.10, sigma=0.65, bad=True),
    "Candidat moins precis": dict(extra_ms=0, err=0.002, sigma=1.0, bad=True),
}
ACTIVE = ("shadow", "10%", "50%")
INCIDENT = "inc-demo0001"


class LiveHarness:
    REQ_PER_MIN = 60
    LABEL_DELAY_MIN = 5

    def __init__(self, scenario: str = "Candidat sain", seed: int = 42,
                 use_memory: bool = True, llm=None, replay=None):
        self.scenario, self.sc = scenario, SCENARIOS[scenario]
        self.rng = np.random.default_rng(seed)
        self.tmpdir = tempfile.mkdtemp(prefix="deployer_demo_")
        os.environ["BOARD_DB"] = os.path.join(self.tmpdir, "board.db")
        os.environ.pop("DRY_RUN", None)
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        memory.DB_PATH = os.environ["BOARD_DB"]          # journal de souvenirs temporaire
        self.t0 = datetime(2026, 10, 15, 10, 0, 0, tzinfo=timezone.utc)
        self.clock = self.t0
        state._now = lambda: self.clock.isoformat()      # horloge simulee
        state.connect().close()                          # cree les tables

        self.agent = DeployerAgent(cfg={}, llm=llm, replay=replay)  # llm=None : regles seulement
        if not use_memory:
            self.agent._remember = lambda *a, **k: None
        self.inbox: list[Message] = []
        self.log: list[dict] = []
        self._seen = 0
        self._last_hold_key = None
        self.rid = 0
        self.wine = 0
        self.pending_labels: list[tuple] = []
        self.candidate_served = 0     # requetes d'utilisateurs servies par le candidat
        self.bad_served = 0           # ... dont servies par un candidat defectueux

    # ------------------------------------------------------------- journal
    @property
    def minute(self) -> int:
        return int((self.clock - self.t0).total_seconds() // 60)

    def _log(self, direction: str, m: Message):
        self.log.append(dict(
            minute=self.minute, sens=direction, kind=m.kind, sender=m.sender,
            to=", ".join(m.to) if m.to else "tous", level=m.level, summary=m.summary))

    def _incoming(self, msg: Message):
        self.inbox.append(msg)
        self._log("vers agent", msg)

    def _step(self):
        inbox, self.inbox = self.inbox, []
        self.agent.step(inbox)
        out = getattr(self.agent, "outbox", [])
        for m in out[self._seen:]:                       # l'outbox n'est jamais vide : on lit le nouveau
            self._log("agent", m)
        self._seen = len(out)
        hold = self.agent.last_hold
        if hold:
            key = re.sub(r"[\d.]+", "#", hold)
            if key != self._last_hold_key:
                self._last_hold_key = key
                self.log.append(dict(minute=self.minute, sens="decision", kind="action.hold",
                                     sender="deployer", to="(interne)", level=1, summary=hold))

    # --------------------------------------------------- messages des autres
    def send_candidate(self):
        st = state.get_release_state()
        if st["stage"] != "none":
            return
        champion = str(st["champion_version"])
        version = str(int(champion) + 1)
        self._incoming(Message(
            type=MessageType.REQUEST, kind="request.deploy_candidate", sender="trainer",
            to=["deployer"], topic=f"model:wine-quality@{version}", level=1,
            summary=f"Candidat v{version} (champion v{champion})",
            payload={"model_uri": "models:/wine-quality@candidate", "version": version,
                     "champion_version": champion,
                     "metrics": {"overall": {"mae": 0.48}, "by_group": {}},
                     "comparison_ref": "fake-comparison-001"}))
        self._last_hold_key = None
        self.agent.last_hold = None
        self._step()

    def send_hold(self, reason: str):
        self._incoming(Message(
            type=MessageType.REQUEST, kind="request.hold", sender="monitor", to=["deployer"],
            topic="deployment:prod", level=1, summary=reason, incident_id=INCIDENT,
            payload={"reason": reason, "release_condition": "incident ferme"}))
        self._step()

    def release_hold(self):
        self._incoming(Message(
            type=MessageType.FINDING, kind="incident.closed", sender="monitor", to=[],
            topic=f"incident:{INCIDENT}", level=0, summary="Incident ferme, hold leve",
            incident_id=INCIDENT,
            payload={"outcome": "resolved", "root_cause_node": "dataset:batch-0001",
                     "actions": [], "time_to_recover_s": 0}))
        self._step()

    def human_decision(self, approve: bool):
        self._incoming(Message(
            type=MessageType.ANSWER, kind="human.decision", sender="human", to=["deployer"],
            topic="incident:deploy", level=3,
            summary="approve" if approve else "reject",
            payload={"decision": "approve" if approve else "reject", "comment": "demo"}))
        self._step()

    def manual_rollback(self):
        """Appelle directement act('rollback') : l'agent n'a pas de message dedie pour ca."""
        if state.get_release_state()["stage"] not in ACTIVE:
            return
        self.agent.act({"action": "rollback", "level": 2,
                        "params": {"reason": "rollback manuel (demo)"}}, None)
        self.agent.pending = None
        self._step()

    # ------------------------------------------------------------- trafic
    def _row(self, variant: str, version, wine: int, quality: int, t: datetime) -> dict:
        cand = variant != "champion"
        self.rid += 1
        lat = float(self.rng.lognormal(np.log(17), 0.25)) + (self.sc["extra_ms"] if cand else 0)
        err = bool(self.rng.random() < (self.sc["err"] if cand else 0.002))
        sigma = self.sc["sigma"] if cand else 0.65
        pred = None if err else round(float(quality + self.rng.normal(0, sigma)), 3)
        return dict(request_id=f"req-{self.rid:07d}", wine_id=wine, time=t.isoformat(),
                    model_version=str(version), variant=variant, features="[]",
                    prediction=pred, latency_ms=lat, status_code=500 if err else 200)

    def _traffic(self, st: dict) -> list[dict]:
        rows, n = [], self.REQ_PER_MIN
        start = self.clock - timedelta(seconds=60)
        for i in range(n):
            t = start + timedelta(seconds=i * 60 / n)
            self.wine += 1
            q = int(self.rng.choice([3, 4, 5, 6, 7, 8], p=[.01, .03, .43, .40, .11, .02]))
            self.pending_labels.append(
                (self.clock + timedelta(minutes=self.LABEL_DELAY_MIN), self.wine, q))
            if st["stage"] in ("10%", "50%") and self.rng.random() < st["candidate_share"]:
                rows.append(self._row("candidate", st["candidate_version"], self.wine, q, t))
                self.candidate_served += 1
                self.bad_served += 1 if self.sc.get("bad") else 0
            else:
                rows.append(self._row("champion", st["champion_version"], self.wine, q, t))
                if st["shadow_on"]:
                    rows.append(self._row("shadow", st["candidate_version"], self.wine, q, t))
        return rows

    def _release_labels(self):
        ready = [x for x in self.pending_labels if x[0] <= self.clock]
        self.pending_labels = [x for x in self.pending_labels if x[0] > self.clock]
        if ready:
            con = state.connect()
            con.executemany("INSERT OR IGNORE INTO labels (wine_id, quality, available_at) VALUES (?,?,?)",
                            [(w, q, a.isoformat()) for a, w, q in ready])
            con.commit()
            con.close()

    def tick(self):
        """Avance d'une minute simulee, puis laisse l'agent faire un step()."""
        st = state.get_release_state()
        if st["stage"] not in ACTIVE and not self.agent.watching():
            return
        self.clock += timedelta(minutes=1)
        state.log_predictions(self._traffic(st))
        self._release_labels()
        self._step()

    def settle(self, max_minutes: int = 4):
        """Laisse l'agent verifier un rollback (2 minutes) apres la fin du deploiement."""
        for _ in range(max_minutes):
            if not self.agent.watching():
                break
            self.tick()

    # ------------------------------------------------------------ lecture
    def release_state(self) -> dict:
        return state.get_release_state()

    def gate(self):
        """Chiffres que l'agent utilise pour decider (None si aucun deploiement actif)."""
        a = self.agent.analyze(self.agent.observe())
        return a.numbers if a.flag else None

    def memory_rows(self) -> pd.DataFrame:
        rows = memory._all_rows("deployer")
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)[["time", "action", "outcome", "problem_text"]]

    def status(self) -> tuple[str, str]:
        a, st = self.agent, state.get_release_state()
        last = a.history[-1] if a.history else None
        if st["stage"] == "none" and last:
            if last["to_stage"] == "rolled_back":
                return "error", f"Rollback effectué : retour au champion v{st['champion_version']}."
            if last["to_stage"] == "100%":
                return "success", f"v{st['champion_version']} est maintenant champion à 100 %."
        if a.pending:
            return "warning", f"Décision humaine requise : promouvoir v{a.pending['version']} à 100 % ?"
        if a.holds:
            return "warning", "HOLD actif : " + " ; ".join(a.holds.values())
        if st["stage"] == "none":
            return "info", "Aucun nouveau modèle pour l'instant. Clique sur « Démarrer »."
        return "info", a.last_hold or f"Étape {st['stage']} en cours."

    def window_stats(self) -> pd.DataFrame:
        st = state.get_release_state()
        if st["stage"] == "none":
            return pd.DataFrame()
        out = []
        for v in ("champion", "candidate", "shadow"):
            rows = state.read_predictions(v, st["updated_at"])
            if not rows:
                continue
            s = rules.summarize(rows)
            errs = [e for e in state.abs_errors(v, st["updated_at"]) if e is not None]
            out.append({"variante": v, "requetes": s["n"], "p95 (ms)": round(s["p95_ms"], 1),
                        "taux d'erreurs": f"{s['error_rate'] * 100:.1f} %",
                        "MAE (labels recus)": round(float(np.mean(errs)), 3) if errs else None,
                        "n labels": len(errs)})
        return pd.DataFrame(out)

    def latency_chart(self) -> pd.DataFrame:
        rows = state.read_predictions()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["minute"] = df["time"].str[11:16]
        df["groupe"] = np.where(df["variant"] == "champion", "champion", "candidat")
        p = df.groupby(["minute", "groupe"])["latency_ms"].quantile(0.95).unstack()
        p["limite"] = rules.DEFAULTS["limit_p95_ms"]
        return p
