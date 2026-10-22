"""
Release manager of the prediction API (P5, Deployer).
Holds champion + candidate models, picks the variant of each request,
buffers the prediction log and writes it to board.db every second.
Self-contained on purpose: only baseline/app is copied into the API image.
"""
import json
import os
import random
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone

MODEL_NAME = "wine-quality"

SCHEMAS = [
    """CREATE TABLE IF NOT EXISTS release_state (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        champion_version TEXT, candidate_version TEXT, stage TEXT,
        candidate_share REAL, shadow_on INTEGER, updated_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS predictions (
        request_id TEXT PRIMARY KEY, wine_id INTEGER, time TEXT,
        model_version TEXT, variant TEXT, features TEXT, prediction REAL,
        latency_ms REAL, status_code INTEGER)""",
    """CREATE TABLE IF NOT EXISTS labels (
        wine_id INTEGER PRIMARY KEY, quality REAL, available_at TEXT)""",
]

STATE_KEYS = ("champion_version", "candidate_version", "stage",
              "candidate_share", "shadow_on")


def _now():
    return datetime.now(timezone.utc).isoformat()


class ReleaseManager:
    def __init__(self, db_path=None, loader=None, model_name=MODEL_NAME):
        self.db_path = db_path or os.getenv("BOARD_DB", "board.db")
        self.loader = loader
        self.model_name = model_name
        self.state = {"champion_version": "1", "candidate_version": None,
                      "stage": "none", "candidate_share": 0.0, "shadow_on": 0}
        self.champion = None
        self.champion_version = None
        self.candidate = None
        self.candidate_version = None
        self.faults = {"slow_ms": None, "err_pct": None}   # fautes a chaud (scenario G)
        self._buffer = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    # ---- base de donnees --------------------------------------------
    def _connect(self):
        folder = os.path.dirname(self.db_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        con = sqlite3.connect(self.db_path, timeout=10)
        con.execute(f"PRAGMA journal_mode={os.getenv('SQLITE_JOURNAL', 'WAL')}")
        for schema in SCHEMAS:
            con.execute(schema)
        con.row_factory = sqlite3.Row
        return con

    # ---- modeles ----------------------------------------------------
    def _load(self, uri):
        if self.loader:
            return self.loader(uri)
        import mlflow.pyfunc
        return mlflow.pyfunc.load_model(uri)

    def refresh(self):
        """Relit release_state et recharge les modeles si les versions ont change."""
        row = None
        try:
            con = self._connect()
            row = con.execute("SELECT * FROM release_state WHERE id = 1").fetchone()
            con.close()
        except sqlite3.Error as exc:
            print(f"[release] cannot read release_state: {exc}")
        if row:
            self.state = {k: row[k] for k in STATE_KEYS}
        self._sync_models()

    def _sync_models(self):
        cv = self.state["champion_version"]
        if self.champion is None or (cv and cv != self.champion_version):
            self.champion = self._load(f"models:/{self.model_name}@champion")
            self.champion_version = cv
        cand = self.state["candidate_version"]
        if cand is None:
            self.candidate, self.candidate_version = None, None
        elif cand != self.candidate_version:
            try:
                self.candidate = self._load(f"models:/{self.model_name}/{cand}")
                self.candidate_version = cand
            except Exception as exc:
                print(f"[release] candidate {cand} not loaded: {exc}")
                self.candidate, self.candidate_version = None, None

    # ---- choix du variant -------------------------------------------
    def route(self, force=None):
        """Renvoie (variant, shadow). En shadow, le champion repond et le
        candidat predit en plus, seulement pour le journal.
        force='candidate' (rejeu de charge) impose le candidat s'il est charge."""
        if force == "candidate" and self.candidate is not None:
            return "candidate", False
        if force == "champion":
            return "champion", False
        st = self.state
        if self.candidate is None or st["stage"] == "none":
            return "champion", False
        if st["shadow_on"]:
            return "champion", True
        if random.random() < float(st["candidate_share"] or 0.0):
            return "candidate", False
        return "champion", False

    def apply(self, **fields):
        """Utilise par POST /admin/release : effet immediat, sans attendre 5 s."""
        for key, value in fields.items():
            if key in ("stage", "candidate_share", "shadow_on") and value is not None:
                self.state[key] = value
        return self.state

    # ---- fautes (scenario G) ----------------------------------------
    def set_fault(self, slow_ms=None, err_pct=None):
        """Fautes a chaud via POST /admin/fault. None = revient a la variable d'environnement."""
        self.faults = {"slow_ms": slow_ms, "err_pct": err_pct}
        return self.faults

    def candidate_fault_plan(self):
        """Renvoie le delai (secondes) a ajouter au candidat et leve l'erreur injectee s'il y en a.
        L'API fait `await asyncio.sleep(delai)` : un time.sleep bloquerait tout le serveur,
        champion compris. Sources : /admin/fault, sinon SLOW_MS / CANDIDATE_ERR_PCT."""
        slow = self.faults["slow_ms"]
        err = self.faults["err_pct"]
        slow = float(os.getenv("SLOW_MS", "0")) if slow is None else float(slow)
        err = float(os.getenv("CANDIDATE_ERR_PCT", "0")) if err is None else float(err)
        if err > 0 and random.random() * 100 < err:
            raise RuntimeError("injected candidate error")
        return max(slow, 0.0) / 1000.0

    def candidate_fault(self):
        """Version bloquante (tests, scripts) : attend le delai puis applique la faute."""
        delay = self.candidate_fault_plan()
        if delay > 0:
            time.sleep(delay)

    # ---- journal des predictions ------------------------------------
    def log(self, variant, model_version, wine_id, features, prediction,
            latency_ms, status_code):
        row = (f"req-{uuid.uuid4().hex[:12]}", wine_id, _now(), str(model_version),
               variant, json.dumps(features),
               None if prediction is None else float(prediction),
               round(float(latency_ms), 2), int(status_code))
        with self._lock:
            self._buffer.append(row)
            if len(self._buffer) > 10000:       # garde-fou memoire
                self._buffer = self._buffer[-10000:]

    def flush(self):
        with self._lock:
            rows, self._buffer = self._buffer, []
        if not rows:
            return 0
        try:
            con = self._connect()
            con.executemany(
                "INSERT OR IGNORE INTO predictions VALUES (?,?,?,?,?,?,?,?,?)", rows)
            con.commit()
            con.close()
            return len(rows)
        except sqlite3.Error as exc:
            print(f"[release] flush failed, will retry: {exc}")
            with self._lock:
                self._buffer = rows + self._buffer
            return 0

    # ---- tache de fond ----------------------------------------------
    def start(self, flush_every=1.0, refresh_every=5):
        def loop():
            tick = 0
            while not self._stop.wait(flush_every):
                tick += 1
                try:
                    self.flush()
                    if tick % refresh_every == 0:
                        self.refresh()
                except Exception as exc:
                    print(f"[release] background error: {exc}")
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.flush()
