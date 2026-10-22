"""Fonctions `replay` a donner a DeployerAgent(replay=...)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def subprocess_replay(api_url: str, db_path: str | None = None, speed: float = 3.0,
                      timeout_s: int = 120):
    """Lance le job dry_run_jobs/traffic_replay.py dans un sous-processus, dans un dossier
    temporaire, avec DRY_RUN=1 (roadmap 5.10). Renvoie une fonction (state, analysis) -> dict."""
    root = Path(__file__).resolve().parents[2]

    def run(release_state, analysis=None) -> dict:
        env = {**os.environ, "DRY_RUN": "1", "PYTHONPATH": str(root),
               "BOARD_DB": db_path or os.getenv("BOARD_DB", "board.db")}
        with tempfile.TemporaryDirectory(prefix="traffic_replay_") as tmp:
            proc = subprocess.run(
                [sys.executable, "-m", "dry_run_jobs.traffic_replay", "--url", api_url,
                 "--db", env["BOARD_DB"], "--speed", str(speed)],
                cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout_s)
        if proc.returncode != 0:
            return {"ok": False, "detail": f"job en echec : {proc.stderr.strip()[-200:]}"}
        return json.loads(proc.stdout.strip().splitlines()[-1])
    return run


def simulated_replay(factor: float = 1.15, p95_limit: float = 300.0, error_limit: float = 0.05):
    """Pour la demo (sans API) : estime la charge a 3x a partir de ce que l'agent a mesure.
    A ne PAS presenter comme un vrai test de charge."""
    def run(release_state, analysis=None) -> dict:
        cand = analysis.numbers["candidate"]
        p95 = round(cand["p95_ms"] * factor, 1)
        err = cand["error_rate"]
        ok = p95 <= p95_limit and err <= error_limit
        return {"ok": ok, "simulated": True, "n": cand["n"], "p95_ms": p95, "error_rate": err,
                "detail": f"simule : p95 {p95} ms a 3x, erreurs {err:.1%}"}
    return run
