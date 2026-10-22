"""
Fautes de la couche serving (roadmap 13.4). Chaque faute = inject + revert.
Elles agissent sur l'API par ses endpoints d'administration (/admin/fault) ou en envoyant
du vrai trafic en plus (surge). L'API doit etre lancee avec le mecanisme de release du Deployer.
"""
from __future__ import annotations

import json
import os
import urllib.request

from dry_run_jobs.traffic_replay import load_requests, replay


def _api() -> str:
    return os.getenv("API_URL", "http://localhost:8000").rstrip("/")


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(_api() + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---- latence du candidat (scenario G) ---------------------------------
def inject_slow(target: dict) -> dict:
    return _post("/admin/fault", {"slow_ms": float(target["slow_ms"])})


def revert_slow(target: dict) -> dict:
    return _post("/admin/fault", {})


# ---- erreurs du candidat (scenario G) ---------------------------------
def inject_errors(target: dict) -> dict:
    return _post("/admin/fault", {"err_pct": float(target["err_pct"])})


def revert_errors(target: dict) -> dict:
    return _post("/admin/fault", {})


# ---- pic de trafic : requetes reelles en plus (visibles dans le journal) --
def inject_surge(target: dict) -> dict:
    """Envoie `factor` fois le trafic recent pendant au plus `duration_s` secondes."""
    reqs = load_requests(os.getenv("BOARD_DB", "board.db"), int(target.get("max_requests", 500)))
    return replay(_api(), reqs, speed=float(target["factor"]), variant=None, dry=False,
                  max_seconds=float(target.get("duration_s", 30)),
                  max_requests=int(target.get("max_requests", 500)))


def revert_surge(target: dict) -> dict:
    return {"reverted": True}        # le pic s'arrete de lui-meme
