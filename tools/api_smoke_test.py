"""
Test de bout en bout de l'API reelle (a lancer APRES avoir demarre la stack Docker, avec un
champion v1 et un candidat v2 dans MLflow).

    set API_URL=http://localhost:8000
    set BOARD_DB=board\\board.db
    set SQLITE_JOURNAL=DELETE
    python tools/api_smoke_test.py

Verifie : le candidat est charge, le partage de trafic est respecte, les predictions arrivent
dans la table predictions, les fautes a chaud marchent, le rejeu de charge fonctionne.
"""
import json
import os
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.getcwd())

from agents.deployer import state                       # noqa: E402
from dry_run_jobs.traffic_replay import evaluate, load_requests, replay   # noqa: E402

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
failures = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(("PASS  " if ok else "FAIL  ") + label + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(label)


def call(path, body=None, headers=None):
    req = urllib.request.Request(
        API + path, data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})})
    start = time.time()
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read()), resp.headers, (time.time() - start) * 1000


def wine():
    return [round(random.uniform(4, 12), 2) for _ in range(11)]


print(f"API : {API}   base : {os.getenv('BOARD_DB', 'board.db')}\n")

# 1. l'API repond
info, _, _ = call("/admin/release")
check(info["champion_loaded"] is not None, "l'API a charge le champion", str(info["champion_loaded"]))

# 2. on declare un candidat v2 a 50 % et on attend que l'API le charge
state.set_release_state(champion_version=str(info["champion_loaded"]), candidate_version="2",
                        stage="10%", candidate_share=0.5, shadow_on=0)
for _ in range(20):
    info, _, _ = call("/admin/release")
    if info["candidate_loaded"] == "2":
        break
    time.sleep(1)
check(info["candidate_loaded"] == "2", "l'API a charge le candidat v2 (rechargement toutes les 5 s)")

# 3. partage du trafic
served = {"champion": 0, "candidate": 0}
for i in range(200):
    _, headers, _ = call("/predict", {"data": wine(), "wine_id": i})
    served[headers["X-Served-Variant"]] += 1
share = served["candidate"] / 200
check(0.35 <= share <= 0.65, "partage de trafic proche de 50 %", f"{served}")

# 4. journal des predictions
time.sleep(3)
rows = state.read_predictions()
by_variant = {}
for r in rows:
    by_variant.setdefault(r["variant"], set()).add(r["model_version"])
check(len(rows) >= 200, "les predictions arrivent dans la table predictions", f"{len(rows)} lignes")
check(by_variant.get("candidate") == {"2"}, "le candidat est journalise avec model_version=2",
      str(by_variant))

# 5. faute a chaud + rejeu force du candidat
call("/admin/fault", {"slow_ms": 300})
_, _, latency = call("/predict", {"data": wine()}, {"X-Variant": "candidate", "X-Dry-Run": "1"})
check(latency >= 300, "la faute SLOW_MS ralentit le candidat", f"{latency:.0f} ms")
call("/admin/fault", {})
_, _, latency = call("/predict", {"data": wine()}, {"X-Variant": "candidate", "X-Dry-Run": "1"})
check(latency < 300, "la faute est retiree", f"{latency:.0f} ms")

# 6. dry run traffic_replay (3x) sans toucher aux tables
before = len(state.read_predictions())
result = evaluate(replay(API, load_requests(os.getenv("BOARD_DB", "board.db"), 150), speed=3.0))
time.sleep(2)
check(result["ok"], "traffic_replay 3x reussit", result["detail"])
check(len(state.read_predictions()) == before, "le dry run n'ecrit rien dans predictions")

# 7. remise a zero
state.set_release_state(candidate_version=None, stage="none", candidate_share=0.0, shadow_on=0)
print("\n" + ("TOUT EST BON" if not failures else f"{len(failures)} verification(s) en echec"))
sys.exit(1 if failures else 0)
