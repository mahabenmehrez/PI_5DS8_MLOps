"""
Dry run job `traffic_replay` (roadmap 5.10 et 11.2).

Rejoue des requetes deja enregistrees (table predictions) contre l'API, a 3x la vitesse
d'origine, en forcant le candidat (en-tete X-Variant) sans rien ecrire dans les vraies
tables (en-tete X-Dry-Run, plus DRY_RUN=1 pour ce processus). Mesure latence et erreurs.

Lancer a la main :
    python -m dry_run_jobs.traffic_replay --url http://localhost:8000 --db board.db --speed 3
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Tuple

Request = Tuple[float, list]      # (decalage en secondes, 11 mesures)


def load_requests(db_path: str, limit: int = 300) -> List[Request]:
    """Lit les dernieres requetes enregistrees qui ont de vraies mesures (features non vides)."""
    con = sqlite3.connect(db_path, timeout=10)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT time, features FROM predictions WHERE features IS NOT NULL AND features != '[]'"
        " ORDER BY time DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    rows = list(reversed(rows))
    if not rows:
        return []
    t0 = datetime.fromisoformat(rows[0]["time"])
    out = []
    for r in rows:
        features = json.loads(r["features"])
        if features:
            out.append(((datetime.fromisoformat(r["time"]) - t0).total_seconds(), features))
    return out


def _post(url: str, features: list, variant: str | None, timeout: float,
          dry: bool = True) -> Tuple[float, bool]:
    headers = {"Content-Type": "application/json"}
    if dry:
        headers["X-Dry-Run"] = "1"
    if variant:
        headers["X-Variant"] = variant
    req = urllib.request.Request(url.rstrip("/") + "/predict",
                                 data=json.dumps({"data": features}).encode("utf-8"),
                                 headers=headers)
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            failed = resp.status >= 500
    except urllib.error.HTTPError as exc:
        failed = exc.code >= 500
    except Exception:
        failed = True
    return (time.time() - start) * 1000.0, failed


def replay(url: str, requests: List[Request], speed: float = 3.0, max_requests: int = 300,
           workers: int = 8, timeout: float = 10.0, variant: str | None = "candidate",
           max_seconds: float = 60.0, dry: bool = True,
           hard_deadline: float | None = None) -> dict:
    """Rejoue `requests` a `speed` fois la vitesse d'origine (jamais plus de max_seconds)."""
    requests = requests[:max_requests]
    if not requests:
        return {"n": 0, "p95_ms": 0.0, "error_rate": 0.0, "seconds": 0.0,
                "speed": speed, "detail": "aucune requete enregistree a rejouer"}
    span = requests[-1][0]
    effective = max(speed, span / max_seconds) if span > 0 else speed
    start = time.time()
    hard_deadline = hard_deadline if hard_deadline is not None else max_seconds + 30.0
    results: List[Tuple[float, bool]] = []
    lock = threading.Lock()

    def worker(item: Request) -> None:
        offset, features = item
        delay = offset / effective - (time.time() - start)
        if delay > 0:
            time.sleep(delay)
        if time.time() - start > hard_deadline:       # trop lent : on arrete d'attendre
            out = (timeout * 1000.0, True)
        else:
            out = _post(url, features, variant, timeout, dry)
        with lock:
            results.append(out)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, requests))
    latencies = sorted(r[0] for r in results)
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
    return {"n": len(results), "p95_ms": round(p95, 1),
            "mean_ms": round(statistics.mean(latencies), 1),
            "error_rate": sum(r[1] for r in results) / len(results),
            "seconds": round(time.time() - start, 2), "speed": round(effective, 1)}


def evaluate(result: dict, p95_limit: float = 300.0, error_limit: float = 0.05,
             min_requests: int = 20) -> dict:
    """Decide si le dry run est reussi, avec une phrase d'explication."""
    problems = []
    if result["n"] < min_requests:
        problems.append(f"seulement {result['n']} requetes rejouees (< {min_requests})")
    if result["p95_ms"] > p95_limit:
        problems.append(f"p95 {result['p95_ms']} ms > {p95_limit} ms")
    if result["error_rate"] > error_limit:
        problems.append(f"erreurs {result['error_rate']:.1%} > {error_limit:.0%}")
    detail = "; ".join(problems) or (
        f"{result['n']} requetes a {result['speed']}x : p95 {result['p95_ms']} ms, "
        f"erreurs {result['error_rate']:.1%}")
    return {**result, "ok": not problems, "detail": detail}


def main() -> None:
    os.environ["DRY_RUN"] = "1"
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.getenv("API_URL", "http://localhost:8000"))
    ap.add_argument("--db", default=os.getenv("BOARD_DB", "board.db"))
    ap.add_argument("--speed", type=float, default=3.0)
    ap.add_argument("--max", type=int, default=300)
    ap.add_argument("--variant", default="candidate")
    ap.add_argument("--p95", type=float, default=300.0)
    ap.add_argument("--errors", type=float, default=0.05)
    args = ap.parse_args()
    reqs = load_requests(args.db, args.max)
    print(json.dumps(evaluate(replay(args.url, reqs, args.speed, args.max, variant=args.variant),
                              args.p95, args.errors)))


if __name__ == "__main__":
    main()
