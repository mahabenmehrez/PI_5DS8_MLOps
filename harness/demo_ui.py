"""
Web demo of the Data Checker: steps 2 to 4 of agents/checker/DEMO.md, with buttons.
Runs the real agent code on every click (nothing is precomputed). Local only, no extra package.

    .venv/Scripts/python -m harness.demo_ui        # then open http://127.0.0.1:8765
"""
from __future__ import annotations

import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).resolve().parent / "demo_ui.html"
HOST, PORT = "127.0.0.1", int(os.environ.get("DEMO_PORT", 8765))

_lock = threading.Lock()            # the shared clock is global: one run at a time


def llm_client():
    """shared/llm.py with agents.yaml / .env (raises when the key is missing)."""
    from harness.demo_checker import real_llm
    return real_llm()[0]


def api_status(_q) -> dict:
    try:
        from harness.demo_checker import real_llm
        return {"llm": True, "llm_name": real_llm()[1]}
    except Exception as e:
        return {"llm": False, "llm_error": str(e)}


def api_cases(q) -> dict:
    from harness.demo_checker import CASES, case_result
    llm = llm_client() if q.get("llm") == "1" else None
    return {"cases": [case_result(fid, label, llm) for fid, label in CASES]}


def api_scenario(_q) -> dict:
    from harness.scenario_runner import run_scenario
    r = run_scenario("C")
    return {"passed": r.passed, "problems": r.problems, "batches": r.batches,
            "messages": [{"time": m.time[11:16], "sender": m.sender, "to": m.to, "kind": m.kind,
                          "type": m.type.value, "summary": m.summary,
                          "status": m.payload.get("status")} for m in r.messages]}


def api_evaluate(q) -> dict:
    from harness.evaluate_checker import evaluate
    llm = llm_client() if q.get("llm") == "1" else None
    seeds, normal = (1, 15) if llm else (3, 60)       # fewer LLM calls on the free quota
    r = evaluate(normal, seeds, llm)
    per15 = 15 * (r["false_alarms"] + r["warnings_on_normal"]) / r["normal_batches"]
    faults = {}
    for fid, status, reason, ok in r["faults"]:
        f = faults.setdefault(fid, {"fault": fid, "runs": 0, "ok": 0, "status": status})
        f["runs"] += 1
        f["ok"] += bool(ok)
    return {"accuracy": r["fault_accuracy"], "normal_batches": r["normal_batches"],
            "false_alarms": r["false_alarms"], "warnings": r["warnings_on_normal"],
            "alarms_per_15": round(per15, 1), "faults": list(faults.values()),
            "seeds": seeds, "llm": bool(llm)}


ROUTES = {"/api/status": api_status, "/api/cases": api_cases,
          "/api/scenario": api_scenario, "/api/evaluate": api_evaluate}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        route = ROUTES.get(url.path)
        if route is None:
            return self.send(404, b"not found", "text/plain")
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            with _lock:
                body = route(query)
            return self.send(200, json.dumps(body, default=str).encode(), "application/json")
        except Exception as e:
            traceback.print_exc()
            return self.send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(),
                             "application/json")

    def send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):     # keep the console quiet during the demo
        pass


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Démo du Data Checker : http://{HOST}:{PORT}   (Ctrl+C pour arrêter)")
    server.serve_forever()


if __name__ == "__main__":
    main()
