"""Web demo of the Setup Agent (P1), in English, with buttons. Each click runs the real agent on a copy of the project.

In the agents' container (make, git and the Linux tools the dry run needs are there), from the repo root:

    docker compose -f docker-compose.agents.yml run --rm -p 127.0.0.1:8766:8766 -e DEMO_HOST=0.0.0.0 setup python -m agents.setup.demo

then open http://127.0.0.1:8766. Outside the container: python -m agents.setup.demo (needs make on the PATH).
Local only, no extra package. The whole demo is this folder: deleting it removes it, the agent doesn't use it.
"""
from __future__ import annotations

import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from agents.setup.demo import runner
from agents.setup.demo.faults import FAULTS

PAGE = Path(__file__).resolve().parent / "page.html"
HOST, PORT = os.environ.get("DEMO_HOST", "127.0.0.1"), int(os.environ.get("DEMO_PORT", 8766))  # Checker: 8765


def api_faults(_q) -> dict:
    return {"faults": [{"id": f.id, "label": f.label, "what": f.what, "code": f.code.strip(),
                        "expected": f.expected, "needs_llm": f.needs_llm, "expect": f.expect} for f in FAULTS]}


def api_run(q) -> dict:
    return runner.start(q.get("action", ""), q.get("fault") or None, q.get("mode", "rules")).view()


def api_job(q) -> dict:
    job = runner.JOBS.get(q.get("id", ""))
    if job is None:
        raise KeyError("unknown run")
    return job.view(int(q.get("since", 0)))


def api_replay(q) -> dict:
    rec = runner.recording(q.get("action", ""), q.get("fault") or None, q.get("mode", "rules"))
    if rec is None:
        raise KeyError("no recording of this run yet: run it once live")
    return rec


ROUTES = {"/api/status": lambda q: runner.status(), "/api/analyse": lambda q: runner.analyse(),
          "/api/faults": api_faults, "/api/run": api_run, "/api/job": api_job, "/api/replay": api_replay,
          "/api/results": lambda q: runner.results()}


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
            body = route(query)
            return self.send(200, json.dumps(body, default=str, ensure_ascii=False).encode(), "application/json")
        except runner.Busy as e:
            return self.send(409, json.dumps({"error": str(e)}).encode(), "application/json")
        except (KeyError, ValueError) as e:
            return self.send(400, json.dumps({"error": str(e).strip("'")}).encode(), "application/json")
        except Exception as e:  # noqa: BLE001 - shown on the page
            traceback.print_exc()
            return self.send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")

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
    shown = "127.0.0.1" if HOST == "0.0.0.0" else HOST
    print(f"Setup Agent demo: http://{shown}:{PORT}   (Ctrl+C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
