import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from dry_run_jobs import traffic_replay as tr


def make_server(delay=0.0, fail_every=0):
    seen = {"headers": [], "n": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            with lock:                       # compteur protege : plusieurs threads en meme temps
                seen["n"] += 1
                number = seen["n"]
                seen["headers"].append((self.headers.get("X-Variant"), self.headers.get("X-Dry-Run")))
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            time.sleep(delay)
            if fail_every and number % fail_every == 0:
                self.send_response(500)
                self.end_headers()
                return
            body = json.dumps({"prediction": 5.0}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    class Server(ThreadingHTTPServer):
        request_queue_size = 128          # 5 par defaut : debordait avec 8 connexions simultanees

    server = Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def reqs(n=40):
    return [(float(i), [1.0] * 11) for i in range(n)]


def test_replay_healthy_api_passes():
    server, seen = make_server(delay=0.005)
    try:
        res = tr.evaluate(tr.replay(f"http://127.0.0.1:{server.server_port}", reqs(),
                                    speed=1000, variant="candidate"),
                          p95_limit=3000)          # on teste ici les erreurs et les en-tetes, pas le chrono
    finally:
        server.shutdown()
    assert res["ok"] and res["n"] == 40 and res["error_rate"] == 0
    assert all(h == ("candidate", "1") for h in seen["headers"])   # variant force, dry run


def test_replay_slow_api_fails():
    server, _ = make_server(delay=0.15)
    try:
        res = tr.evaluate(tr.replay(f"http://127.0.0.1:{server.server_port}", reqs(30),
                                    speed=1000), p95_limit=100)
    finally:
        server.shutdown()
    assert not res["ok"] and "p95" in res["detail"]


def test_replay_erroring_api_fails():
    server, _ = make_server(fail_every=3)
    try:
        res = tr.evaluate(tr.replay(f"http://127.0.0.1:{server.server_port}", reqs(30),
                                    speed=1000))
    finally:
        server.shutdown()
    assert not res["ok"] and res["error_rate"] > 0.25


def test_replay_unreachable_api_counts_errors():
    res = tr.replay("http://127.0.0.1:9", reqs(5), speed=1000, timeout=0.5)
    assert res["error_rate"] == 1.0


def test_replay_empty_is_not_ok():
    res = tr.evaluate(tr.replay("http://127.0.0.1:9", []))
    assert not res["ok"]


def test_load_requests_skips_rows_without_features(tmp_path, monkeypatch):
    monkeypatch.setenv("BOARD_DB", str(tmp_path / "b.db"))
    from agents.deployer import state
    mk = lambda i, f: {"request_id": f"r{i}", "wine_id": i, "time": f"2099-01-01T00:00:0{i}+00:00",
                       "model_version": "2", "variant": "candidate", "features": f,
                       "prediction": 5.0, "latency_ms": 20.0, "status_code": 200}
    state.log_predictions([mk(1, "[]"), mk(2, json.dumps([1.0] * 11)), mk(3, json.dumps([2.0] * 11))])
    out = tr.load_requests(str(tmp_path / "b.db"))
    assert len(out) == 2 and out[0][0] == 0.0 and out[1][0] == 1.0


def test_replay_gives_up_after_the_hard_deadline():
    server, _ = make_server(delay=0.3)
    try:
        start = time.time()
        res = tr.replay(f"http://127.0.0.1:{server.server_port}", reqs(60), speed=1000,
                        workers=2, timeout=1.0, hard_deadline=0.5)
    finally:
        server.shutdown()
    assert time.time() - start < 6                  # 60 x 0,3 s / 2 = 9 s sans echeance
    assert res["n"] == 60 and res["error_rate"] > 0.5