import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from agents.deployer import gen_examples
from shared.messages import Message

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"id", "layer", "author", "target", "start", "duration", "applies_to",
            "inject", "revert", "expected"}


def fault_files():
    return sorted((ROOT / "harness" / "faults" / "serving").glob("*.yaml"))


def test_three_serving_faults_exist():
    assert {f.stem for f in fault_files()} == {
        "candidate_latency_spike", "candidate_error_burst", "traffic_surge"}


@pytest.mark.parametrize("path", fault_files(), ids=lambda p: p.stem)
def test_fault_file_follows_the_roadmap_format(path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert REQUIRED <= set(data)
    assert data["layer"] == "serving" and data["author"] == "P5"
    assert data["expected"]["owner"] == "deployer"
    assert "promote_100" in data["expected"]["must_not"]
    for key in ("inject", "revert"):
        module, func = data[key].split(":")
        assert callable(getattr(importlib.import_module(module), func))


@pytest.mark.parametrize("name", ["G.yaml", "G_errors.yaml"])
def test_scenario_g_references_an_existing_fault(name):
    sc = yaml.safe_load((ROOT / "scenarios" / name).read_text(encoding="utf-8"))
    ids = {yaml.safe_load(f.read_text())["id"] for f in fault_files()}
    assert sc["fault"] in ids and sc["expect"]["owner"] == "deployer"
    assert sc["expect"]["max_human_cards"] == 0


def test_fault_functions_call_the_admin_endpoint(monkeypatch):
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            seen.append((self.path, json.loads(self.rfile.read(n))))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("API_URL", f"http://127.0.0.1:{server.server_port}")
    from harness.faults import serving
    try:
        serving.inject_slow({"slow_ms": 400})
        serving.revert_slow({})
        serving.inject_errors({"err_pct": 10})
    finally:
        server.shutdown()
    assert seen == [("/admin/fault", {"slow_ms": 400.0}), ("/admin/fault", {}),
                    ("/admin/fault", {"err_pct": 10.0})]


def test_generated_examples_cover_every_message_the_deployer_sends(tmp_path):
    written = gen_examples.generate(str(tmp_path))
    kinds = {p.parent.name for p in written}
    assert kinds == set(gen_examples.KINDS)
    for p in written:
        msg = Message.model_validate_json(p.read_text(encoding="utf-8"))
        assert msg.sender == "deployer" and msg.kind == p.parent.name
        assert msg.summary and msg.payload
