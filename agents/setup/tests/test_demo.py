"""The demo page (agents/setup/demo): the faults go in the right place, the run reports every step, the server answers.
The demo is removable: delete agents/setup/demo/ and this file."""
import json
import shutil
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from agents.setup.demo import runner
from agents.setup.demo.faults import BY_ID, FAULTS, inject, insert_at

ROOT = Path(__file__).resolve().parents[3]
needs_make = pytest.mark.skipif(shutil.which("make") is None or not (ROOT / "data" / "winequality-red.csv").exists(),
                                reason="needs make and the wine data")


def test_the_fault_goes_after_the_docstring_and_future_imports(tmp_path):
    script = tmp_path / "train.py"
    script.write_text('"""Train.\n\nMore."""\nfrom __future__ import annotations\n\nimport os\nprint(os.sep)\n')
    assert insert_at(script.read_text()) == 4
    where = inject(script, BY_ID["numpy_float"], shown="baseline/train.py")
    assert where == "baseline/train.py, line 5"
    compile(script.read_text(), "train.py", "exec")                    # still valid Python: no SyntaxError
    assert script.read_text().splitlines()[4] == "import numpy as _np"


def test_every_fault_keeps_the_real_training_script_valid(tmp_path):
    for fault in FAULTS:
        script = tmp_path / f"{fault.id}.py"
        shutil.copy(ROOT / "baseline" / "train.py", script)
        inject(script, fault)
        compile(script.read_text(encoding="utf-8"), script.name, "exec")
        assert fault.expect in ("build", "report") and fault.label and fault.expected


def test_stages_and_error_lines_say_what_failed():
    assert [s["state"] for s in runner.stages({"ok": False, "stage": "train"})] == \
        ["ok", "ok", "ok", "failed", "skipped", "skipped"]
    assert {s["state"] for s in runner.stages({"ok": True})} == {"ok"}
    log = ('Collecting numpy\nTraceback (most recent call last):\n  File "/tmp/setup-dryrun-x/repo/baseline/train.py", '
           'line 6, in <module>\n    _np.float(1.0)\n  File "/v/lib/python3.10/site-packages/numpy/__init__.py", line 324\n'
           "AttributeError: module 'numpy' has no attribute 'float'.\nmake: *** [Makefile:52: train] Error 1\n")
    lines = runner.error_lines(log)
    assert lines[-1].startswith("AttributeError") and '"baseline/train.py"' in lines[1] and "Collecting" not in lines[0]
    assert ".../numpy/__init__.py" in "\n".join(lines)
    assert runner.error_name({"error_class": "Error", "log": log}) == "AttributeError"


@needs_make
def test_a_rules_run_reports_each_step_and_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RECORDINGS", tmp_path / "rec")
    job = runner.start("fault", "missing_data_file", "rules")
    with pytest.raises(runner.Busy):
        runner.start("build", None, "rules")                            # one run at a time
    deadline = time.time() + 600
    while not job.done and time.time() < deadline:
        time.sleep(0.5)
    kinds = [e["type"] for e in job.events]
    assert kinds[:3] == ["start", "copy", "inject"] and kinds[-1] == "done"
    assert "build" in kinds and "build_done" in kinds and "rule" in kinds
    failed = next(e for e in job.events if e["type"] == "build_done")
    assert failed["error_class"] == "FileNotFoundError" and failed["stages"][3]["state"] == "failed"
    assert job.result["kind"] == "pipeline.setup_failed" and job.result["right"]          # reported, as it should be
    rec = runner.recording("fault", "missing_data_file", "rules")
    assert rec and rec["events"] == job.events


def test_the_server_answers(monkeypatch, tmp_path):
    from agents.setup.demo.__main__ import Handler
    monkeypatch.setattr(runner, "RECORDINGS", tmp_path / "none")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "<title>Setup Agent — demo</title>" in page
        faults = json.load(urllib.request.urlopen(base + "/api/faults"))["faults"]
        assert [f["id"] for f in faults] == [f.id for f in FAULTS]
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/api/replay?action=build&mode=rules")      # nothing recorded yet
        assert e.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/api/run?action=build&mode=everything")
        assert e.value.code == 400
    finally:
        server.shutdown()
