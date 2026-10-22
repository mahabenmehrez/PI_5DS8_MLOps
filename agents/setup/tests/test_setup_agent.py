"""The Setup Agent's loop on a copy of the team repo: first run, a fix, a failure, a rebuild request.

Each test copies the repo into a temp git repo: nothing touches the real project."""
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dry_run_jobs.build import classify  # noqa: E402

from agents.setup.agent import SetupAgent  # noqa: E402
from shared.messages import Message, MessageType  # noqa: E402

COPY = ["baseline", "samples", "data", "agents", "shared", "tests", "contracts"]
needs_make = pytest.mark.skipif(shutil.which("make") is None or not (ROOT / "data" / "winequality-red.csv").exists(),
                                reason="needs make and data/winequality-red.csv")


def add_first(script: Path, code: str) -> None:
    """Put code at the top of a script, after its docstring and `from __future__` imports (those must stay first)."""
    import ast
    text = script.read_text()
    at = 0
    for i, node in enumerate(ast.parse(text).body):
        doc = i == 0 and isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant)
        if not (doc or (isinstance(node, ast.ImportFrom) and node.module == "__future__")):
            break
        at = node.end_lineno
    lines = text.splitlines(keepends=True)
    script.write_text("".join(lines[:at]) + code + "".join(lines[at:]))


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    skip = shutil.ignore_patterns("__pycache__", ".cache", "logs", "mlruns", ".pytest_cache")
    for name in COPY:
        shutil.copytree(ROOT / name, tmp_path / name, ignore=skip)
    shutil.copy(ROOT / ".gitignore", tmp_path / ".gitignore")
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "init")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_classify_names_the_error_and_what_is_missing():
    assert classify("ModuleNotFoundError: No module named 'sklearn.linear_model'") == \
        ("ModuleNotFoundError", "sklearn.linear_model")
    assert classify("ERROR: No matching distribution found for torch==9.9")[0] == "ResolutionImpossible"
    assert classify("FileNotFoundError: [Errno 2] No such file or directory: 'x.json'") == ("FileNotFoundError", "x.json")
    assert classify("make: *** [Makefile:30: train] Killed")[0] == "OutOfMemory"
    assert classify("something odd") == ("Error", None)
    assert classify("ERROR: Could not install packages due to an OSError: [Errno 2]")[0] == "BrokenEnvironment"


def test_fix_rule_installs_a_missing_module_once():
    class Layout:
        root, code_dir = str(ROOT), "baseline"

    class Plan:
        extra_packages = ["uvicorn"]
        layout = Layout
    plan = Plan()
    result = {"error_class": "ModuleNotFoundError", "detail": "sklearn.ensemble"}
    assert SetupAgent.fix_for(result, plan) == "install scikit-learn (the code imports sklearn)"
    assert SetupAgent.fix_for(result, plan) is None          # same fix twice: no rule left, give up
    assert SetupAgent.fix_for({"error_class": "FileNotFoundError", "detail": "x"}, plan) is None
    # a local package is never pip-installed (a PyPI package with the same name is someone else's code)
    assert SetupAgent.fix_for({"error_class": "ModuleNotFoundError", "detail": "shared.messages"}, plan) is None
    env = {}
    broken = {"error_class": "BrokenEnvironment", "detail": None}
    assert SetupAgent.fix_for(broken, plan, env).startswith("recreate") and env == {"fresh_venv": True}
    assert SetupAgent.fix_for(broken, plan, env) is None                     # only once


@needs_make
def test_first_run_commits_on_a_builder_branch_and_comes_back(repo):
    agent = SetupAgent()
    msg = agent.step()
    assert msg.kind == "pipeline.setup_completed", msg.summary
    assert msg.payload["branch"].startswith("builder/") and len(msg.payload["commit"]) == 40
    assert git(repo, "branch", "--show-current") == "main"                     # back where it started
    shown = git(repo, "show", "--name-only", "--format=", msg.payload["commit"]).split()
    own_schema = set() if (repo / "samples/schema.py").exists() else {"pipeline/schema.py"}   # else the team's is used
    assert set(shown) == {"baseline/Makefile", ".github/workflows/pipeline.yml",
                          "pipeline/dependency_map.yaml", "agents.yaml"} | own_schema
    assert agent.step() is None                                                 # nothing to do afterwards


@needs_make
def test_failure_posts_stage_error_class_and_log(repo):
    train = repo / "baseline" / "train.py"
    add_first(train, 'open("baseline/missing_config.json")\n')
    msg = SetupAgent(commit=False).step()
    assert msg.kind == "pipeline.setup_failed" and msg.to == ["human"]
    assert msg.payload["stage"] == "train" and msg.payload["error_class"] == "FileNotFoundError"
    assert (repo / msg.payload["log_ref"]).exists()


@needs_make
@pytest.mark.skipif(importlib.util.find_spec("emoji") is not None, reason="emoji is installed: no fault to fix")
def test_missing_module_is_fixed_then_verified(repo, tmp_path_factory, monkeypatch):
    """Needs pip to reach PyPI: the fix installs emoji in a dry-run venv of its own (empty at the start)."""
    monkeypatch.setenv("SETUP_AGENT_CACHE", str(tmp_path_factory.mktemp("cache")))
    train = repo / "baseline" / "train.py"
    add_first(train, 'import importlib; importlib.import_module("emoji")\n')
    agent = SetupAgent(commit=False)
    msg = agent.step()
    assert msg.kind == "pipeline.setup_completed", msg.summary
    assert msg.evidence["fixes"] == ["install emoji (the code imports emoji)"]
    assert agent.history[-1]["fixes"] == msg.evidence["fixes"]                    # remembered: it was verified


@needs_make
def test_rebuild_request_gets_accepted_then_a_result(repo):
    agent = SetupAgent(first_run=False, commit=False)
    req = Message(type=MessageType.REQUEST, kind="request.rebuild", sender="reviewer", topic="pipeline:setup",
                  level=1, summary="rebuild", incident_id="inc-1",
                  payload={"reason": "train.py changed", "changed_paths": ["baseline/train.py"]})
    answer = agent.handle(req)
    assert answer.payload["status"] == "accepted"
    msg = agent.step()
    assert msg.kind == "pipeline.setup_completed" and msg.reply_to == req.id and msg.incident_id == "inc-1"
    assert agent.handle(req) is None and agent.step() is None                   # idempotent


@needs_make
def test_rebuild_updates_only_the_agents_own_makefile(repo):
    agent = SetupAgent()
    first = agent.step()
    git(repo, "merge", "-q", first.payload["branch"])                       # the Reviewer merged the first setup
    (repo / "agents.yaml").write_text((repo / "agents.yaml").read_text() + "# a board edit\n")
    git(repo, "commit", "-qam", "board edits agents.yaml")
    train = repo / "baseline" / "train.py"
    add_first(train, "import importlib; importlib.import_module('emoji')\n")  # needs a new package
    git(repo, "commit", "-qam", "train.py changed")
    agent.handle(Message(type=MessageType.REQUEST, kind="request.rebuild", sender="reviewer", topic="pipeline:setup",
                         level=1, summary="rebuild", payload={"reason": "train.py changed",
                                                              "changed_paths": ["baseline/train.py"]}))
    msg = agent.step()
    assert msg.kind == "pipeline.setup_completed", msg.summary
    assert msg.evidence["updated"] == ["baseline/Makefile"]                  # its own file, with the fix in it
    assert "emoji" in git(repo, "show", f"{msg.payload['branch']}:baseline/Makefile")
    assert "# a board edit" in (repo / "agents.yaml").read_text()           # the board's file untouched
    assert git(repo, "branch", "--show-current") == "main"
