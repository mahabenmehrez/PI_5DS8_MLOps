"""v2 of the team diagnosis (after the v1 evaluation): the validator rejects the answers that wasted v1's attempts,
the prompt shows real versions, and the path rule looks in the script's parent folders."""
from types import SimpleNamespace

import pytest

from agents.setup.builder_agent.fix.models import AddDependency, PinPackage
from agents.setup.builder_agent.team import fix as tf
from agents.setup.builder_agent.team.fix import ChangeEntryCommand, FixPath, team_checks

SKLEARN_LOG = """Requirement already satisfied: scikit-learn in /venv/lib/python3.10/site-packages (1.7.2)
Requirement already satisfied: numpy in /venv/lib/python3.10/site-packages (2.2.6)
Traceback (most recent call last):
  File "/tmp/repo/train.py", line 2, in <module>
    _mse([1.0, 2.0], [1.0, 2.5], squared=False)
  File "/venv/lib/python3.10/site-packages/sklearn/utils/_param_validation.py", line 196, in wrapper
TypeError: got an unexpected keyword argument 'squared'
make: *** [Makefile:20: train] Error 1"""
INSTALLED = {"scikit-learn": "1.7.2", "numpy": "2.2.6", "pandas": "2.3.3", "mlflow": "2.14.1", "pip": "24.0"}
RELEASES = {"scikit-learn": ["1.4.2", "1.5.0", "1.5.2", "1.6.0", "1.6.1", "1.7.0", "1.7.1", "1.7.2"]}


PYPI_RELEASES = tf.pypi_releases          # the real function (the fixtures below replace it)


@pytest.fixture(autouse=True)
def offline_pypi(monkeypatch):
    monkeypatch.setattr(tf, "pypi_releases", lambda name, timeout=10, python=None: RELEASES.get(tf.normalize(name)))


def plan(tmp_path, **kw):
    (tmp_path / "requirements.txt").write_text("scikit-learn\nnumpy\npandas\nmlflow\n")
    (tmp_path / "baseline").mkdir(exist_ok=True)
    (tmp_path / "data").mkdir(exist_ok=True)
    layout = SimpleNamespace(root=str(tmp_path), makefile="baseline/Makefile", code_dir="baseline")
    base = dict(layout=layout, requirements=["requirements.txt"], extra_packages=[], train_cwd=None,
                train_style="script", train_script="baseline/train.py")
    return SimpleNamespace(**{**base, **kw})


def result(log=SKLEARN_LOG, installed=INSTALLED, hint=None):
    return {"log": log, "installed": installed, "import_hint": hint}


def test_a_pin_of_the_version_that_failed_is_rejected(tmp_path):
    check = team_checks(plan(tmp_path), SKLEARN_LOG, result())
    assert "already used" in check(PinPackage(action="pin_package", name="scikit-learn", version="1.7.2"))[0]
    assert check(PinPackage(action="pin_package", name="scikit-learn", version="1.5.2")) == []


def test_a_version_that_does_not_exist_is_rejected_with_the_real_ones(tmp_path):
    reasons = team_checks(plan(tmp_path), SKLEARN_LOG, result())(
        PinPackage(action="pin_package", name="scikit-learn", version="1.6.2"))
    assert "does not exist on PyPI" in reasons[0] and "1.6.1" in reasons[0]


def test_pypi_unreachable_skips_the_check(tmp_path, monkeypatch):
    monkeypatch.setattr(tf, "pypi_releases", lambda name, timeout=10, python=None: None)
    assert team_checks(plan(tmp_path), SKLEARN_LOG, result())(
        PinPackage(action="pin_package", name="scikit-learn", version="1.6.2")) == []


def test_a_pin_of_a_package_the_error_does_not_name_is_rejected(tmp_path):
    """v1: llm_memory pinned mlflow and numpy for a scikit-learn error (pip's own lines name every package)."""
    check = team_checks(plan(tmp_path), SKLEARN_LOG, result())
    assert "not named in the error" in check(PinPackage(action="pin_package", name="mlflow", version="2.9.2"))[0]
    assert "not named in the error" in check(PinPackage(action="pin_package", name="numpy", version="1.23.5"))[0]


def test_a_fix_already_tried_is_rejected(tmp_path):
    tried = [{"action": "pin_package", "name": "scikit_learn", "version": "1.5.2"},
             {"action": "fix_path", "target": "workdir", "path": "data/"}]
    check = team_checks(plan(tmp_path), SKLEARN_LOG, result(), tried)
    assert "already tried" in check(PinPackage(action="pin_package", name="scikit-learn", version="1.5.2"))[0]
    assert "already tried" in check(FixPath(action="fix_path", target="workdir", path="data"))[0]
    assert check(PinPackage(action="pin_package", name="scikit-learn", version="1.4.2")) == []


def test_fix_path_to_the_folder_train_already_runs_in_is_rejected(tmp_path):
    """v1: fix_path baseline was accepted although the Makefile already runs train in baseline/."""
    check = team_checks(plan(tmp_path), SKLEARN_LOG, result())
    assert "already runs in 'baseline'" in check(FixPath(action="fix_path", target="workdir", path="baseline"))[0]
    assert check(FixPath(action="fix_path", target="workdir", path=".")) == []
    module = team_checks(plan(tmp_path, train_style="module"), SKLEARN_LOG, result())
    assert "already runs in '.'" in module(FixPath(action="fix_path", target="workdir", path="."))[0]


def test_a_misspelt_name_is_not_a_version_problem(tmp_path):
    log = "ImportError: cannot import name 'LogisticRegresion' from 'sklearn.linear_model'"
    hint = {"name": "LogisticRegresion", "module": "sklearn.linear_model", "close": ["LogisticRegression"]}
    reasons = team_checks(plan(tmp_path), log, result(log, hint=hint))(
        PinPackage(action="pin_package", name="scikit-learn", version="1.5.2"))
    assert "misspells" in reasons[0] and "give_up" in reasons[0]


def test_add_dependency_of_an_installed_package_is_rejected(tmp_path):
    reasons = team_checks(plan(tmp_path), SKLEARN_LOG, result())(AddDependency(action="add_dependency", name="numpy"))
    assert "already installed" in reasons[0]


def test_versions_text_names_only_the_packages_of_the_error():
    text = tf.versions_text(result())
    assert "scikit-learn: installed 1.7.2" in text and "1.7.2, 1.6.1, 1.5.2, 1.4.2" in text
    assert "numpy" not in text and "mlflow" not in text           # only in pip's lines
    hint = {"name": "X", "module": "m", "close": ["Y"]}
    assert "misspells" in tf.versions_text({"log": "", "installed": {}, "import_hint": hint})


def test_path_rule_looks_in_the_parent_folders(tmp_path):
    """v1: the wine script reads data/... from the root while the Makefile runs it in baseline/."""
    from agents.setup.agent import SetupAgent
    p = plan(tmp_path)
    (tmp_path / "data" / "wine.csv").write_text("a\n1\n")
    r = {"ok": False, "stage": "train", "error_class": "FileNotFoundError", "detail": "data/wine.csv", "log": ""}
    env = {}
    text = SetupAgent.fix_for(r, p, env)
    assert p.train_cwd == "." and "above the script" in text
    assert env["action"] == {"action": "fix_path", "target": "workdir", "path": "."}
    assert SetupAgent.fix_for(r, p, {}) is None                   # runs from there now: nothing more to try


def test_path_rule_prefers_the_script_folder(tmp_path):
    p = plan(tmp_path, train_script="model/train.py")
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "iris.csv").write_text("a\n")
    r = {"ok": False, "stage": "train", "error_class": "FileNotFoundError", "detail": "iris.csv", "log": ""}
    assert "next to the script" in SetupAgent_fix(r, p) and p.train_cwd == "model"


def SetupAgent_fix(r, p):
    from agents.setup.agent import SetupAgent
    return SetupAgent.fix_for(r, p, {})


def test_change_entry_command_still_checked(tmp_path):
    check = team_checks(plan(tmp_path), SKLEARN_LOG, result())
    assert "already runs as a script" in check(ChangeEntryCommand(action="change_entry_command", stage="train",
                                                                  style="script"))[0]


# ---- v3 (after the v2 evaluation) ----

def missing(detail):
    return {"log": "", "installed": INSTALLED, "error_class": "FileNotFoundError", "detail": detail}


def test_where_text_finds_the_folder_that_opens_the_file(tmp_path):
    p = plan(tmp_path)
    (tmp_path / "data" / "iris.csv").write_text("a\n")
    text = tf.where_text(p, missing("iris.csv"))
    assert "from baseline/" in text and "found at data/iris.csv: run from data/" in text
    assert "no file named" in tf.where_text(p, missing("data/iris_v2.csv"))
    assert tf.where_text(p, {"error_class": "TypeError"}) == "the error is not a missing file"


def test_a_package_never_fixes_a_missing_file(tmp_path):
    """v2: for a FileNotFoundError the LLM pinned pandas (the traceback goes through pandas)."""
    log = 'File "/venv/site-packages/pandas/io/common.py"\nFileNotFoundError: [Errno 2] No such file: \'iris.csv\''
    p = plan(tmp_path)
    (tmp_path / "data" / "iris.csv").write_text("a" + chr(10))
    check = team_checks(p, log, {**missing("iris.csv"), "log": log})
    assert "missing file" in check(PinPackage(action="pin_package", name="pandas", version="2.1.4"))[0]
    assert check(FixPath(action="fix_path", target="workdir", path="data")) == []


def test_a_missing_file_inside_a_package_is_not_a_path_problem():
    assert tf.missing_file({"error_class": "FileNotFoundError", "detail": "/venv/lib/site-packages/x/data.json"}) is None


def test_the_agents_check_tools_fall_back_to_an_older_pandera():
    """v2: the newest pandera (the agent's schema check) refused numpy 1.23.5 at import and failed `make test`."""
    from pathlib import Path

    from agents.setup.builder_agent.team import TeamLayout, render_team, team_plan
    repo = Path(__file__).resolve().parents[1] / "test_repos" / "sk_root"
    plan = team_plan(TeamLayout.detect(str(repo)))
    makefile = render_team(plan)[plan.layout.makefile]
    tools = makefile.split("tools:", 1)[1].split("\n\n", 1)[0]
    assert "pa.DataFrameSchema" in tools and 'pip install "pandera==$(PANDERA_FALLBACK)"' in tools


# ---- v4 (after the v3 evaluation) ----

def test_only_versions_that_install_on_the_builds_python(tmp_path, monkeypatch):
    """v3: the LLM pinned scipy 1.18.1 / 1.17.1 / 1.16.3, which need Python 3.11, on a Python 3.10 build."""
    raw = [("1.13.1", [">=3.9"]), ("1.15.3", [">=3.10"]), ("1.16.3", [">=3.11"])]
    monkeypatch.setattr(tf, "_RELEASES", {"scipy": raw})
    monkeypatch.setattr(tf, "pypi_releases", PYPI_RELEASES)
    assert tf.pypi_releases("scipy") == ["1.13.1", "1.15.3", "1.16.3"]
    assert tf.pypi_releases("scipy", python="3.10.14") == ["1.13.1", "1.15.3"]
    log = 'File "/venv/site-packages/scipy/integrate/__init__.py"\nImportError: cannot import name \'simps\''
    r = {"log": log, "installed": {"scipy": "1.15.3"}, "python": "3.10.14"}
    check = team_checks(plan(tmp_path), log, r)
    assert "does not install on Python 3.10.14" in check(PinPackage(action="pin_package", name="scipy",
                                                                    version="1.16.3"))[0]
    assert check(PinPackage(action="pin_package", name="scipy", version="1.13.1")) == []
    assert "that install on Python 3.10.14" in tf.versions_text(r) and "1.16.3" not in tf.versions_text(r)


def test_fix_path_must_make_the_missing_file_exist(tmp_path):
    """v3: fix_path data was tried although no file of that name is in the repository."""
    p = plan(tmp_path)
    (tmp_path / "data" / "iris.csv").write_text("a\n")
    assert team_checks(p, "", missing("iris.csv"))(FixPath(action="fix_path", target="workdir", path="data")) == []
    reasons = team_checks(p, "", missing("iris_v2.csv"))(FixPath(action="fix_path", target="workdir", path="data"))
    assert "still doesn't exist" in reasons[0]
