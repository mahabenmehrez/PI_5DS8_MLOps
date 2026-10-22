"""Team mode of the Setup Agent: complete the wine project with what is missing, never overwrite.

Every test works on a copy of the repository in a temp folder: nothing is written into the real project."""
import shutil
from pathlib import Path

import pytest
import yaml

from agents.setup.builder_agent.team import TeamLayout, render_team, team_plan, write_team

ROOT = Path(__file__).resolve().parents[3]
NEEDED = ["baseline", "samples", "data", "tests"]
LAYOUT = TeamLayout.from_contracts


@pytest.fixture
def repo(tmp_path, monkeypatch):
    if not (ROOT / "data" / "winequality-red.csv").exists():
        pytest.skip("data/winequality-red.csv is not here (data/ is not in git)")
    for name in NEEDED:
        shutil.copytree(ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__", "mlruns"))
    shutil.copy(ROOT / "requirements.txt", tmp_path / "requirements.txt")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_plan_reads_the_wine_project(repo):
    plan = team_plan(LAYOUT())
    assert plan.python_version == "3.10"
    assert plan.train_script == "baseline/train.py" and (plan.train_argv_data or plan.train_data_flag)
    assert plan.serve_target == "baseline.app.main:app"
    assert plan.separator == ";" and plan.target == "quality"
    assert len(plan.columns) == 11 and "pH" in [c.name for c in plan.columns]
    # the findings follow the project as it is (the team fixed the first ones): each one is checked against the files
    reqs = " ".join((repo / f).read_text() for f in ("baseline/requirements.txt", "baseline/app/requirements.txt"))
    kinds = {f.kind for f in plan.findings}
    assert ("scikit-learn" in plan.extra_packages) == ("scikit-learn" not in reqs)
    assert ("version_mismatch" in kinds) == ("mlflow==2.14.1" not in (repo / "baseline/app/requirements.txt").read_text())
    for mismatch in [f for f in plan.findings if f.kind == "version_mismatch"]:
        assert mismatch.path == "baseline/app/requirements.txt" and "mlflow==2.14.1" in mismatch.message


def test_generated_files_check_and_have_the_roadmap_targets(repo):
    files = render_team(team_plan(LAYOUT()))
    make = files["baseline/Makefile"]
    for target in ("install:", "train:", "serve:", "test:", "all:"):
        assert target in make
    # paths are relative to baseline/ (the root Makefile is the team runner)
    assert "-r requirements.txt -r app/requirements.txt" in make or "-r app/requirements.txt -r requirements.txt" in make
    assert ("$(PYTHON) train.py $(abspath $(RUN_DATA))" in make or "$(PYTHON) train.py --data $(abspath $(RUN_DATA))"
            in make) and "DATA ?= ../data/winequality-red.csv" in make
    assert "uvicorn app.main:app" in make
    assert "../samples/schema.py" in make and "pipeline/schema.py" not in files       # the team's schema, not a second one
    ci = yaml.safe_load(files[".github/workflows/pipeline.yml"])
    steps = " ".join(s.get("run", "") for s in ci["jobs"]["lint-and-test"]["steps"])
    assert "ruff check" in steps and "make -C baseline all SAMPLE=1" in steps
    assert "python -m pytest -q tests" in steps                       # the team's tests: CI, not make test
    assert "pytest" not in make.split("test: ")[1].split("all:")[0].replace("pip install pytest", "")
    depmap = yaml.safe_load(files["pipeline/dependency_map.yaml"])
    assert depmap["owners"]["model:wine-quality@champion"] == "trainer"
    assert {link["relation"] for link in depmap["links"]} == {"computes", "column_of", "trained_on", "serves"}
    cfg = yaml.safe_load(files["agents.yaml"])
    assert cfg["data"] == {"path": "data/winequality-red.csv", "separator": ";", "target": "quality"}
    assert set(cfg["agents"]) >= {"setup", "reviewer", "checker", "trainer", "deployer", "monitor"}


def test_generated_schema_accepts_the_data_and_catches_a_unit_error(repo):
    import pandas as pd
    (repo / "samples/schema.py").unlink()                  # no team schema: the agent writes its own
    files = render_team(team_plan(LAYOUT()))
    namespace = {"__name__": "schema"}
    exec(compile(files["pipeline/schema.py"], "schema.py", "exec"), namespace)
    df = pd.read_csv("data/winequality-red.csv", sep=";")
    namespace["validate"](df)
    bad = df.copy()
    bad["alcohol"] = bad["alcohol"] / 1000
    with pytest.raises(Exception):
        namespace["validate"](bad)


def test_write_never_overwrites(repo):
    plan = team_plan(LAYOUT())
    files = render_team(plan)
    (repo / "baseline" / "Makefile").write_text("# the team's own Makefile\n")
    report = write_team(plan, files)
    assert (repo / "baseline" / "Makefile").read_text() == "# the team's own Makefile\n"
    assert report.verified["baseline/Makefile"].startswith("differs")
    assert "agents.yaml" in report.created
    again = write_team(plan, files)
    assert again.created == [] and again.verified["agents.yaml"] == "same"


def test_a_fresh_clone_downloads_the_data(tmp_path):
    """Real-project check (2026-10-21): data/ is not in git, and the agent crashed on a fresh clone."""
    from agents.setup.builder_agent.team import TeamLayout, ensure_data
    src = tmp_path / "upstream.csv"
    src.write_text("a;quality\n1;5\n")
    layout = TeamLayout(root=str(tmp_path / "repo"), code_dir=".", data_path="data/wine.csv",
                        data_url=src.as_uri(), makefile="Makefile", ci="ci.yml", schema_file="s.py",
                        depmap="d.yaml", agents_yaml="agents.yaml", tests_dir="tests", config_sample="samples/agents.yaml")
    (tmp_path / "repo").mkdir()
    assert "downloaded data/wine.csv" in ensure_data(layout)
    assert (tmp_path / "repo/data/wine.csv").read_text() == "a;quality\n1;5\n"
    assert ensure_data(layout) is None                          # there now: nothing to do
    bad = TeamLayout(**{**layout.model_dump(), "data_path": "data/other.csv", "data_url": (tmp_path / "no.csv").as_uri()})
    import pytest
    with pytest.raises(FileNotFoundError, match="download from"):
        ensure_data(bad)
    assert not list((tmp_path / "repo/data").glob("*.part"))   # no half-written file left


def test_a_fresh_clone_is_still_the_team_layout(tmp_path):
    """Real-project check: on a fresh clone (no data/), detect() fell back to the small-repo layout and crashed."""
    from agents.setup.builder_agent.team import TeamLayout
    (tmp_path / "baseline").mkdir()
    layout = TeamLayout.detect(tmp_path)
    assert layout.code_dir == "baseline" and layout.data_url.startswith("https://")


def test_every_finding_on_the_wine_repo_names_its_owner():
    """The board routes a finding to the agent whose file it is: train code -> trainer, the app -> deployer."""
    from pathlib import Path

    import pytest
    from agents.setup.builder_agent.team import TeamLayout, _owner_of, team_plan
    assert _owner_of("baseline/train.py") == "trainer"
    assert _owner_of("baseline/app/Dockerfile") == "deployer" and _owner_of("baseline/app/main.py") == "deployer"
    assert _owner_of("shared/util.py") is None
    root = Path(__file__).resolve().parents[3]
    if not (root / "data/winequality-red.csv").is_file():
        pytest.skip("the wine data is not here")
    assert all(f.owner for f in team_plan(TeamLayout.detect(str(root))).findings)


def test_make_test_validates_with_the_teams_schema(repo):
    """The Data Checker owns samples/schema.py: make test uses it (any module with a pandera DataFrameSchema)."""
    import subprocess
    import sys
    plan = team_plan(LAYOUT())
    assert plan.team_schema == "samples/schema.py" and plan.modes.get("pipeline/schema.py") is None
    line = render_team(plan)["baseline/Makefile"].split("test: tools data $(SAMPLE_DEP)\n")[1].splitlines()[0]
    code = line.split('-c "', 1)[1].rsplit('" ', 1)[0]
    ok = subprocess.run([sys.executable, "-c", code, "samples/schema.py", "data/winequality-red.csv"],
                        capture_output=True, text=True)
    assert ok.returncode == 0 and "passes the team schema" in ok.stdout, ok.stderr[-400:]
    bad = repo / "bad.csv"
    bad.write_text((repo / "data/winequality-red.csv").read_text().replace(";9.4;5", ";-9.4;5", 1))
    assert subprocess.run([sys.executable, "-c", code, "samples/schema.py", str(bad)], capture_output=True).returncode


def test_a_data_option_gets_the_data_path(tmp_path):
    """Real-project check: the team's train.py moved from sys.argv[1] to argparse --data."""
    from agents.setup.builder_agent.team import data_flag
    from agents.setup.builder_agent.scan.models import CliArg, TrainScript
    script = TrainScript(path="train.py", has_main_guard=True, cli="argparse",
                         cli_args=[CliArg(flags=["--alias"], required=False), CliArg(flags=["--data"], required=False)])
    assert data_flag(script) == "--data"
    assert data_flag(TrainScript(path="t.py", has_main_guard=True, cli_args=[CliArg(flags=["--alpha"], required=False)])) is None
