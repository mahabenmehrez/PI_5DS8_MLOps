"""Roadmap 7.1 MVP and 7.6: the Setup Agent on the two tiny scikit-learn repos (and a third layout), real dry runs,
built in at most 3 attempts. Each test works on a copy: test_repos/ stays as it is."""
import shutil
from pathlib import Path

import pytest

from agents.setup.agent import SetupAgent
from agents.setup.builder_agent.team import TeamLayout, team_plan
from agents.setup.builder_agent.team.fix import ChangeEntryCommand, FixPath, team_checks

REPOS = Path(__file__).resolve().parents[1] / "test_repos"
needs_make = pytest.mark.skipif(shutil.which("make") is None, reason="needs make")


def copy(tmp_path, name):
    shutil.copytree(REPOS / name, tmp_path / name)
    return tmp_path / name


def sk_dir(tmp_path):
    """A script that opens its data relative to its own folder (fix_path)."""
    repo = tmp_path / "sk_dir"
    (repo / "model").mkdir(parents=True)
    (repo / "data").mkdir()
    for d in ("data", "model"):
        shutil.copy(REPOS / "sk_root/data/iris.csv", repo / d / "iris.csv")
    (repo / "model/train.py").write_text((REPOS / "sk_root/train.py").read_text().replace("data/iris.csv", "iris.csv"))
    (repo / "requirements.txt").write_text("pandas\nscikit-learn\njoblib\n")
    return repo


def test_layout_of_a_small_repo(tmp_path):
    lay = TeamLayout.detect(copy(tmp_path, "sk_src"))
    assert lay.code_dir == "." and lay.makefile == "Makefile" and lay.data_path == "data/iris.csv"
    plan = team_plan(lay)
    assert plan.train_script == "src/irisclf/train.py" and plan.target == "species"
    assert set(plan.extra_packages) == {"pandas", "scikit-learn", "pyyaml"}         # no requirements file
    assert not plan.has_tests and "no_model_output" not in {f.kind for f in plan.findings}  # pickle.dumps counts
    root_plan = team_plan(TeamLayout.detect(copy(tmp_path, "sk_root")))
    assert root_plan.train_script == "train.py" and root_plan.requirements == ["requirements.txt"]


@needs_make
@pytest.mark.parametrize("make_repo, fixes", [
    (lambda t: copy(t, "sk_root"), []),
    (lambda t: copy(t, "sk_src"), ["run train as a module (python -m irisclf.train): it uses relative imports"]),
    (sk_dir, ["run train from model/ (iris.csv is there, next to the script)"]),
])
def test_each_repo_builds_in_at_most_three_attempts(tmp_path, make_repo, fixes):
    repo = make_repo(tmp_path)
    agent = SetupAgent(root=str(repo), commit=False, client=None, crag=False, memory_dir=str(tmp_path / "mem"))
    msg = agent.step()
    assert msg.kind == "pipeline.setup_completed", msg.summary
    assert msg.evidence["fixes"] == fixes and len(fixes) <= 3
    assert (repo / "Makefile").exists() and (repo / "pipeline/schema.py").exists()
    if fixes:                                                    # the verified fix is in the memory, as an action
        learned = list((tmp_path / "mem").glob("*.json"))
        assert len(learned) == 1 and "passed" in learned[0].read_text()


def test_team_checks_reject_impossible_path_and_entry_fixes(tmp_path):
    plan = team_plan(TeamLayout.detect(copy(tmp_path, "sk_src")))
    check = team_checks(plan)
    assert check(FixPath(action="fix_path", target="workdir", path="nope")) == \
        ["fix_path: 'nope' is not a folder of the repository"]
    assert check(FixPath(action="fix_path", target="workdir", path="../..")) != []
    assert check(FixPath(action="fix_path", target="workdir", path="src")) == []
    assert check(ChangeEntryCommand(action="change_entry_command", stage="train", style="module")) == []
    assert "already runs as a script" in check(ChangeEntryCommand(action="change_entry_command", stage="train",
                                                                  style="script"))[0]
    flat = team_plan(TeamLayout.detect(copy(tmp_path, "sk_root")))
    assert "not inside a package" in team_checks(flat)(ChangeEntryCommand(action="change_entry_command",
                                                                          stage="train", style="module"))[0]
