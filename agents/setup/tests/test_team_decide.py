"""The decide step in team mode: rules decide when the evidence is clear, the LLM only for needs_llm choices,
answers validated against the candidates, and no silent guess when there is no LLM."""
import json

import pytest

from agents.setup.builder_agent.llm.ollama import ChatResponse
from agents.setup.builder_agent.team import TeamLayout, resolve_with_llm, team_plan

TRAIN = """import pandas as pd
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    df = pd.read_csv("data/sales.csv")
    y = df["{target}"]
    X = df.drop(columns=["{target}", "{other}"])
    LinearRegression().fit(X, y)
"""


@pytest.fixture
def ambiguous(tmp_path):
    """Two scripts train a model, each on a different column: the rules can't tell which is the pipeline's."""
    repo = tmp_path / "shop"
    (repo / "data").mkdir(parents=True)
    rows = ["price,ads,visits,revenue,margin"] + [f"{i},{i % 7},{i * 3},{i * 10},{i % 5}" for i in range(1, 60)]
    (repo / "data/sales.csv").write_text("\n".join(rows) + "\n")
    (repo / "train_revenue.py").write_text(TRAIN.format(target="revenue", other="margin"))
    (repo / "train_margin.py").write_text(TRAIN.format(target="margin", other="revenue"))
    (repo / "requirements.txt").write_text("pandas\nscikit-learn\n")
    return repo


class LLM:
    model, options, digest = "fake", {}, "fake:1"

    def __init__(self, *answers):
        self.answers, self.seen = list(answers), []

    def chat(self, messages, schema=None):
        self.seen.append((messages, schema))
        return ChatResponse(content=json.dumps(self.answers.pop(0)), latency_s=0.0)


def by_item(plan):
    return {d["item"]: d for d in plan.decisions}


def test_clear_evidence_is_decided_by_rules_without_the_llm():
    for repo in (".", "agents/setup/test_repos/sk_root", "agents/setup/test_repos/sk_src"):
        plan = team_plan(TeamLayout.detect(repo))
        assert all(d["status"] == "decided" for d in plan.decisions), plan.decisions
        assert by_item(plan)["python"]["value"] == "3.10"                     # the team's Python when none declared
        new, log = resolve_with_llm(plan, LLM())
        assert log == [] and new is plan                                      # no call when nothing needs it


def test_unclear_choices_are_marked_needs_llm_with_candidates(ambiguous):
    d = by_item(team_plan(TeamLayout.detect(ambiguous)))
    assert d["train_script"]["status"] == "needs_llm"
    assert d["train_script"]["candidates"] == ["train_margin.py", "train_revenue.py"]
    assert d["target"]["status"] == "needs_llm" and {"revenue", "margin"} <= set(d["target"]["candidates"])


def test_the_llm_answer_is_validated_then_used(ambiguous):
    plan = team_plan(TeamLayout.detect(ambiguous))
    llm = LLM({"train_script": "train.py", "target": "profit"},                 # off the lists: rejected
              {"train_script": "train_revenue.py", "target": "revenue"})
    new, log = resolve_with_llm(plan, llm)
    assert [c["valid"] for c in log] == [False, True] and "not one of" in log[0]["reasons"][0]
    d = by_item(new)
    assert d["train_script"]["value"] == "train_revenue.py" and d["train_script"]["source"] == "llm"
    assert new.train_script == "train_revenue.py" and new.target == "revenue"
    assert "revenue" not in [c.name for c in new.columns]                    # the schema follows the target
    schema = llm.seen[0][1]
    assert schema["properties"]["train_script"]["enum"] == ["train_margin.py", "train_revenue.py"]


def test_without_an_llm_the_guess_is_reported_not_silent(ambiguous):
    new, log = resolve_with_llm(team_plan(TeamLayout.detect(ambiguous)), None)
    assert log == []
    assert {d["source"] for d in new.decisions if d["status"] == "needs_llm"} == {"assumed"}
    assumed = [f for f in new.findings if f.kind == "decision_assumed"]
    assert {f.message.split(":")[0] for f in assumed} == {"train_script", "target"}


@pytest.mark.skipif(__import__("shutil").which("make") is None, reason="needs make")
def test_the_agent_asks_the_llm_then_builds(ambiguous, tmp_path):
    from agents.setup.agent import SetupAgent
    llm = LLM({"train_script": "train_revenue.py", "target": "revenue"})
    agent = SetupAgent(root=str(ambiguous), commit=False, client=llm, crag=False, memory_dir=str(tmp_path / "m"))
    msg = agent.step()
    assert msg.kind == "pipeline.setup_completed", msg.summary
    decided = {d["item"]: d for d in msg.evidence["decisions"]}
    assert decided["train_script"]["source"] == "llm" and decided["target"]["value"] == "revenue"
    assert "train_revenue.py" in (ambiguous / "Makefile").read_text()
    calls = [json.loads(line) for line in (ambiguous / "agents/setup/logs/llm_calls.jsonl").read_text().splitlines()]
    assert [c["step"] for c in calls] == ["decide"] and calls[0]["valid"]


TOP = """import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/sales.csv")
X, y = df.drop(columns=["revenue"]), df["revenue"]
model = LinearRegression().fit(X, y)
{save}
"""


def _top_level_repo(tmp_path, save_a: bool, save_b: bool):
    repo = tmp_path / "top"
    (repo / "data").mkdir(parents=True)
    (repo / "data/sales.csv").write_text("price,ads,revenue\n" + "".join(f"{i},{i % 3},{i * 2}\n" for i in range(30)))
    (repo / "train_a.py").write_text(TOP.format(save='joblib.dump(model, "a.joblib")' if save_a else "print(1)"))
    (repo / "train_b.py").write_text(TOP.format(save='joblib.dump(model, "b.joblib")' if save_b else "print(1)"))
    (repo / "requirements.txt").write_text("pandas\nscikit-learn\njoblib\n")
    return team_plan(TeamLayout.for_repo(str(repo)))


def _decision(plan, item):
    return next(d for d in plan.decisions if d["item"] == item)


def test_several_top_level_scripts_are_not_called_the_only_one(tmp_path):
    """v1 bug: the first top-level train*.py was taken and reported as 'the only script that trains a model'."""
    d = _decision(_top_level_repo(tmp_path, True, True), "train_script")
    assert d["status"] == "needs_llm" and d["candidates"] == ["train_a.py", "train_b.py"]
    assert all("saves a model" in f for f in d["facts"])


def test_the_only_script_that_saves_a_model_is_decided(tmp_path):
    d = _decision(_top_level_repo(tmp_path, False, True), "train_script")
    assert d["status"] == "decided" and d["value"] == "train_b.py" and "only train_b.py saves" in d["reason"]


def test_target_facts_show_the_training_scripts_lines(ambiguous):
    d = _decision(team_plan(TeamLayout.detect(ambiguous)), "target")     # train script: the guess, train_margin.py
    assert any(f.startswith("margin in train_margin.py") and 'y = df["margin"]' in f for f in d["facts"])
