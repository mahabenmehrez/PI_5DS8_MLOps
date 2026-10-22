"""The Setup Agent's LLM step on a failed dry run, with a fake LLM and a fake dry run (fast, deterministic).

What is checked: the team menu (an action outside it is rejected and asked again), the regression guard, give_up,
no LLM configured, and the memory write-back of verified fixes (into a temp folder)."""
import json
from pathlib import Path

import pytest

from agents.setup.agent import SetupAgent
from agents.setup.builder_agent.llm.ollama import ChatResponse
from agents.setup.builder_agent.memory import FixMemory
from agents.setup.builder_agent.memory.learned import memory_docs

ROOT = Path(__file__).resolve().parents[3]
ERROR = "ImportError: cannot import name 'soft_unicode' from 'markupsafe'"


class FakeSearch:
    """Web search stand-in: canned pages, and the queries it was asked."""
    def __init__(self, results):
        self.results, self.queries = results, []

    def __call__(self, query):
        self.queries.append(query)
        return self.results, "fake web"


WEB = FakeSearch([{"title": "MarkupSafe 2.1.0 changelog", "url": "https://example.test/markupsafe-changes",
                   "content": "Version 2.1.0. Removed soft_unicode, which was previously deprecated. Use soft_str "
                              "instead. Pin markupsafe==2.0.1 if a dependency still imports soft_unicode."}])


class FakeLLM:
    model, options, digest = "fake", {}, "fake:1"

    def __init__(self, *answers):
        self.answers, self.seen = list(answers), []

    def chat(self, messages, schema=None):
        self.seen.append(messages)
        return ChatResponse(content=json.dumps(self.answers.pop(0)), latency_s=0.0)


def fail(stage, log):
    return {"ok": False, "stage": stage, "error_class": "ImportError", "detail": None, "log": log}


def fake_dry_run(results):
    """dry_run replacement: returns the result of the first rule whose test matches the Makefile."""
    def run(kind, inputs):
        make = inputs["files"]["baseline/Makefile"]
        for test, result in results:
            if test(make):
                return result
        raise AssertionError("no fake result")
    return run


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    if not (ROOT / "data" / "winequality-red.csv").exists():
        pytest.skip("needs data/winequality-red.csv")

    def make(client, results):
        a = SetupAgent(commit=False, client=client, memory=FixMemory.build(None, docs=memory_docs("base")),
                       memory_dir=str(tmp_path / "learned"), search=WEB)
        monkeypatch.setattr(a, "dry_run", fake_dry_run(results))
        return a
    return make


class _Report:
    created, updated, verified = [], [], {}


@pytest.fixture(autouse=True)
def no_writes(monkeypatch):
    """The real repo stays as it is: write_team is replaced (team mode's own writing is tested elsewhere)."""
    import agents.setup.builder_agent.team as team
    monkeypatch.setattr(team, "write_team", lambda plan, files, root=None, update_own=False: _Report())


def test_llm_answer_outside_the_team_menu_is_rejected_then_the_fix_is_verified(agent, tmp_path):
    llm = FakeLLM({"fix": {"action": "add_system_package", "name": "libffi-dev"}, "reason": "system lib"},
                  {"fix": {"action": "pin_package", "name": "markupsafe", "version": "2.0.1"},
                   "reason": "soft_unicode was removed in markupsafe 2.1"})
    a = agent(llm, [(lambda m: "markupsafe==2.0.1" in m, {"ok": True, "stage": None, "error_class": None, "log": ""}),
                    (lambda m: True, fail("train", ERROR))])
    msg = a.step()
    assert msg.kind == "pipeline.setup_completed", msg.summary
    assert msg.evidence["fixes"] == ["pin_package markupsafe==2.0.1"]
    d = msg.evidence["diagnoses"][0]
    assert d["calls"] == 2 and "does not match any of the expected tags" in d["rejected"][0][0]   # the team schema
    assert "ERROR OUTPUT" in llm.seen[0][1]["content"] and ERROR in llm.seen[0][1]["content"]
    learned = list((tmp_path / "learned").glob("*.json"))
    assert len(learned) == 1 and json.loads(learned[0].read_text())["outcome"] == "passed"
    assert any(d.fix == {"action": "pin_package", "name": "markupsafe", "version": "2.0.1"}
               for d in memory_docs("learned", tmp_path / "learned"))           # retrievable next time


def test_a_fix_that_breaks_an_earlier_stage_is_reverted(agent):
    llm = FakeLLM({"fix": {"action": "pin_package", "name": "markupsafe", "version": "9.9.9"}, "reason": "newest"},
                  {"fix": {"action": "pin_package", "name": "markupsafe", "version": "2.0.1"}, "reason": "pin"})
    a = agent(llm, [(lambda m: "markupsafe==2.0.1" in m, {"ok": True, "stage": None, "error_class": None, "log": ""}),
                    (lambda m: "markupsafe==9.9.9" in m,
                     fail("install", "ERROR: No matching distribution found for markupsafe==9.9.9")),
                    (lambda m: True, fail("train", ERROR))])
    msg = a.step()
    assert msg.kind == "pipeline.setup_completed", msg.summary
    assert msg.evidence["fixes"][0].startswith("pin_package markupsafe==9.9.9: reverted")
    assert msg.evidence["fixes"][1] == "pin_package markupsafe==2.0.1"
    assert "pin_package markupsafe==9.9.9" in llm.seen[1][1]["content"]       # told what was tried


def test_give_up_is_reported_with_the_llm_reason(agent):
    llm = FakeLLM({"fix": {"action": "give_up", "reason": "train.py (trainer) reads a missing file"}, "reason": "code"})
    a = agent(llm, [(lambda m: True, fail("train", "FileNotFoundError: 'baseline/x.json'"))])
    msg = a.step()
    assert msg.kind == "pipeline.setup_failed" and msg.payload["stage"] == "train"
    assert "give_up" in msg.evidence["why"] and "trainer" in msg.evidence["why"]


def test_without_an_llm_the_agent_reports(agent):
    a = agent(None, [(lambda m: True, fail("train", ERROR))])
    msg = a.step()
    assert msg.kind == "pipeline.setup_failed"
    assert msg.evidence["why"] == "no rule fixes this error and no LLM is configured"


def test_a_module_the_error_names_as_found_is_installed(agent):
    """check_llm.py on qwen2.5-coder:7b answered add_dependency markupsafe for this error; the validator now
    rejects it (markupsafe was found: the name is missing from its version) and the retry pins it."""
    from agents.setup.builder_agent.team.fix import present_in_error
    assert present_in_error(ERROR) == {"markupsafe"}
    llm = FakeLLM({"fix": {"action": "add_dependency", "name": "markupsafe"}, "reason": "missing"},
                  {"fix": {"action": "pin_package", "name": "markupsafe", "version": "2.0.1"}, "reason": "removed in 2.1"})
    a = agent(llm, [(lambda m: "markupsafe==2.0.1" in m, {"ok": True, "stage": None, "error_class": None, "log": ""}),
                    (lambda m: True, fail("train", ERROR))])
    msg = a.step()
    assert msg.evidence["fixes"] == ["pin_package markupsafe==2.0.1"]
    assert "already installed" in msg.evidence["diagnoses"][0]["rejected"][0][0]


def test_crag_discards_unrelated_memory_and_brings_web_knowledge_into_the_prompt(agent):
    """Last night's real case: plain RAG showed gen-006, gen-003, gen-002 for the markupsafe error. CRAG grades
    them irrelevant (Incorrect), discards them, rewrites the query and adds refined, unverified web knowledge."""
    llm = FakeLLM({"fix": {"action": "pin_package", "name": "markupsafe", "version": "2.0.1"},
                   "reason": "soft_unicode was removed in markupsafe 2.1.0"})
    a = agent(llm, [(lambda m: "markupsafe==2.0.1" in m, {"ok": True, "stage": None, "error_class": None, "log": ""}),
                    (lambda m: True, fail("train", ERROR))])
    msg = a.step()
    assert msg.evidence["fixes"] == ["pin_package markupsafe==2.0.1"]
    crag = msg.evidence["diagnoses"][0]["crag"]
    assert crag["action"] == "incorrect" and not any(g["kept"] for g in crag["graded"])
    assert crag["query"].startswith("python ImportError") and "markupsafe" in crag["query"]
    prompt = llm.seen[0][1]["content"]
    assert "Corrective RAG, incorrect" in prompt and "NOT verified" in prompt
    assert "Pin markupsafe==2.0.1" in prompt                                 # refined: the relevant web sentences
    assert "libffi" not in prompt and "xgboost" not in prompt                  # the discarded cases are gone
    # the verified fix is learned, and says where its knowledge came from
    [learned] = [json.loads(p.read_text()) for p in a.memory_dir.glob("*.json")]
    assert learned["outcome"] == "passed" and learned["fix"]["version"] == "2.0.1"
    src = learned["source"]
    assert src["fixed_by"] == "llm" and src["retrieval"]["action"] == "incorrect"
    assert src["retrieval"]["web_urls"] == ["https://example.test/markupsafe-changes"]
    assert src["retrieval"]["query"].startswith("python ImportError")


def test_every_llm_call_and_run_is_logged(agent):
    """Roadmap 13.1.6: every LLM call, action, check and outcome is logged."""
    llm = FakeLLM({"fix": {"action": "add_system_package", "name": "x"}, "reason": "r"},
                  {"fix": {"action": "pin_package", "name": "markupsafe", "version": "2.0.1"}, "reason": "removed"})
    a = agent(llm, [(lambda m: "markupsafe==2.0.1" in m, {"ok": True, "stage": None, "error_class": None, "log": ""}),
                    (lambda m: True, fail("train", ERROR))])
    msg = a.step()
    logs = ROOT / "agents/setup/logs"
    calls = [json.loads(line) for line in (logs / "llm_calls.jsonl").read_text().splitlines()]
    mine = [c for c in calls if c["message_id"] == msg.id]
    assert len(mine) == 2 and not mine[0]["valid"] and mine[1]["valid"]           # the rejected call is logged too
    assert mine[0]["messages"][0]["role"] == "system" and mine[1]["crag_action"] == "incorrect"
    runs = [json.loads(line) for line in (logs / "runs.jsonl").read_text().splitlines()]
    [run] = [r for r in runs if r["message_id"] == msg.id]
    assert run["fixes"] == ["pin_package markupsafe==2.0.1"] and run["attempts"][0]["outcome"] == "passed"
    assert run["checks"]["docker_compose_config"] in ("passes", "skipped (docker not installed here)")


def test_a_package_nobody_mentions_is_never_installed(agent):
    """Supply chain: the LLM (or a web page it read) can't make the agent pip-install an arbitrary name."""
    llm = FakeLLM({"fix": {"action": "add_dependency", "name": "totally-safe-helper"}, "reason": "install this"},
                  {"fix": {"action": "pin_package", "name": "markupsafe", "version": "2.0.1"}, "reason": "removed"})
    a = agent(llm, [(lambda m: "markupsafe==2.0.1" in m, {"ok": True, "stage": None, "error_class": None, "log": ""}),
                    (lambda m: True, fail("train", ERROR))])
    msg = a.step()
    assert msg.evidence["fixes"] == ["pin_package markupsafe==2.0.1"]
    assert "not named in the error output" in msg.evidence["diagnoses"][0]["rejected"][0][0]


# ---- the cascade (fine-tuned model first, the API model when it fails) and the slots fallback ----

class SecondLLM(FakeLLM):
    model = "api-model"


def test_cascade_the_second_model_answers_when_the_first_gives_up(agent):
    first = FakeLLM({"fix": {"action": "give_up", "reason": "no idea"}, "reason": "unsure"})
    second = SecondLLM({"fix": {"action": "pin_package", "name": "markupsafe", "version": "2.0.1"},
                        "reason": "soft_unicode was removed in markupsafe 2.1"})
    a = agent(first, [(lambda m: "markupsafe==2.0.1" in m, {"ok": True, "stage": None, "error_class": None, "log": ""}),
                      (lambda m: True, fail("train", ERROR))])
    a._escalation_client = second
    msg = a.step()
    assert msg.kind == "pipeline.setup_completed", msg.summary
    assert msg.evidence["fixes"] == ["pin_package markupsafe==2.0.1"]               # same attempt, budget unchanged
    ds = msg.evidence["diagnoses"]
    assert [d["model"] for d in ds] == ["fake", "api-model"] and ds[1]["escalated"] and not ds[0].get("escalated")


def test_cascade_takes_over_when_the_first_models_fix_changes_nothing(agent):
    first = FakeLLM({"fix": {"action": "pin_package", "name": "markupsafe", "version": "2.1.1"}, "reason": "older"})
    second = SecondLLM({"fix": {"action": "pin_package", "name": "markupsafe", "version": "2.0.1"}, "reason": "pin"})
    a = agent(first, [(lambda m: "markupsafe==2.0.1" in m, {"ok": True, "stage": None, "error_class": None, "log": ""}),
                      (lambda m: True, fail("train", ERROR))])
    a._escalation_client = second
    msg = a.step()
    assert msg.kind == "pipeline.setup_completed", msg.summary
    assert msg.evidence["fixes"] == ["pin_package markupsafe==2.1.1", "pin_package markupsafe==2.0.1"]
    assert [d["model"] for d in msg.evidence["diagnoses"]] == ["fake", "api-model"]
    assert len(first.seen) == 1                                                      # the first model was not asked again


def test_slots_fallback_goes_to_the_local_model_when_the_api_fails():
    from agents.setup.builder_agent.llm.api import ApiError, FallbackClient

    class Down:
        model = "api-model"

        def chat(self, messages, schema=None):
            raise TimeoutError("no answer in 60 s")
    local = FakeLLM({"target": "species"})
    f = FallbackClient(Down(), local)
    assert json.loads(f.chat([{"role": "user", "content": "?"}]).content) == {"target": "species"}
    assert f.last_model == "fake" and "TimeoutError" in f.failures[0]
    with pytest.raises(ApiError):
        FallbackClient(Down(), Down()).chat([])
