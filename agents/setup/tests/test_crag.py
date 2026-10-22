"""Corrective RAG (memory/crag.py): evaluator, the three actions, knowledge refinement, query rewriting,
web search cache. Offline: the base memory without embeddings (BM25), canned web pages."""
import json

import pytest

from agents.setup.builder_agent.memory import FixMemory
from agents.setup.builder_agent.memory.crag import (CorrectiveRAG, LlmEvaluator, TavilySearch, entities,
                                                    refine_web, relevance, rewrite_query)
from agents.setup.builder_agent.memory.learned import memory_docs
from agents.setup.builder_agent.llm.ollama import ChatResponse

MARKUPSAFE = ["  from markupsafe import escape, soft_unicode",
              "ImportError: cannot import name 'soft_unicode' from 'markupsafe' "
              "(/usr/lib/python3.10/site-packages/markupsafe/__init__.py)"]


class Web:
    def __init__(self, results=()):
        self.results, self.queries = list(results), []

    def __call__(self, q):
        self.queries.append(q)
        return self.results, "fake"


@pytest.fixture(scope="module")
def memory():
    return FixMemory.build(None, docs=memory_docs("base"))


@pytest.mark.parametrize("tail, action, kept", [
    (MARKUPSAFE, "incorrect", []),                                                         # nothing in memory
    (["ModuleNotFoundError: No module named 'sklearn'"], "ambiguous", ["gen-002"]),         # same kind, other package
    (["ModuleNotFoundError: No module named 'xgboost'"], "correct", ["gen-002"]),
    (["ImportError: libffi.so.8: cannot open shared object file: No such file or directory"], "correct", ["gen-006"]),
    (["ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, "
      "got 88 from PyObject"], "correct", ["gen-003"]),
    (["mlflow.exceptions.MlflowException: API request to endpoint /api/2.0/mlflow/logged-models failed with "
      "error code 404 != 200"], "correct", ["gen-007", "gen-008"]),
])
def test_the_evaluator_picks_the_action(memory, tail, action, kept):
    web = Web()
    r = CorrectiveRAG(memory, search=web).retrieve("train", tail)
    assert r.action == action, [(g.id, g.score) for g in r.graded]
    assert sorted(g.id for g in r.kept()) == kept
    assert bool(web.queries) == (action != "correct")                     # the web only when memory isn't enough


def test_entities_ignore_keywords_and_find_unquoted_names():
    assert entities("ModuleNotFoundError: No module named 'sklearn'") == {"sklearn"}
    assert {"mlflow"} <= entities("mlflow.exceptions.MlflowException: API request failed")
    assert "libffi" in entities("ImportError: libffi.so.8: cannot open shared object file")
    assert relevance("No module named 'sklearn'", "No module named 'xgboost'")[1]["shared_names"] == []


def test_query_rewriting_removes_paths_and_noise():
    q = rewrite_query(MARKUPSAFE)
    assert q == "python ImportError: cannot import name 'soft_unicode' from 'markupsafe' fix"


def test_refinement_keeps_relevant_sentences_in_page_order():
    pages = [{"url": "u1", "content": "Welcome to our blog about cooking pasta with fresh tomatoes. "
                                      "MarkupSafe 2.1.0 removed soft_unicode, use soft_str instead."},
             {"url": "u2", "content": "Subscribe to the newsletter for weekly recipes and kitchen tips. "
                                      "Pin markupsafe==2.0.1 when an old Jinja2 imports soft_unicode."}]
    strips = refine_web(pages, MARKUPSAFE[-1])
    assert [s["url"] for s in strips] == ["u1", "u2"]
    assert all("soft_unicode" in s["text"] for s in strips)                 # the cooking sentences are dropped


def test_render_separates_verified_memory_from_unverified_web(memory):
    r = CorrectiveRAG(memory, search=Web([{"url": "https://x.test", "content":
        "The sklearn package is installed with pip install scikit-learn, not pip install sklearn."}])).retrieve(
        "train", ["ModuleNotFoundError: No module named 'sklearn'"])
    text = r.render()
    assert "our build verified" in text and "verified fix: add_dependency xgboost" in text
    assert "NOT verified" in text and "scikit-learn" in text
    assert "cause:" not in text                     # gen-002's cause names xgboost, not sklearn: dropped


def test_web_search_cache_replay_and_off(tmp_path, monkeypatch):
    calls = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"results": [{"title": "t", "url": "https://x.test", "content": "c"}]}).encode()

    def urlopen(req, timeout):
        calls.append(json.loads(req.data))
        assert req.headers["Authorization"] == "Bearer tvly-test"
        return Resp()
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    live = TavilySearch(mode="cache", cache_dir=tmp_path)
    assert live("q1") == ([{"title": "t", "url": "https://x.test", "content": "c"}], "web")
    assert live("q1")[1] == "cache" and len(calls) == 1                     # a second ask is not a second search
    assert TavilySearch(mode="replay", cache_dir=tmp_path)("q1")[1] == "cache"
    assert TavilySearch(mode="replay", cache_dir=tmp_path)("q2") == ([], "not in the replay cache")
    assert TavilySearch(mode="off", cache_dir=tmp_path)("q1") == ([], "web search off")
    assert len(calls) == 1


def test_llm_evaluator_option(memory):
    class Grader:
        def chat(self, messages, schema=None):
            relevant = "yes" if "xgboost" in messages[1]["content"].split("PAST FAILURE")[1] else "no"
            return ChatResponse(content=json.dumps({"relevant": relevant, "why": "test"}), latency_s=0)
    r = CorrectiveRAG(memory, search=Web(), evaluator=LlmEvaluator(Grader())).retrieve(
        "train", ["ModuleNotFoundError: No module named 'xgboost'"])
    assert r.action == "correct" and [g.id for g in r.kept()] == ["gen-002"]


def test_refinement_drops_echoes_of_the_error_and_keeps_explanations():
    """From the first real Tavily search (2026-10-17): five of six kept sentences only repeated the error."""
    page = {"url": "u", "content": "from markupsafe import soft_unicode\n"
            "ImportError: cannot import name 'soft_unicode' from 'markupsafe' (/usr/local/lib/python3.7/site-packages/"
            "markupsafe/__init__.py)\n"
            'File "/opt/python3.8/site-packages/jinja2/defaults.py", line 3, in <module>\n'
            "The error occurs because the soft_unicode method has been removed in markupsafe version 2.1.0."}
    strips = refine_web([page], MARKUPSAFE[-1])
    assert [s["text"] for s in strips] == ["The error occurs because the soft_unicode method has been removed in "
                                           "markupsafe version 2.1.0."]
