import json
import sqlite3

import pytest

from shared import llm
from shared.llm import Decision, FakeLLM, decide

MENU = [
    {"name": "refuse", "level": 1, "params_schema": ["expected_gain", "threshold"]},
    {"name": "retrain_in_dry_run", "level": 2, "params_schema": ["batch_ids"]},
]
RULE = {"action": "refuse", "level": 1, "params": {"expected_gain": 0.5, "threshold": 2}, "reason": "rule: gain below threshold"}
GOOD = {"action": "retrain_in_dry_run", "level": 2, "params": {"batch_ids": ["batch-0001"]}, "reason": "MAE up 18%",
        "confidence": 0.8, "success_check": "run finished", "memory_refs": []}


class Agent:
    name = "trainer"
    prompt = "You are the Trainer."


@pytest.fixture
def db(tmp_path):
    return {"provider": "fake", "db_url": f"sqlite:///{tmp_path / 'board.db'}"}


def rows(cfg):
    with sqlite3.connect(cfg["db_url"][len("sqlite:///"):]) as con:
        return con.execute("SELECT attempt, outcome, agent FROM llm_calls ORDER BY rowid").fetchall()


def test_valid_answer(db):
    fake = FakeLLM([GOOD])
    d = decide(Agent(), {"mae_rise_pct": 18}, MENU, RULE, client=fake, config=db)
    assert d.action == "retrain_in_dry_run" and d.source == "llm"
    assert rows(db) == [(1, "ok", "trainer")]
    system = fake.calls[0][0]["content"]
    assert "retrain_in_dry_run (level 2" in system and "JSON" in system


def test_one_retry_then_ok(db):
    fake = FakeLLM([{**GOOD, "action": "delete_everything"}, GOOD])
    d = decide(Agent(), {}, MENU, RULE, client=fake, config=db)
    assert d.source == "llm" and len(fake.calls) == 2
    assert "unknown action" in fake.calls[1][-1]["content"]
    assert [r[1] for r in rows(db)] == ["invalid", "ok"]


@pytest.mark.parametrize("bad", [
    "not json",
    {**GOOD, "level": 1},                      # wrong level
    {**GOOD, "params": {}},                    # missing params
    {**GOOD, "confidence": 3},                 # out of range
])
def test_two_failures_fall_back_to_rules(db, bad):
    d = decide(Agent(), {}, MENU, RULE, client=FakeLLM([bad, bad]), config=db)
    assert d.source == "llm_invalid" and d.action == "refuse" and d.params == RULE["params"]
    assert [r[1] for r in rows(db)] == ["invalid", "invalid"]


def test_provider_error_falls_back(db):
    d = decide(Agent(), {}, MENU, RULE, client=FakeLLM([ConnectionError("ollama down")]), config=db)
    assert d.source == "llm_error" and d.action == "refuse"
    assert rows(db) == [(1, "error", "trainer")]


def test_off_means_rules_only():
    d = decide(Agent(), {}, MENU, Decision(**RULE), config={"provider": "off", "db_url": None})
    assert d.source == "rules" and d.action == "refuse"


def test_unwired_api_provider_falls_back():
    d = decide(Agent(), {}, MENU, RULE, config={"provider": "api", "db_url": None})
    assert d.source == "llm_error"


def test_empty_fake_echoes_rule():
    d = decide(Agent(), {}, MENU, RULE, client=FakeLLM(), config={"provider": "fake", "db_url": None})
    assert d.action == "refuse" and d.source == "llm"


def test_context_cap():
    ctx = {"numbers": {"mae": 0.5}, "similar_cases": [{"id": f"mem-{i}", "text": "x" * 200} for i in range(200)]}
    capped = llm.cap_context(ctx)
    assert len(json.dumps(capped)) <= llm.MAX_CONTEXT_TOKENS * llm.CHARS_PER_TOKEN
    assert capped["_truncated"] and capped["numbers"] == {"mae": 0.5} and capped["similar_cases"]


def test_ollama_request_shape(monkeypatch):
    sent = {}

    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": json.dumps(GOOD)}, "prompt_eval_count": 120, "eval_count": 40}

    import httpx
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: sent.update(url=url, body=json) or Resp())
    d = decide(Agent(), {}, MENU, RULE, config={"provider": "ollama", "model": "llama3.2:3b", "db_url": None,
                                                 "base_url": "http://host.docker.internal:11434"})
    assert d.source == "llm"
    assert sent["url"] == "http://host.docker.internal:11434/api/chat"
    assert sent["body"]["format"] == "json" and sent["body"]["options"]["temperature"] == 0


def test_nvidia_nim_provider_sends_openai_request(monkeypatch):
    import httpx

    seen = {}

    class Resp:
        status_code = 200

        def json(self):
            answer = {"action": "refuse", "level": 1, "params": {"expected_gain": 1, "threshold": 2}, "reason": "r", "confidence": 0.5}
            return {"choices": [{"message": {"content": json.dumps(answer)}}], "usage": {"prompt_tokens": 10}}

    def post(url, headers, json, timeout):
        seen.update(url=url, auth=headers["Authorization"], body=dict(json))
        return Resp()

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    d = decide(Agent(), {}, MENU, RULE, config={"provider": "nvidia_nim", "model": "m", "db_url": None,
                                                 "base_url": "https://integrate.api.nvidia.com/v1"})
    assert seen["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert seen["auth"] == "Bearer k" and seen["body"]["response_format"] == {"type": "json_object"}
    assert d.source == "llm" and d.action == "refuse"


def test_nvidia_nim_retries_without_response_format_and_reports_http_errors(monkeypatch):
    import httpx

    from shared.llm import CALL_LOG

    bodies = []

    class Resp:
        def __init__(self, code):
            self.status_code, self.text = code, "bad request"

    def post(url, headers, json, timeout):
        bodies.append(dict(json))
        return Resp(400)

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    d = decide(Agent(), {}, MENU, RULE, config={"provider": "nvidia_nim", "model": "m", "db_url": None,
                                                 "base_url": "https://integrate.api.nvidia.com/v1"})
    assert "response_format" in bodies[0] and "response_format" not in bodies[1]
    assert d.source == "llm_error" and "HTTP 400" in CALL_LOG[-1].error


def test_nvidia_nim_without_key_falls_back(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    d = decide(Agent(), {}, MENU, RULE, config={"provider": "nvidia_nim", "model": "m", "db_url": None,
                                                 "base_url": "https://integrate.api.nvidia.com/v1"})
    assert d.source == "llm_error"


def test_reasoning_model_answer_with_think_block_is_accepted():
    from shared.llm import validate, MenuItem

    raw = ('<think>gain 1% is below 2%, so refuse</think>\n```json\n'
           + json.dumps({"action": "refuse", "level": 1, "params": {"expected_gain": 1, "threshold": 2},
                         "reason": "r", "confidence": 0.5}) + "\n```")
    d = validate(raw, [MenuItem.coerce(m) for m in MENU])
    assert d.action == "refuse"


def test_groq_provider_uses_groq_url_and_key(monkeypatch):
    from shared.llm import config_for

    monkeypatch.delenv("GROQ_MODEL", raising=False)
    cfg = config_for("groq")
    assert cfg["base_url"] == "https://api.groq.com/openai/v1"
    assert cfg["api_key_env"] == "GROQ_API_KEY" and cfg["model"] == "qwen/qwen3.8-27b"
