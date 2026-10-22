"""LLM helper (roadmap 5.9, owner P4).

    llm.decide(agent, context, menu, fallback) -> Decision

The only way an agent calls an LLM. The model must answer one action from
the menu, in JSON:

    {"action": "quarantine", "level": 1, "params": {"batch_id": "batch-0042"},
     "reason": "...", "confidence": 0.86, "success_check": "...", "memory_refs": ["mem-0012"]}

Rules:
- temperature 0, JSON output enforced (Ollama ``format: json``);
- the answer is checked with pydantic and against the menu: unknown action,
  wrong level or missing params -> one retry with the error added; a second
  failure -> the rule decision (``fallback``), logged as ``llm_invalid``;
- a provider error (timeout, connection refused) -> the rule decision, logged
  as ``llm_error``;
- the context holds numbers and memory matches only, capped near 2,000 tokens;
- every call is logged to the ``llm_calls`` table.

Provider comes from ``llm.provider`` in agents.yaml (or ``LLM_PROVIDER``):
``ollama`` (free, local, the default) | ``nvidia_nim`` / ``groq`` / ``openai`` (any
OpenAI-style endpoint, key read from ``llm.api_key_env``) | ``fake`` | ``off``. ``fake`` answers from a script
(unit tests); ``off`` skips the LLM and returns the rule decision, which is
the "traditional pipeline" setting.
"""

from __future__ import annotations

import json
import re
import os
from contextlib import closing
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

ROOT = Path(__file__).resolve().parent.parent
MAX_CONTEXT_TOKENS = 2000
CHARS_PER_TOKEN = 4  # rough estimate, good enough for a cap

DEFAULT_CONFIG = {
    "provider": "ollama",       # free and local: `ollama pull qwen2.5:3b`
    "model": "qwen2.5:3b",
    "temperature": 0,
    "base_url": "http://localhost:11434",
}  # timeout_s: 60 for ollama, 300 for NVIDIA NIM (its free tier can queue); LLM_TIMEOUT_S overrides


# --------------------------------------------------------------------------
# Data shapes
# --------------------------------------------------------------------------
class Decision(BaseModel):
    action: str
    level: int = Field(ge=0, le=3)
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    confidence: float | None = Field(default=None, ge=0, le=1)
    success_check: str = ""
    memory_refs: list[str] = Field(default_factory=list)
    # set by decide(), not by the model: llm | rules | llm_invalid | llm_error
    source: str = "llm"


@dataclass
class MenuItem:
    """What decide() needs to know about one allowed action.

    Built from shared.base_agent.Action (name, level, params_schema,
    success_check) or from a plain dict with the same keys.
    """

    name: str
    level: int
    required_params: list[str] = field(default_factory=list)
    success_check: str = ""

    @classmethod
    def coerce(cls, item: Any) -> MenuItem:
        if isinstance(item, MenuItem):
            return item
        get = item.get if isinstance(item, dict) else (lambda k, d=None: getattr(item, k, d))
        schema = get("params_schema") or get("params") or {}
        if isinstance(schema, dict):
            required = schema.get("required", list(schema.get("properties", schema).keys()))
        else:
            required = list(schema)
        return cls(name=get("name"), level=int(get("level")), required_params=list(required), success_check=get("success_check") or "")


@dataclass
class CallRecord:
    id: str
    agent: str
    incident_id: str | None
    attempt: int
    prompt: str
    raw_answer: str | None
    parsed: dict | None
    outcome: str  # ok | invalid | error | fallback
    error: str | None
    tokens_in: int | None
    tokens_out: int | None
    latency_ms: int
    provider: str
    model: str | None


class LLMError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------
class FakeLLM:
    """Scripted answers for unit tests and stubs.

    Each script entry is a dict (sent as JSON), a raw string, an Exception
    (raised), or a callable(messages) returning one of those. When the
    script runs out, it echoes the fallback decision.
    """

    def __init__(self, script: Iterable[Any] = ()):
        self.script = list(script)
        self.calls: list[list[dict]] = []
        self.fallback: dict | None = None

    def chat(self, messages: list[dict], cfg: dict) -> tuple[str, int | None, int | None]:
        self.calls.append(messages)
        answer: Any = self.script.pop(0) if self.script else self.fallback or {}
        if callable(answer):
            answer = answer(messages)
        if isinstance(answer, BaseException):
            raise answer
        text = answer if isinstance(answer, str) else json.dumps(answer)
        return text, None, None


class OllamaLLM:
    def chat(self, messages: list[dict], cfg: dict) -> tuple[str, int | None, int | None]:
        import httpx

        if not cfg.get("model"):
            raise LLMError("llm.model is not set in agents.yaml")
        resp = httpx.post(
            f"{cfg['base_url'].rstrip('/')}/api/chat",
            json={
                "model": cfg["model"],
                "messages": messages,
                "format": "json",
                "stream": False,
                "options": {"temperature": cfg.get("temperature", 0)},
            },
            timeout=float(cfg.get("timeout_s") or 60),
        )
        resp.raise_for_status()
        body = resp.json()
        return body["message"]["content"], body.get("prompt_eval_count"), body.get("eval_count")


class OpenAICompatLLM:
    """Any OpenAI-style /chat/completions endpoint, e.g. NVIDIA NIM (the Reviewer's choice).

    agents.yaml: llm.provider nvidia_nim, llm.base_url, llm.model and
    llm.api_key_env (the environment variable holding the key, NVIDIA_API_KEY by default).
    """

    def chat(self, messages: list[dict], cfg: dict) -> tuple[str, int | None, int | None]:
        import httpx

        key = os.environ.get(cfg.get("api_key_env") or "NVIDIA_API_KEY")
        if not key:
            raise LLMError(f"no API key in ${cfg.get('api_key_env') or 'NVIDIA_API_KEY'}")
        if not cfg.get("model"):
            raise LLMError("llm.model is not set in agents.yaml")
        base = cfg["base_url"].rstrip("/").removesuffix("/chat/completions")  # some .env files give the full URL
        url = f"{base}/chat/completions"
        body = {
            "model": cfg["model"],
            "messages": messages,
            "temperature": cfg.get("temperature", 0),
            "max_tokens": int(cfg.get("max_tokens") or 4096),  # room for a reasoning model's thinking
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
        timeout = float(cfg.get("timeout_s") or 300)
        resp = httpx.post(url, headers=headers, json=body, timeout=timeout)
        if resp.status_code in (400, 422):  # some models refuse response_format; the prompt already asks for JSON
            body.pop("response_format")
            resp = httpx.post(url, headers=headers, json=body, timeout=timeout)
        if resp.status_code >= 400:
            raise LLMError(f"HTTP {resp.status_code} from {url}: {resp.text[:300]}")
        body = resp.json()
        usage = body.get("usage") or {}
        content = body["choices"][0]["message"].get("content") or ""
        return content, usage.get("prompt_tokens"), usage.get("completion_tokens")


class ApiLLM:
    def chat(self, messages: list[dict], cfg: dict) -> tuple[str, int | None, int | None]:
        raise LLMError("llm.provider 'api' is not wired yet: pick the API and model first")


PROVIDERS: dict[str, Callable[[], Any]] = {
    "ollama": OllamaLLM, "nvidia_nim": OpenAICompatLLM, "groq": OpenAICompatLLM, "openai": OpenAICompatLLM, "api": ApiLLM, "fake": FakeLLM,
}


# --------------------------------------------------------------------------
# Config and logging
# --------------------------------------------------------------------------
# What each provider needs when agents.yaml names a different one (e.g. `solo --llm ollama`).
PROVIDER_DEFAULTS = {
    "ollama": {"model": "qwen2.5:3b", "base_url": "http://localhost:11434"},
    "groq": {"model": "qwen/qwen3.8-27b", "base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY"},
    "nvidia_nim": {"model": "google/gemma-4-31b-it", "base_url": "https://integrate.api.nvidia.com/v1",
                   "api_key_env": "NVIDIA_API_KEY"},
}


def load_dotenv(path: str | Path | None = None) -> None:
    """Put KEY=VALUE lines of the repo's .env into os.environ (variables already set win). Keys stay out of git."""
    path = Path(path or ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(path: str | Path | None = None) -> dict:
    """llm section of agents.yaml + db_url + env overrides (LLM_PROVIDER, LLM_MODEL, LLM_BASE_URL, LLM_TIMEOUT_S)."""
    load_dotenv()
    cfg = dict(DEFAULT_CONFIG)
    db_url = os.environ.get("DB_URL")
    path = Path(path or os.environ.get("AGENTS_YAML", ROOT / "agents.yaml"))
    if path.exists():
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg.update(data.get("llm") or {})
        db_url = db_url or data.get("db_url")
    for env, key in (("LLM_PROVIDER", "provider"), ("LLM_MODEL", "model"), ("LLM_BASE_URL", "base_url"),
                     ("LLM_TIMEOUT_S", "timeout_s")):
        if os.environ.get(env):
            cfg[key] = os.environ[env]
    if cfg.get("provider") == "ollama" and os.environ.get("OLLAMA_BASE_URL"):  # e.g. host.docker.internal in Docker
        cfg["base_url"] = os.environ["OLLAMA_BASE_URL"]
    cfg["db_url"] = db_url
    return cfg


def config_for(provider: str) -> dict:
    """load_config(), switched to ``provider``; model and URL come from PROVIDER_DEFAULTS when agents.yaml names another one."""
    cfg = load_config()
    if provider in ("off", "fake") or provider == cfg.get("provider"):
        return {**cfg, "provider": provider}
    defaults = dict(PROVIDER_DEFAULTS.get(provider, {}))
    if provider == "groq":  # free key from console.groq.com/keys
        defaults["model"] = os.environ.get("GROQ_MODEL") or defaults["model"]
    if provider == "nvidia_nim":  # the Reviewer's .env names these
        defaults["model"] = os.environ.get("NVIDIA_MODEL") or defaults["model"]
        defaults["base_url"] = os.environ.get("NVIDIA_BASE_URL") or defaults["base_url"]
    if provider == "ollama":
        defaults["base_url"] = os.environ.get("OLLAMA_BASE_URL") or defaults.get("base_url")
    return {**cfg, **defaults, "provider": provider}


CALL_LOG: list[CallRecord] = []  # in-memory copy, handy in tests


def _sqlite_path(db_url: str | None) -> str | None:
    if db_url and db_url.startswith("sqlite:///"):
        return db_url[len("sqlite:///"):]
    return None


def _log_call(rec: CallRecord, db_url: str | None) -> None:
    CALL_LOG.append(rec)
    path = _sqlite_path(db_url)
    if not path:
        return
    with closing(sqlite3.connect(path, timeout=10)) as con, con:  # close too, so Windows can delete the file
        con.execute("PRAGMA journal_mode=WAL")
        con.execute(
            """CREATE TABLE IF NOT EXISTS llm_calls (
                id TEXT PRIMARY KEY, time TEXT, run_id TEXT, agent TEXT, incident_id TEXT,
                attempt INTEGER, prompt TEXT, raw_answer TEXT, parsed TEXT, outcome TEXT,
                error TEXT, tokens_in INTEGER, tokens_out INTEGER, latency_ms INTEGER,
                provider TEXT, model TEXT)"""
        )
        con.execute(
            "INSERT INTO llm_calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                rec.id, _now(), os.environ.get("RUN_ID", "dev"), rec.agent, rec.incident_id,
                rec.attempt, rec.prompt, rec.raw_answer, json.dumps(rec.parsed) if rec.parsed else None,
                rec.outcome, rec.error, rec.tokens_in, rec.tokens_out, rec.latency_ms, rec.provider, rec.model,
            ),
        )


def _now() -> str:
    try:
        from shared import clock  # owner P3; replay time during experiments

        return clock.now().isoformat()
    except Exception:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Prompt building
# --------------------------------------------------------------------------
def cap_context(context: dict, max_tokens: int = MAX_CONTEXT_TOKENS) -> dict:
    """Shrink the context until it fits: drop items from the longest list first."""
    ctx = json.loads(json.dumps(context, default=str))
    limit = max_tokens * CHARS_PER_TOKEN

    def lists(obj, depth=0):
        if isinstance(obj, dict) and depth < 3:
            for v in obj.values():
                if isinstance(v, list) and v:
                    yield v
                yield from lists(v, depth + 1)

    while len(json.dumps(ctx)) > limit:
        candidates = sorted(lists(ctx), key=lambda lst: len(json.dumps(lst)), reverse=True)
        if not candidates:
            break
        candidates[0].pop()
        ctx["_truncated"] = True
    if len(json.dumps(ctx)) > limit:
        ctx = {"_truncated": True, "summary": json.dumps(ctx)[: limit - 100]}
    return ctx


def _agent_prompt(agent: Any) -> str:
    text = getattr(agent, "prompt", None)
    if text:
        return text
    name = getattr(agent, "name", None) or str(agent)
    path = ROOT / "agents" / name / "prompt.md"
    return path.read_text() if path.exists() else f"You are the {name} agent of a wine-quality MLOps pipeline."


def build_messages(agent: Any, context: dict, menu: list[MenuItem]) -> list[dict]:
    menu_text = "\n".join(
        f"- {m.name} (level {m.level}; params: {', '.join(m.required_params) or 'none'})"
        + (f"; success check: {m.success_check}" if m.success_check else "")
        for m in menu
    )
    system = (
        _agent_prompt(agent).strip()
        + "\n\nChoose exactly ONE action from this menu:\n"
        + menu_text
        + "\n\nAnswer with a single JSON object and nothing else, with keys: "
        '"action" (a menu name), "level" (the menu level of that action), "params" (object), '
        '"reason" (one sentence quoting the numbers you used), "confidence" (0.0-1.0), '
        '"success_check" (how to tell it worked), "memory_refs" (list of memory ids you relied on).'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(cap_context(context), indent=1)},
    ]


def _json_part(raw: str | None) -> str:
    """The JSON object in an answer: drops a <think>...</think> block (Qwen3 and other reasoning models) and code fences."""
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


def validate(raw: str, menu: list[MenuItem]) -> Decision:
    """Parse and check one answer. Raises ValueError with a message the model can act on."""
    try:
        data = json.loads(_json_part(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"answer is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("answer must be a JSON object")
    data.pop("source", None)
    try:
        decision = Decision(**data)
    except ValidationError as exc:
        raise ValueError(f"answer does not match the schema: {exc.errors(include_url=False)}") from exc
    by_name = {m.name: m for m in menu}
    item = by_name.get(decision.action)
    if item is None:
        raise ValueError(f"unknown action {decision.action!r}; allowed: {sorted(by_name)}")
    if decision.level != item.level:
        raise ValueError(f"action {item.name!r} has level {item.level}, not {decision.level}")
    missing = [p for p in item.required_params if p not in decision.params]
    if missing:
        raise ValueError(f"action {item.name!r} is missing params {missing}")
    if not decision.success_check:
        decision.success_check = item.success_check
    return decision


# --------------------------------------------------------------------------
# decide()
# --------------------------------------------------------------------------
def decide(
    agent: Any,
    context: dict,
    menu: Iterable[Any],
    fallback: Decision | dict,
    *,
    incident_id: str | None = None,
    client: Any | None = None,
    config: dict | None = None,
) -> Decision:
    cfg = {**load_config(), **(config or {})}
    if config and "provider" in config and "api_key_env" not in config:
        # another provider than agents.yaml's: never send that file's key (e.g. a Groq key to NVIDIA)
        cfg["api_key_env"] = PROVIDER_DEFAULTS.get(config["provider"], {}).get("api_key_env")
    items = [MenuItem.coerce(m) for m in menu]
    rule = fallback if isinstance(fallback, Decision) else Decision(**fallback)
    agent_name = getattr(agent, "name", None) or str(agent)
    provider = cfg.get("provider", "fake")

    if provider == "off":
        return rule.model_copy(update={"source": "rules"})

    if client is None:
        factory = PROVIDERS.get(provider)
        if factory is None:
            client = ApiLLM()  # raises a clear error, logged below
        else:
            client = factory()
    if isinstance(client, FakeLLM):  # echo this call's rule decision once the script runs out
        client.fallback = rule.model_dump(exclude={"source"})

    messages = build_messages(agent, context, items)
    last_error = None
    for attempt in (1, 2):
        start = time.monotonic()
        raw = tokens_in = tokens_out = None
        try:
            raw, tokens_in, tokens_out = client.chat(messages, cfg)
        except Exception as exc:  # provider down, timeout, bad config
            _log_call(
                CallRecord(_id(), agent_name, incident_id, attempt, messages[-1]["content"], None, None, "error",
                           f"{type(exc).__name__}: {exc}", None, None, _ms(start), provider, cfg.get("model")),
                cfg.get("db_url"),
            )
            return rule.model_copy(update={"source": "llm_error"})
        try:
            decision = validate(raw, items)
        except ValueError as exc:
            last_error = str(exc)
            _log_call(
                CallRecord(_id(), agent_name, incident_id, attempt, messages[-1]["content"], raw, None, "invalid",
                           last_error, tokens_in, tokens_out, _ms(start), provider, cfg.get("model")),
                cfg.get("db_url"),
            )
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Your answer was rejected: {last_error}. Answer again with one valid JSON object."},
            ]
            continue
        _log_call(
            CallRecord(_id(), agent_name, incident_id, attempt, messages[-1]["content"], raw, decision.model_dump(), "ok",
                       None, tokens_in, tokens_out, _ms(start), provider, cfg.get("model")),
            cfg.get("db_url"),
        )
        return decision.model_copy(update={"source": "llm"})

    return rule.model_copy(update={"source": "llm_invalid", "reason": f"{rule.reason} [llm_invalid: {last_error}]".strip()})


def _id() -> str:
    return "llm-" + uuid.uuid4().hex[:12]


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)
