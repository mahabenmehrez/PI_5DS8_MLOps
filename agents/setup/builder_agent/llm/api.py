"""OpenAI-compatible chat client (NVIDIA NIM by default), stdlib only, with the OllamaClient interface.

    from builder_agent.llm.api import ApiClient, make_client
    client = make_client()            # NVIDIA NIM when NVIDIA_API_KEY is set, else Ollama
    client.chat(messages, schema=...) -> ChatResponse

Same settings as the team's shared/llm.py: NVIDIA_API_KEY, NVIDIA_MODEL (default google/gemma-4-31b-it),
NVIDIA_BASE_URL (default https://integrate.api.nvidia.com/v1), read from the environment or from the
first .env file found walking up from the working directory. Temperature 0 and seed 42, as with Ollama.

Structured output: the JSON schema is sent as response_format json_schema; if the server refuses that
(HTTP 400/422), the client switches once to json_object and keeps the schema in a system message. Either
way the caller's validator checks every answer, as before. One transport retry on 429/5xx/timeouts.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .ollama import DEFAULT_OPTIONS, ChatResponse, OllamaClient, OllamaError

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "google/gemma-4-31b-it"
DEFAULT_EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
RETRY_WAIT_S = 10
MAX_TOKENS = 2048


class ApiError(OllamaError):
    """Same shape as OllamaError, so the callers' error handling is unchanged."""


def load_dotenv(start: str | Path | None = None) -> None:
    """Set variables from the first .env found walking up from `start` (never overrides the environment)."""
    here = Path(start or Path.cwd()).resolve()
    for folder in [here, *here.parents]:
        env = folder / ".env"
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
            return


def _settings(model: str | None, base_url: str | None, api_key: str | None) -> tuple[str, str, str]:
    load_dotenv()
    key = api_key or os.environ.get("NVIDIA_API_KEY", "")
    if not key or key == "your_nvidia_api_key_here":
        raise ApiError(["NVIDIA_API_KEY is not set (environment or .env)"])
    return (model or os.environ.get("NVIDIA_MODEL") or DEFAULT_MODEL,
            (base_url or os.environ.get("NVIDIA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"), key)


def _post(url: str, key: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Accept": "application/json",
                                          "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _retrying(call, errors: list[str], retry_wait_s: float, what: str):
    """Run call() with one retry on rate limits, server errors and timeouts; raise ApiError otherwise."""
    for attempt in (1, 2):
        try:
            return call()
        except urllib.error.HTTPError as e:
            errors.append(f"HTTP {e.code} from {what}: {e.read().decode('utf-8', 'replace')[:500]}")
            if e.code != 429 and e.code < 500:
                raise ApiError(errors) from e
        except urllib.error.URLError as e:
            errors.append(f"can't reach {what}: {e.reason}")
        except (ConnectionError, TimeoutError) as e:
            errors.append(f"{type(e).__name__}: {e}")
        if attempt == 1:
            time.sleep(retry_wait_s)
    raise ApiError(errors)


class ApiClient:
    """chat() like OllamaClient.chat(): the same messages, the same JSON schema, the same ChatResponse."""

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 options: dict | None = None, timeout: float = 300, retry_wait_s: float = RETRY_WAIT_S):
        self.model, self.base_url, self._key = _settings(model, base_url, api_key)
        self.options = dict(DEFAULT_OPTIONS if options is None else options)
        self.timeout, self.retry_wait_s = timeout, retry_wait_s
        self.json_mode = "json_schema"          # switches to "json_object" if the server refuses schemas

    @property
    def digest(self) -> str:
        """Hosted models have no local digest: the provider and model name identify the run."""
        return f"api:{self.base_url}:{self.model}"

    def _body(self, messages: list[dict], schema: dict | None) -> dict:
        body = {"model": self.model, "messages": messages, "max_tokens": MAX_TOKENS,
                "temperature": self.options.get("temperature", 0), "seed": self.options.get("seed", 42)}
        if schema is not None and self.json_mode == "json_schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "answer", "schema": schema, "strict": False}}
        elif schema is not None:
            hint = {"role": "system", "content": "Answer with one JSON object matching this JSON schema:\n"
                                                 + json.dumps(schema)}
            body["messages"] = [hint, *messages]
            body["response_format"] = {"type": "json_object"}
        return body

    def chat(self, messages: list[dict], schema: dict | None = None) -> ChatResponse:
        errors: list[str] = []
        url = f"{self.base_url}/chat/completions"
        t0 = time.perf_counter()
        try:
            resp = _retrying(lambda: _post(url, self._key, self._body(messages, schema), self.timeout),
                             errors, self.retry_wait_s, self.base_url)
        except ApiError:
            if schema is None or self.json_mode != "json_schema" or not any(
                    e.startswith(("HTTP 400", "HTTP 422")) for e in errors):
                raise
            self.json_mode = "json_object"          # the server doesn't take json_schema: once, for good
            t0 = time.perf_counter()
            resp = _retrying(lambda: _post(url, self._key, self._body(messages, schema), self.timeout),
                             errors, self.retry_wait_s, self.base_url)
        usage = resp.get("usage") or {}
        content = resp["choices"][0]["message"].get("content") or ""
        return ChatResponse(content=content, latency_s=round(time.perf_counter() - t0, 2),
                            prompt_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                            transport_errors=list(errors))


class FallbackClient:
    """chat() through the first client that answers: an API model first, a local model when the API fails
    (timeout, HTTP error, no network). `last_model` says which one answered; `failures` lists what failed."""

    def __init__(self, *clients):
        self.clients = [c for c in clients if c is not None]
        self.model = self.clients[0].model
        self.options = getattr(self.clients[0], "options", {})
        self.last_model: str | None = None
        self.failures: list[str] = []

    @property
    def digest(self) -> str:
        return " | ".join(str(getattr(c, "digest", c.model)) for c in self.clients)

    def chat(self, messages: list[dict], schema: dict | None = None) -> ChatResponse:
        errors = []
        for c in self.clients:
            try:
                resp = c.chat(messages, schema=schema)
                self.last_model = c.model
                return resp
            except Exception as e:  # noqa: BLE001 - the next client answers; all failing is raised
                errors.append(f"{c.model}: {type(e).__name__}: {str(e)[:160]}")
                self.failures.append(errors[-1])
        raise ApiError(errors)


class ApiEmbedder:
    """embed() like OllamaEmbedder.embed(), through the provider's /embeddings endpoint."""

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 timeout: float = 120, retry_wait_s: float = RETRY_WAIT_S):
        _, self.base_url, self._key = _settings(None, base_url, api_key)
        self.model = model or os.environ.get("NVIDIA_EMBED_MODEL") or DEFAULT_EMBED_MODEL
        self.timeout, self.retry_wait_s = timeout, retry_wait_s

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        body = {"model": self.model, "input": texts, "encoding_format": "float",
                "input_type": "query" if kind == "query" else "passage", "truncate": "END"}
        resp = _retrying(lambda: _post(f"{self.base_url}/embeddings", self._key, body, self.timeout),
                         [], self.retry_wait_s, self.base_url)
        return [d["embedding"] for d in sorted(resp["data"], key=lambda d: d["index"])]


PROVIDER_ALIASES = {"nvidia_nim": "api", "nvidia": "api", "nim": "api"}    # the team's agents.yaml says nvidia_nim


OLLAMA_MODEL = "qwen2.5-coder:7b"            # the model of every measured run (docs/RESULTS.md)


def make_client(model: str | None = None, provider: str | None = None, configured: str | None = None):
    """"api" (NVIDIA NIM) or "ollama", chosen in this order: `provider` (asked for explicitly, e.g. --provider),
    BUILDER_LLM_PROVIDER (environment or .env: the local switch, so one machine can use Ollama while the team's
    agents.yaml says nvidia_nim), `configured` (agents.yaml llm.provider), then api when a key is set, else ollama.
    With Ollama, a NIM model name ("google/...") is replaced by BUILDER_OLLAMA_MODEL or qwen2.5-coder:7b."""
    load_dotenv()
    provider = provider or os.environ.get("BUILDER_LLM_PROVIDER") or configured or (
        "api" if os.environ.get("NVIDIA_API_KEY") else "ollama")
    provider = PROVIDER_ALIASES.get(provider, provider)
    if provider == "api":
        return ApiClient(model)
    if provider == "ollama":
        if not model or "/" in model:
            model = os.environ.get("BUILDER_OLLAMA_MODEL") or OLLAMA_MODEL
        return OllamaClient(model)
    raise ValueError(f"unknown LLM provider {provider!r}: api or ollama")


def make_embedder(provider: str | None = None, model: str | None = None):
    """The embedder for the fix memory, or None (keyword search only) when none is reachable."""
    load_dotenv()
    provider = provider or os.environ.get("BUILDER_EMBED_PROVIDER") or (
        "api" if os.environ.get("NVIDIA_API_KEY") else "ollama")
    provider = PROVIDER_ALIASES.get(provider, provider)
    try:
        if provider == "api":
            return ApiEmbedder(model)
        if provider == "ollama":
            from .ollama import OllamaEmbedder
            return OllamaEmbedder(model or "nomic-embed-text")
    except ApiError:
        return None
    if provider == "none":
        return None
    raise ValueError(f"unknown embedding provider {provider!r}: api, ollama or none")
