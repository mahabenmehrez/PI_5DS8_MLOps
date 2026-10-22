"""The OpenAI-compatible client (NVIDIA NIM) against a local fake server: no key, no network."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agents.setup.builder_agent.llm.api import ApiClient, ApiEmbedder, ApiError, make_client
from agents.setup.builder_agent.llm.ollama import OllamaClient


class FakeNim(BaseHTTPRequestHandler):
    script: list = []          # (status, body) answers, in order
    seen: list = []            # (path, headers, json body) of every request

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeNim.seen.append((self.path, dict(self.headers), body))
        status, answer = FakeNim.script.pop(0)
        data = json.dumps(answer).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def nim(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), FakeNim)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    FakeNim.script, FakeNim.seen = [], []
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("NVIDIA_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    yield FakeNim
    server.shutdown()


def chat_answer(content, prompt=120, completion=30):
    return {"choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


SCHEMA = {"type": "object", "properties": {"fix": {"type": "object"}}, "required": ["fix"]}


def test_chat_sends_schema_key_and_settings(nim):
    nim.script = [(200, chat_answer('{"fix": {"action": "give_up"}}'))]
    client = ApiClient(retry_wait_s=0)
    r = client.chat([{"role": "user", "content": "hi"}], schema=SCHEMA)
    assert json.loads(r.content)["fix"]["action"] == "give_up"
    assert (r.prompt_tokens, r.output_tokens) == (120, 30)
    path, headers, body = nim.seen[0]
    assert path == "/v1/chat/completions" and headers["Authorization"] == "Bearer test-key"
    assert body["model"] == "google/gemma-4-31b-it" and body["temperature"] == 0 and body["seed"] == 42
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert client.digest.startswith("api:") and client.digest.endswith("google/gemma-4-31b-it")


def test_falls_back_to_json_object_once_when_schemas_are_refused(nim):
    nim.script = [(400, {"error": "response_format json_schema not supported"}), (200, chat_answer('{"fix": {}}')),
                  (200, chat_answer('{"fix": {}}'))]
    client = ApiClient(retry_wait_s=0)
    client.chat([{"role": "user", "content": "hi"}], schema=SCHEMA)
    client.chat([{"role": "user", "content": "again"}], schema=SCHEMA)
    assert client.json_mode == "json_object" and len(nim.seen) == 3          # refused once, then remembered
    second = nim.seen[1][2]
    assert second["response_format"] == {"type": "json_object"}
    assert second["messages"][0]["role"] == "system" and '"required": ["fix"]' in second["messages"][0]["content"]


def test_rate_limit_is_retried_once_then_reported(nim):
    nim.script = [(429, {"error": "slow down"}), (200, chat_answer("{}"))]
    r = ApiClient(retry_wait_s=0).chat([{"role": "user", "content": "hi"}])
    assert r.content == "{}" and r.transport_errors and "429" in r.transport_errors[0]
    nim.script = [(429, {}), (429, {})]
    with pytest.raises(ApiError):
        ApiClient(retry_wait_s=0).chat([{"role": "user", "content": "hi"}])


def test_bad_request_without_schema_is_not_retried(nim):
    nim.script = [(401, {"error": "bad key"})]
    with pytest.raises(ApiError, match="401"):
        ApiClient(retry_wait_s=0).chat([{"role": "user", "content": "hi"}])
    assert len(nim.seen) == 1


def test_embeddings_keep_order_and_input_type(nim):
    nim.script = [(200, {"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]})]
    vecs = ApiEmbedder(retry_wait_s=0).embed(["a", "b"], kind="query")
    assert vecs == [[1.0, 0.0], [0.0, 1.0]]
    path, _, body = nim.seen[0]
    assert path == "/v1/embeddings" and body["input_type"] == "query" and body["input"] == ["a", "b"]


def test_make_client_picks_the_provider(nim, monkeypatch):
    monkeypatch.delenv("BUILDER_LLM_PROVIDER", raising=False)
    monkeypatch.chdir("/")                                              # no .env up from here (a local switch in
    assert isinstance(make_client(), ApiClient)                         # the repo's .env must not decide); a key is set: the team API
    assert isinstance(make_client(provider="ollama"), OllamaClient)
    monkeypatch.delenv("NVIDIA_API_KEY")
    assert isinstance(make_client(), OllamaClient)
    with pytest.raises(ApiError, match="NVIDIA_API_KEY"):
        ApiClient()


def test_memory_falls_back_to_keywords_when_embeddings_are_unreachable():
    from agents.setup.builder_agent.memory import FixMemory

    class Down:
        def embed(self, texts, kind="document"):
            raise ConnectionError("no embedding service")

    with pytest.raises(ConnectionError):
        FixMemory.build(Down())                                          # evaluations: fail loudly
    m = FixMemory.build(Down(), keywords_if_unreachable=True)           # create: keep going
    hits = m.advanced("train", ["ImportError: libffi.so.8: cannot open shared object file"])
    assert hits[0].doc.id == "gen-006" and set(hits[0].ranks) == {"bm25"}
