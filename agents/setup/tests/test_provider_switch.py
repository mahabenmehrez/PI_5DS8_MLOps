"""Which LLM the agent uses: --provider > BUILDER_LLM_PROVIDER (.env, the local switch) > agents.yaml > key."""
from agents.setup.builder_agent.llm import api
from agents.setup.builder_agent.llm.ollama import OllamaClient


def test_local_switch_beats_the_team_config(monkeypatch):
    monkeypatch.setattr(api, "load_dotenv", lambda *a: None)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("BUILDER_LLM_PROVIDER", "ollama")
    c = api.make_client("google/gemma-4-31b-it", configured="nvidia_nim")
    assert isinstance(c, OllamaClient) and c.model == "qwen2.5-coder:7b"      # NIM name replaced for Ollama
    assert isinstance(api.make_client(provider="api"), api.ApiClient)       # an explicit choice still wins


def test_team_config_when_no_local_switch(monkeypatch):
    monkeypatch.setattr(api, "load_dotenv", lambda *a: None)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.delenv("BUILDER_LLM_PROVIDER", raising=False)
    c = api.make_client("google/gemma-4-31b-it", configured="nvidia_nim")
    assert isinstance(c, api.ApiClient) and c.model == "google/gemma-4-31b-it"
    monkeypatch.setenv("BUILDER_OLLAMA_MODEL", "qwen-builder-sft-v5")
    assert api.make_client(configured="ollama").model == "qwen-builder-sft-v5"


# the two LLMs of the Setup Agent: slots (decide step, API model) and diagnosis (fix loop, fine-tuned model)
def _agent(**kw):
    from agents.setup.agent import SetupAgent
    return SetupAgent(**kw)


def test_diagnosis_uses_the_fine_tuned_model_and_slots_keep_theirs(monkeypatch):
    monkeypatch.setattr(api, "load_dotenv", lambda *a: None)
    monkeypatch.setattr(OllamaClient, "digest", property(lambda self: "sha256:test"))
    monkeypatch.setenv("BUILDER_DIAGNOSIS_MODEL", "qwen-builder-sft-v5")
    monkeypatch.delenv("BUILDER_DIAGNOSIS_PROVIDER", raising=False)
    slots = object()
    agent = _agent(diagnosis_client="auto")
    agent._client = slots                                   # what llm() would have built (the API model)
    assert agent.llm() is slots
    d = agent.diagnosis_llm()
    assert isinstance(d, OllamaClient) and d.model == "qwen-builder-sft-v5"


def test_no_diagnosis_model_means_one_model_for_both(monkeypatch):
    monkeypatch.setattr(api, "load_dotenv", lambda *a: None)
    monkeypatch.delenv("BUILDER_DIAGNOSIS_MODEL", raising=False)
    agent = _agent()
    slots = object()
    agent._client = slots
    assert agent.diagnosis_llm() is slots


def test_a_client_passed_in_is_used_for_both(monkeypatch):
    """Tests and rules-only runs pass client=None: the fine-tuned model in .env must not sneak in."""
    monkeypatch.setattr(api, "load_dotenv", lambda *a: None)
    monkeypatch.setenv("BUILDER_DIAGNOSIS_MODEL", "qwen-builder-sft-v5")
    assert _agent(client=None).diagnosis_llm() is None
    fake = object()
    assert _agent(client=fake).diagnosis_llm() is fake


def test_a_missing_fine_tuned_model_falls_back_to_the_slots_llm(monkeypatch):
    monkeypatch.setattr(api, "load_dotenv", lambda *a: None)
    monkeypatch.setenv("BUILDER_DIAGNOSIS_MODEL", "not-pulled")

    def missing(self):
        raise RuntimeError("model 'not-pulled' is not pulled in Ollama")
    monkeypatch.setattr(OllamaClient, "digest", property(missing))
    agent = _agent()
    slots = object()
    agent._client = slots
    assert agent.diagnosis_llm() is slots


def test_an_api_slots_model_gets_the_local_model_as_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "load_dotenv", lambda *a: None)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("BUILDER_LLM_PROVIDER", "api")
    monkeypatch.setenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    monkeypatch.delenv("BUILDER_SLOTS_FALLBACK", raising=False)
    c = _agent(root=str(tmp_path)).llm()
    assert isinstance(c, api.FallbackClient)
    assert [type(x).__name__ for x in c.clients] == ["ApiClient", "OllamaClient"] and c.clients[0].timeout == 60
    monkeypatch.setenv("BUILDER_SLOTS_FALLBACK", "off")
    assert isinstance(_agent(root=str(tmp_path)).llm(), api.ApiClient)


def test_no_cascade_unless_asked(monkeypatch):
    monkeypatch.setattr(api, "load_dotenv", lambda *a: None)
    monkeypatch.delenv("BUILDER_ESCALATION_MODEL", raising=False)
    assert _agent().escalation_llm() is None
    monkeypatch.setenv("BUILDER_ESCALATION_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    assert isinstance(_agent().escalation_llm(), api.ApiClient)
    assert _agent(client=None).escalation_llm() is None          # tests and rules-only runs: never a cascade
