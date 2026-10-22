"""Is the NVIDIA key alive? Run from the Pi-Mlops folder:  python agents/setup/check_key.py

Shows where the key comes from (never the key itself), the models NVIDIA serves today, and one tiny call
on Gemma and on another live chat model: tells a dead key from a model the key can't use.
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agents.setup.builder_agent.llm.api import DEFAULT_BASE_URL, load_dotenv  # noqa: E402

GEMMA = "google/gemma-4-31b-it"
TIMEOUT = 120
before = os.environ.get("NVIDIA_API_KEY")
load_dotenv()
key = os.environ.get("NVIDIA_API_KEY", "")
source = "a Windows environment variable (it wins over .env)" if before else ".env"
print(f"key from {source}: {len(key)} characters, ...{key[-4:]}" if key else "no key found")
base = os.environ.get("NVIDIA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def call(method, path, body=None):
    req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:200]
    except (TimeoutError, OSError) as e:                 # the key was accepted, the model is slow or queued
        return None, f"no answer in {TIMEOUT}s ({type(e).__name__}): the model is busy, try again in a minute"


status, models = call("GET", "/models")
ids = sorted(m["id"] for m in models.get("data", [])) if isinstance(models, dict) else []
print(f"GET /models: HTTP {status}, {len(ids)} models; Gemma listed: {GEMMA in ids}")
# widely served chat models first (a model can be listed but not served to free accounts: HTTP 404)
POPULAR = ("meta/llama-3.3-70b-instruct", "meta/llama-3.1-70b-instruct", "qwen/qwen2.5-coder-32b-instruct",
           "mistralai/mistral-small-24b-instruct", "deepseek-ai/deepseek-r1-distill-qwen-32b", "google/gemma-3-27b-it",
           "microsoft/phi-4-mini-instruct", "nvidia/llama-3.1-nemotron-nano-8b-v1")
others = [m for m in POPULAR if m in ids][:3] or [m for m in ids if m != GEMMA and "instruct" in m][:3]
for model in [GEMMA] + others:
    status, out = call("POST", "/chat/completions", {"model": model, "max_tokens": 5,
                                                     "messages": [{"role": "user", "content": "Say hi"}]})
    print(f"chat {model}: HTTP {status}", "OK" if status == 200 else out)
