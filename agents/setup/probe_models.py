"""Which NVIDIA chat models answer your key right now?  python agents/setup/probe_models.py

Sends one tiny JSON-only request to each candidate model in parallel and prints the latency and whether the
reply is valid JSON (the agent's diagnosis needs JSON). Never prints the key.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agents.setup.builder_agent.llm.api import DEFAULT_BASE_URL, load_dotenv  # noqa: E402

TIMEOUT = 60
load_dotenv()
key = os.environ.get("NVIDIA_API_KEY", "")
base = os.environ.get("NVIDIA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
PROMPT = ('A Python build failed with: ModuleNotFoundError: No module named \'yaml\'. '
          'Reply with JSON only: {"action": "add_dependency", "package": "<pip name>"}')


def req(method, path, body=None):
    r = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body else None,
                               headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=TIMEOUT) as resp:
        return json.loads(resp.read())


def probe(model):
    t = time.time()
    try:
        out = req("POST", "/chat/completions", {"model": model, "max_tokens": 200, "temperature": 0,
                                                "messages": [{"role": "user", "content": PROMPT}]})
        text = out["choices"][0]["message"]["content"] or ""
        s, e = text.find("{"), text.rfind("}")
        try:
            ok = json.loads(text[s:e + 1]).get("package", "").lower() in ("pyyaml", "pyyaml>=5")
            verdict = "JSON ok, right answer" if ok else "JSON ok, wrong answer"
        except Exception:
            verdict = "not JSON: " + text[:60].replace("\n", " ")
        return model, f"{time.time() - t:5.1f}s", verdict
    except urllib.error.HTTPError as e:
        return model, f"{time.time() - t:5.1f}s", f"HTTP {e.code}"
    except Exception as e:
        return model, f"{time.time() - t:5.1f}s", f"no answer ({type(e).__name__})"


ids = sorted(m["id"] for m in req("GET", "/models")["data"])
WORDS = ("instruct", "-it", "chat", "coder", "qwen", "llama-3", "mistral", "deepseek", "nemotron", "gpt-oss", "kimi", "glm")
SKIP = ("embed", "rerank", "guard", "vision", "vl", "reward", "safety", "clip", "parse", "retriever", "base")
cands = [m for m in ids if any(w in m for w in WORDS) and not any(s in m for s in SKIP)]
print(f"{len(ids)} models listed, probing {len(cands)} chat models ({TIMEOUT}s each, in parallel)...\n")
with ThreadPoolExecutor(16) as pool:
    rows = list(pool.map(probe, cands))
for m, t, v in sorted(rows, key=lambda r: (not r[2].startswith("JSON ok, right"), float(r[1].strip().rstrip("s")))):
    print(f"{t}  {v:28.28}  {m}")
