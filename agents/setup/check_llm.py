"""Three real LLM calls, to show the Setup Agent's LLM works (run from the Pi-Mlops folder):

    python agents/setup/check_llm.py

1. chat: Gemma answers in JSON that matches a schema (what slot filling and diagnosis need)
2. embedding: one vector from nvidia/nv-embedqa-e5-v5 (what the RAG's vector half needs)
3. diagnosis: a real dry-run error goes through Corrective RAG (memory graded, web search when it doesn't fit,
   knowledge refined), the team prompt and the validator, and the LLM picks one fix from the team menu
Provider: BUILDER_LLM_PROVIDER / BUILDER_EMBED_PROVIDER in .env (ollama or api), else NVIDIA when a key is set.
The key is read from .env (NVIDIA_API_KEY), never printed.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.setup.builder_agent.llm.api import make_client, make_embedder  # noqa: E402

ERROR_LOG = """python baseline/train.py .setup/sample/winequality-red.csv
Traceback (most recent call last):
  File "baseline/train.py", line 3, in <module>
    from jinja2 import Markup
  File "/usr/lib/python3.10/site-packages/jinja2/__init__.py", line 12, in <module>
    from markupsafe import escape, soft_unicode
ImportError: cannot import name 'soft_unicode' from 'markupsafe' (/usr/lib/python3.10/site-packages/markupsafe/__init__.py)
make: *** [Makefile:37: train] Error 1"""


def main() -> int:
    client = make_client()
    print(f"1. chat with {client.model} ({type(client).__name__})")
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    r = client.chat([{"role": "user", "content": "In one sentence: what does pH measure in red wine? Answer as JSON."}],
                    schema=schema)
    print(f"   ok in {r.latency_s:.1f}s, {r.prompt_tokens} -> {r.output_tokens} tokens: {json.loads(r.content)}")

    print("2. embedding")
    embedder = make_embedder()
    if embedder is None:
        print("   no embedder (no key, or Ollama not running)")
        return 1
    v = embedder.embed(["ModuleNotFoundError: No module named 'sklearn'"], kind="query")
    print(f"   ok: one vector of {len(v[0])} numbers")

    print("3. diagnosis of a failed dry run (Corrective RAG + team prompt + validator)")
    from agents.setup.builder_agent.memory import FixMemory
    from agents.setup.builder_agent.memory.crag import CorrectiveRAG, TavilySearch
    from agents.setup.builder_agent.memory.learned import memory_docs
    from agents.setup.builder_agent.team import TeamLayout, team_plan
    from agents.setup.builder_agent.team.fix import diagnose_team
    plan = team_plan(TeamLayout.from_contracts())
    memory = FixMemory.build(embedder, docs=memory_docs("learned"), keywords_if_unreachable=True)
    crag = CorrectiveRAG(memory, search=TavilySearch())
    r = crag.retrieve("train", ERROR_LOG.splitlines())
    print(f"   CRAG action: {r.action}")
    for g in r.graded:
        print(f"     {g.id}: relevance {g.score} -> {'kept' if g.kept else 'discarded'}  {g.why}")
    if r.query:
        print(f"   rewritten web query: {r.query}")
        print(f"   web results: {len(r.web)} sentences kept ({r.web_source})")
        for s in r.web:
            print(f"     - {s['text'][:140]}  [{s['url']}]")
    t0 = time.perf_counter()
    d, fix = diagnose_team(plan, {"stage": "train", "log": ERROR_LOG}, client, [], 1, 3, memory=memory, crag=crag)
    print(f"   {d.calls} LLM call(s) in {time.perf_counter() - t0:.1f}s")
    for reasons in d.rejected:
        print(f"   rejected by the validator: {'; '.join(reasons)}")
    print(f"   fix: {d.text}")
    print(f"   reason: {d.reason}")
    return 0 if fix is not None or d.give_up else 1


if __name__ == "__main__":
    raise SystemExit(main())
