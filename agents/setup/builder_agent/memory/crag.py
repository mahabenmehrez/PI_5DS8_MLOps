"""Corrective RAG (CRAG, Yan et al. 2024, arXiv 2401.15884) on top of the fix memory's advanced retrieval.

    crag = CorrectiveRAG(memory, search=TavilySearch())
    result = crag.retrieve(stage, error_tail)          # -> CragResult
    text = result.render()                              # -> {{ retrieved_text }} in prompts/diagnose_team_v2.md

The paper's steps, as implemented here:
1. Retrieve: the advanced retriever's top 3 (BM25 + vectors, reciprocal rank fusion, stage boost).
2. Retrieval evaluator: a lightweight scorer gives each retrieved case a relevance score in [0, 1] against the
   current error. Default "rules" (deterministic, no model call, like the paper's small T5 evaluator): same
   exception class, same packages/modules named, and word overlap of the error signatures. Option "llm": the
   diagnosis LLM grades relevance as strict JSON.
3. Action, from two thresholds (the paper's upper/lower):
   - Correct    (at least one case >= UPPER): keep the relevant cases, refined (step 4); no web search.
   - Incorrect  (every case < LOWER): discard them all, rewrite the query, web search, refine the results.
   - Ambiguous  (otherwise): keep the cases >= LOWER, refined, AND add refined web results.
4. Knowledge refinement (decompose-then-recompose): each source is split into strips (a memory case into its
   error / verified fix / cause; a web page into sentences), each strip is scored against the query, the
   irrelevant strips are dropped, and the rest are put back together in their original order.
5. Query rewriting for the web: the error signature without paths, addresses, line numbers and noise, plus
   the language and the word "fix" (deterministic; the paper rewrites with an LLM).
Web results are cached by query (memory/web_cache/): mode "cache" searches only on a miss, "replay" never
searches (evaluations: the same results every run), "off" disables the web. Web knowledge is marked
"not verified": only fixes the build confirmed count as verified (memory/learned.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from pydantic import BaseModel, Field

from ..faults import error_signature
from . import FixMemory, Hit

UPPER = 0.6                      # at least one case this relevant: Correct
LOWER = 0.35                     # every case below this: Incorrect
MAX_WEB_RESULTS = 5
MAX_WEB_STRIPS = 6               # sentences kept from the web, all results together
MAX_STRIPS_PER_PAGE = 3          # so more than one source is heard
CACHE_DIR = Path(os.environ.get("CRAG_CACHE_DIR") or Path(__file__).resolve().parents[2] / "memory" / "web_cache")

_EXC = re.compile(r"\b([A-Z]\w*(?:Error|Exception|Exit))\b")
_QUOTED = re.compile(r"""['"`]([A-Za-z_][\w.\-]*)['"`]""")
_MODULE = re.compile(r"""\b(?:from|module|package|named|for|import)\s+['"]?([A-Za-z_][\w.\-]*)""")
_DOTTED = re.compile(r"\b([a-z_][a-z0-9_]*)\.[a-z_][\w.]*\b")          # mlflow.exceptions.MlflowException
_SHARED_LIB = re.compile(r"\b(lib[\w+\-]+?)\.so\b")                       # libffi.so.8 -> libffi
_KEYWORDS = {"named", "module", "package", "from", "import", "for", "it", "name", "cannot", "object"}
_WORD = re.compile(r"[a-z][a-z0-9_]+")
_NOISE_WORDS = {"the", "a", "an", "of", "to", "in", "is", "for", "with", "and", "or", "from", "file", "line",
                "error", "errors", "no", "not", "at", "on", "by", "be", "this", "that", "was", "were", "py", "usr",
                "lib", "site", "packages", "python3", "python", "local", "home", "self", "none", "true", "false"}
_PATH = re.compile(r"(/[\w.\-]+)+/?|[A-Za-z]:\\[\w\\.\-]+")
_HEX = re.compile(r"\b0x[0-9a-f]+\b|\b[0-9a-f]{12,}\b", re.I)


# --- the evaluator ---------------------------------------------------------------------------------

def exception_class(text: str) -> str | None:
    m = _EXC.search(text)
    return m.group(1) if m else None


def entities(text: str) -> set[str]:
    """Package / module / name tokens the error names ('markupsafe', 'soft_unicode', 'mlflow', 'libffi', ...)."""
    names = set(_QUOTED.findall(text)) | set(_MODULE.findall(text)) | set(_SHARED_LIB.findall(text))
    names |= {m for m in _DOTTED.findall(text) if not m.isdigit()}
    out = set()
    for n in names:
        n = n.lower().strip(".")
        if n.endswith(".so") or ".so." in n:
            n = n.split(".so")[0]
        if len(n) > 1 and n not in _NOISE_WORDS | _KEYWORDS and not n.endswith((".txt", ".py")):
            out.add(n)
            out.add(n.split(".")[0])
    return out


def words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _NOISE_WORDS and not w.isdigit()}


def relevance(query: str, doc_text: str) -> tuple[float, dict]:
    """Rules evaluator: 0.25 same exception class + 0.5 shared named packages/modules + 0.25 word overlap.
    An error that names no package (e.g. a server message) is judged on its words: 0.25 class + 0.75 overlap."""
    qe, de = entities(query), entities(doc_text)
    qw, dw = words(query), words(doc_text)
    same_class = exception_class(query) is not None and exception_class(query) == exception_class(doc_text)
    shared = qe & de
    ent = len(shared) / len(qe) if qe else 0.0
    jac = len(qw & dw) / len(qw | dw) if qw | dw else 0.0
    overlap = min(1.0, 2 * jac)
    score = 0.25 * same_class + (0.5 * min(1.0, ent) + 0.25 * overlap if qe else 0.75 * overlap)
    return round(score, 3), {"same_exception": same_class, "shared_names": sorted(shared), "word_overlap": round(jac, 3)}


class Graded(BaseModel):
    id: str
    score: float
    why: dict
    kept: bool = False
    strips: list[str] = Field(default_factory=list)


class LlmEvaluator:
    """Option: the LLM grades each case (strict JSON). Same interface as the rules evaluator."""
    SCHEMA = {"type": "object", "properties": {"relevant": {"type": "string", "enum": ["yes", "partly", "no"]},
                                               "why": {"type": "string"}}, "required": ["relevant", "why"]}
    SCORES = {"yes": 1.0, "partly": 0.5, "no": 0.0}

    def __init__(self, client):
        self.client = client

    def __call__(self, query: str, doc_text: str) -> tuple[float, dict]:
        msg = [{"role": "system", "content": "You grade whether a past build failure is relevant to the current "
                                             "one: same cause, so its fix could apply. Answer as JSON."},
               {"role": "user", "content": f"CURRENT ERROR:\n{query}\n\nPAST FAILURE:\n{doc_text}\n\n"
                                           'Answer {"relevant": "yes" | "partly" | "no", "why": "..."}.'}]
        try:
            ans = json.loads(self.client.chat(msg, schema=self.SCHEMA).content)
            return self.SCORES.get(ans.get("relevant"), 0.0), {"llm": ans.get("relevant"), "why": ans.get("why", "")}
        except Exception as e:  # noqa: BLE001 - grading failed: fall back to the rules for this case
            score, why = relevance(query, doc_text)
            return score, why | {"llm_error": f"{type(e).__name__}"}


# --- knowledge refinement (decompose-then-recompose) -------------------------------------------------

def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 20]


def refine_case(hit: Hit, query: str) -> list[str]:
    """A memory case as strips: the error and the verified fix are kept; the cause only if it names what the
    current error names (an unrelated cause is what the model copied into its reasons, docs/RESULTS.md 4-6)."""
    d = hit.doc
    strips = [f"stage {d.stage}, error: {d.error}", f"verified fix: {d.fix_text}"]
    if entities(d.cause) & entities(query):
        strips.insert(1, f"cause: {d.cause}")
    return strips


# a strip that explains or fixes (the knowledge we want), and one that only echoes code or a traceback
_KNOWLEDGE = re.compile(r"\b(removed|deprecated|renamed|replaced|introduced|because|caused|requires?|compatible|"
                        r"incompatib\w*|downgrad\w*|upgrad\w*|pin(ned|ning)?|install(ed|ing)?|version|instead|"
                        r"solution|solve[sd]?|fix(e[sd])?|works?)\b", re.I)
_ECHO = re.compile(r"^\s*(from\s+\S+\s+import|import\s|File\s+\"|Traceback|\w*(Error|Exception):|[`>$#])|"
                   r"site-packages|\.py\"?,? line \d+")


def strip_score(sentence: str, query: str) -> float:
    """How much a web sentence adds: it names what the error names AND says something about cause or fix.
    A sentence that only repeats the error (import line, traceback, the error message itself) scores 0."""
    if _ECHO.search(sentence):
        return 0.0
    sw, qw = words(sentence), words(query)
    if qw and len(sw & qw) / len(sw or {1}) > 0.8:
        return 0.0                                   # nearly all its words are the error's own words
    named = len(entities(query) & (entities(sentence) | sw))
    knowledge = len(_KNOWLEDGE.findall(sentence))
    return named * (1 + min(knowledge, 3)) if knowledge else 0.0


def refine_web(results: list[dict], query: str, max_strips: int = MAX_WEB_STRIPS) -> list[dict]:
    """Web pages split into sentences (decompose), scored by strip_score (filter), best kept and put back in
    page order (recompose)."""
    scored = []
    for r_i, r in enumerate(results):
        for s_i, s in enumerate(_sentences(r.get("content", ""))):
            score = strip_score(s, query)
            if score > 0:
                scored.append((score, r_i, s_i, s, r.get("url", "")))
    best, per_page = [], {}
    for item in sorted(scored, key=lambda t: -t[0]):
        if per_page.get(item[1], 0) < MAX_STRIPS_PER_PAGE and len(best) < max_strips:
            best.append(item)
            per_page[item[1]] = per_page.get(item[1], 0) + 1
    return [{"text": s, "url": u} for _, _, _, s, u in sorted(best, key=lambda t: (t[1], t[2]))]


# --- query rewriting and web search ------------------------------------------------------------------

def rewrite_query(error_tail: list[str] | str, language: str = "python") -> str:
    sig = error_signature(error_tail if isinstance(error_tail, list) else error_tail.splitlines())
    sig = _HEX.sub(" ", _PATH.sub(" ", sig))
    sig = re.sub(r"\(\s*\)|line \d+|\s+", " ", sig).strip(" :")
    return f"{language} {sig} fix"[:300]


class TavilySearch:
    """Tavily web search (TAVILY_API_KEY in the environment or .env), results cached by query."""
    URL = "https://api.tavily.com/search"

    def __init__(self, mode: str | None = None, cache_dir: Path | None = None, timeout: float = 30):
        from ..llm.api import load_dotenv
        load_dotenv()
        self.mode = mode or os.environ.get("CRAG_SEARCH", "cache")        # cache | replay | off
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.timeout = timeout
        self.key = os.environ.get("TAVILY_API_KEY", "")

    def _path(self, query: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(query.encode()).hexdigest()[:16]}.json"

    def __call__(self, query: str) -> tuple[list[dict], str]:
        """(results, where they came from: cache / web / why there are none)."""
        if self.mode == "off":
            return [], "web search off"
        path = self._path(query)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["results"], "cache"
        if self.mode == "replay":
            return [], "not in the replay cache"
        if not self.key:
            return [], "no TAVILY_API_KEY"
        body = json.dumps({"query": query, "max_results": MAX_WEB_RESULTS, "search_depth": "basic"}).encode()
        req = urllib.request.Request(self.URL, data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            return [], f"web search failed ({type(e).__name__}: {str(e)[:120]})"
        results = [{"title": x.get("title", ""), "url": x.get("url", ""), "content": x.get("content", "")}
                   for x in data.get("results", [])]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"query": query, "results": results}, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return results, "web"


# --- CRAG --------------------------------------------------------------------------------------------

class CragResult(BaseModel):
    action: str                                  # correct | incorrect | ambiguous
    graded: list[Graded]
    query: str | None = None                     # the rewritten web query (incorrect / ambiguous)
    web: list[dict] = Field(default_factory=list)   # refined strips {text, url}
    web_source: str | None = None                # cache / web / why none

    def kept(self) -> list[Graded]:
        return [g for g in self.graded if g.kept]

    def render(self) -> str:
        parts = []
        kept = self.kept()
        if kept:
            parts.append("Past failures our build verified (relevant to this error):")
            for n, g in enumerate(kept, 1):
                parts.append(f"{n}. " + "\n   ".join(g.strips))
        else:
            parts.append("Past failures: none relevant to this error (the retrieved ones were discarded).")
        if self.web:
            parts.append("External knowledge from a web search (NOT verified by our build: hints only):")
            parts += [f"- {s['text']} ({s['url']})" for s in self.web]
        elif self.action != "correct":
            parts.append(f"External knowledge: none ({self.web_source}).")
        return "\n".join(parts)


class CorrectiveRAG:
    def __init__(self, memory: FixMemory, search=None, evaluator=None, upper: float = UPPER, lower: float = LOWER):
        self.memory = memory
        self.search = search
        self.evaluator = evaluator or relevance
        self.upper, self.lower = upper, lower

    def retrieve(self, stage: str, error_tail: list[str] | str) -> CragResult:
        tail = error_tail if isinstance(error_tail, list) else error_tail.splitlines()
        query = error_signature(tail)
        hits = self.memory.advanced(stage, tail)
        graded = []
        for h in hits:
            score, why = self.evaluator(query, f"{h.doc.error}\n{h.doc.cause}")
            graded.append(Graded(id=h.doc.id, score=score, why=why))
        if any(g.score >= self.upper for g in graded):
            action = "correct"
        elif all(g.score < self.lower for g in graded):
            action = "incorrect"
        else:
            action = "ambiguous"
        keep_at = self.upper if action == "correct" else self.lower
        for g, h in zip(graded, hits):
            if action != "incorrect" and g.score >= keep_at:
                g.kept, g.strips = True, refine_case(h, query)
        result = CragResult(action=action, graded=graded)
        if action != "correct" and self.search is not None:
            result.query = rewrite_query(tail)
            raw, result.web_source = self.search(result.query)
            result.web = refine_web(raw, query)
        elif action != "correct":
            result.web_source = "no web search configured"
        return result
