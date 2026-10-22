"""RAG metrics of team mode (retrieval recall@3 and the CRAG evaluator), from the errors the LLM really sees.

    docker compose -f docker-compose.agents.yml run --rm setup python -m agents.setup.eval.rag_metrics

No LLM and no web search: a scripted client answers with the verified fixes of each case (GROUND_TRUTH), so the
fix loop walks through the same errors an LLM config meets when it gets each step right (e.g. pandas 1.5.3 first,
then the numpy binary error). At every diagnosis the Corrective RAG step is recorded: the advanced retriever's
top 3, the evaluator's score of each, kept or not, and the action. Embeddings: the agent's embedder (Ollama
nomic-embed-text), as in team_eval. CRAG_SEARCH=off: the web is not needed to grade the memory.

Relevance, a rule fixed before computing anything: a memory case is relevant to an error when its verified fix
acts on a package the error is about (the packages of team_eval.EXPECTED for that case that the error output
names) or when its error line is the same message. A case with no relevant memory document is graded by
whether CRAG correctly sent it to the web.

Output: eval/results/rag_metrics.json (every diagnosis) and rag_metrics.md (the tables).
"""
import json
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("CRAG_SEARCH", "off")

from agents.setup.eval import team_eval as te  # noqa: E402

HERE = Path(__file__).resolve().parent
PIN = lambda name, version: {"action": "pin_package", "name": name, "version": version}  # noqa: E731
GIVE_UP = {"action": "give_up", "reason": "the project's to fix"}
# the verified fixes, in order (the ground truth of team_eval; rules take the steps they know before the LLM)
GROUND_TRUTH = {
    "unpinned_version": [PIN("scikit-learn", "1.5.2")],
    "broken_import": [GIVE_UP],
    "removed_numpy_alias": [PIN("numpy", "1.23.5")],
    "removed_pandas_append": [PIN("pandas", "1.5.3"), PIN("numpy", "1.26.4")],
    "data_dir_path": [{"action": "fix_path", "target": "workdir", "path": "data"}],
    "removed_scipy_simps": [PIN("scipy", "1.13.1")],
    "removed_numpy_product": [PIN("numpy", "1.26.4")],
    "missing_data_file": [GIVE_UP],
    "scipy_misc_face": [PIN("scipy", "1.13.1")],
    "pandas_read_csv_squeeze": [PIN("pandas", "1.5.3"), PIN("numpy", "1.26.4")],
    "sklearn_bagging_base_estimator": [PIN("scikit-learn", "1.3.2"), PIN("numpy", "1.26.4")],
    "numpy_NaN": [PIN("numpy", "1.26.4")],
    "pandas_df_iteritems": [PIN("pandas", "1.5.3"), PIN("numpy", "1.26.4")],
    "sklearn_load_boston": [PIN("scikit-learn", "1.1.3"), PIN("numpy", "1.26.4")],
}
# the cases where the LLM is asked at all (the other v1 faults are fixed by rules first)
SETS = ["v1", "heldout", "heldout_v3", "final"]
LLM_FAULTS = set(GROUND_TRUTH)


class Scripted:
    model, options, digest = "scripted-ground-truth", {}, "none"

    def __init__(self, answers):
        self.answers = list(answers)

    def chat(self, messages, schema=None):
        from agents.setup.builder_agent.llm.ollama import ChatResponse
        fix = self.answers.pop(0) if self.answers else GIVE_UP
        return ChatResponse(content=json.dumps({"fix": fix, "reason": "ground truth"}), latency_s=0.0)


def fix_package(doc) -> str | None:
    name = (doc.fix or {}).get("name")
    return te_normalize(name) if name else None


def te_normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def relevant(doc, error_text: str, about: set[str]) -> bool:
    if fix_package(doc) in about:
        return True
    line = doc.error.strip().splitlines()[0][:60] if doc.error.strip() else ""
    return bool(line) and line in error_text


def run_case(repo: str, fault: str, memory) -> list[dict]:
    from agents.setup.agent import SetupAgent
    from agents.setup.builder_agent.memory.crag import CorrectiveRAG
    from agents.setup.builder_agent.team.fix import error_text, packages_in_error
    records = []

    class Recorder(SetupAgent):
        def crag(self):
            rag = CorrectiveRAG(memory, search=None)
            retrieve = rag.retrieve

            def recorded(stage, tail):
                result = retrieve(stage, tail)
                text = "\n".join(tail if isinstance(tail, list) else tail.splitlines())
                expected = te.EXPECTED_FOR.get((repo, fault), te.EXPECTED[fault])["packages"]
                versions = {te_normalize(p): "" for p in expected}
                about = set(packages_in_error(text, versions)) & {te_normalize(p) for p in expected}
                docs = dict(memory.docs)
                query = __import__("agents.setup.builder_agent.memory.crag", fromlist=["error_signature"]) \
                    .error_signature(tail if isinstance(tail, list) else tail.splitlines())
                all_scores = {d.id: rag.evaluator(query, f"{d.error}\n{d.cause}")[0] for d in memory.docs.values()}
                rel = sorted(i for i, d in docs.items() if relevant(d, error_text(text) or text, about))
                records.append({"repo": repo, "fault": fault, "stage": stage, "about": sorted(about),
                                "relevant": rel, "action": result.action,
                                "top3": [{"id": g.id, "score": round(g.score, 3), "kept": g.kept,
                                          "relevant": g.id in rel} for g in result.graded],
                                "all_scores": {k: round(v, 3) for k, v in all_scores.items()},
                                "error": last_error_line(error_text(text) or text)})
                return result
            rag.retrieve = recorded
            return rag

    with tempfile.TemporaryDirectory(prefix="rag-metrics-") as tmp:
        root = te.copy_repo(repo, Path(tmp))
        te.inject(root, repo, fault)
        agent = Recorder(root=str(root), commit=False, learn=False, memory=memory, crag=True,
                         client=Scripted(GROUND_TRUTH[fault]), diagnosis_client=Scripted(GROUND_TRUTH[fault]),
                         memory_dir=str(Path(tmp) / "m"))
        agent.step()
    return records


def last_error_line(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("make")]
    errs = [ln for ln in lines if re.match(r"[\w.]*(Error|Exception|Warning)\b", ln)]
    return (errs or lines or [""])[-1][:200]


def wilson(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    import math
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100 * k / n:.0f}% ({k}/{n}) [{100 * (c - h):.0f}–{100 * (c + h):.0f}%]"


def tables(recs: list[dict]) -> str:
    out = ["# RAG metrics of team mode (from `rag_metrics.py`)", "",
           f"{len(recs)} diagnoses (every LLM step of the evaluation cases, walked with the verified fixes). "
           "Intervals: Wilson 95%, for reference (diagnoses of one fault are not independent).", ""]
    with_rel = [r for r in recs if r["relevant"]]
    hit = [r for r in with_rel if any(g["relevant"] for g in r["top3"])]
    out += ["## Retrieval (advanced retriever, top 3)", "",
            f"- Diagnoses with at least one relevant memory case: {len(with_rel)} of {len(recs)}",
            f"- **Recall@3** (a relevant case in the top 3): {wilson(len(hit), len(with_rel))}", ""]
    pairs = [g for r in recs for g in r["top3"]]
    tp = sum(g["kept"] and g["relevant"] for g in pairs)
    fp = sum(g["kept"] and not g["relevant"] for g in pairs)
    fn = sum(not g["kept"] and g["relevant"] for g in pairs)
    tn = sum(not g["kept"] and not g["relevant"] for g in pairs)
    out += ["## CRAG evaluator: keep or discard each retrieved case", "",
            "| | relevant | not relevant |", "|---|---|---|",
            f"| kept | {tp} | {fp} |", f"| discarded | {fn} | {tn} |", "",
            f"- Accuracy: {wilson(tp + tn, len(pairs))}",
            f"- Precision of the kept cases: {wilson(tp, tp + fp)}",
            f"- Recall of the relevant retrieved cases: {wilson(tp, tp + fn)}", ""]
    ideal = lambda r: "correct" if any(g["relevant"] for g in r["top3"]) else "incorrect"  # noqa: E731
    out += ["## CRAG action vs the ideal action", "",
            "Ideal: **correct** when a relevant case was retrieved, else **incorrect** (go to the web). "
            "Ambiguous uses both, so it is counted as acceptable either way.", "",
            "| ideal \\ chosen | correct | ambiguous | incorrect |", "|---|---|---|---|"]
    for i in ("correct", "incorrect"):
        rs = [r for r in recs if ideal(r) == i]
        out.append(f"| {i} | " + " | ".join(str(sum(r["action"] == a for r in rs)) for a in
                                             ("correct", "ambiguous", "incorrect")) + " |")
    exact = sum(r["action"] == ideal(r) for r in recs)
    ok = sum(r["action"] in (ideal(r), "ambiguous") for r in recs)
    out += ["", f"- Exact action: {wilson(exact, len(recs))}",
            f"- Acceptable action (exact or ambiguous): {wilson(ok, len(recs))}", ""]
    out += ["## Every diagnosis", "", "| Repo | Fault | Error | Relevant in memory | Top 3 (score, kept) | Action |",
            "|---|---|---|---|---|---|"]
    for r in recs:
        top = ", ".join(f"{g['id']}{'*' if g['relevant'] else ''} {g['score']}{' kept' if g['kept'] else ''}"
                        for g in r["top3"])
        out.append(f"| {r['repo']} | {r['fault']} | {r['error'][:70]} | {', '.join(r['relevant']) or '-'} | {top} "
                   f"| {r['action']} |")
    return "\n".join(out) + "\n\n`*` = relevant by the rule.\n"


def main() -> int:
    from agents.setup.builder_agent.llm.api import make_embedder
    from agents.setup.builder_agent.memory import FixMemory
    from agents.setup.builder_agent.memory.learned import memory_docs
    memory = FixMemory.build(make_embedder(), docs=memory_docs("base"), keywords_if_unreachable=True)
    cases = sorted({c for s in SETS for c in te.SETS[s] if c[1] in LLM_FAULTS})
    out = HERE / "results"
    path = out / "rag_metrics.json"
    recs = json.loads(path.read_text()) if path.exists() else []
    done = {(r["repo"], r["fault"]) for r in recs}
    for repo, fault in cases:
        if (repo, fault) in done:
            continue
        print(f"{repo} / {fault} ...", flush=True)
        new = run_case(repo, fault, memory)
        recs += new or [{"repo": repo, "fault": fault, "skipped": "no diagnosis (rules or no error)"}]
        path.write_text(json.dumps(recs, indent=1), encoding="utf-8")
    real = [r for r in recs if "skipped" not in r]
    (out / "rag_metrics.md").write_text(tables(real), encoding="utf-8")
    print(tables(real))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
