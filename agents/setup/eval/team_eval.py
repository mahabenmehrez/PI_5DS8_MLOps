"""Team-mode evaluation (roadmap 7.6 faults, roadmap 7.4 W4 measurement): the Setup Agent on the 3 repos with
injected faults, in 4 configurations, real dry runs. Run it in the team image (make, Linux), from the repo root:

    docker compose -f docker-compose.agents.yml run --rm setup python -m agents.setup.eval.team_eval
    python -m agents.setup.eval.team_eval --configs rules --cases clean,runtime_dependency     # a subset

Cases: 3 repos (wine = the team project, sk_root, sk_src) x 5 faults (clean control + 4 roadmap faults).
Configs: rules (no LLM), llm_plain (LLM, no memory), llm_memory (LLM + Advanced RAG over the base memory),
llm_crag (LLM + Corrective RAG with web search), llm_sft (llm_crag with the fine-tuned model for the diagnosis,
BUILDER_DIAGNOSIS_MODEL or qwen-builder-sft-v5 on Ollama). The LLM configs other than llm_sft diagnose with the
base model (qwen2.5-coder:7b on Ollama, or BUILDER_OLLAMA_MODEL), so .env can't change them. The slots (decide
step) use the agent's slots LLM in every LLM config. No write-back to memory (every run starts from the same
memory); CRAG web results are cached (CRAG_SEARCH=cache) so a rerun replays them.

A run is a success when the agent does the right thing: builds when the fault is fixable, and reports
(pipeline.setup_failed) without installing anything unrelated when it is not (a bug in the team's code).
Results: agents/setup/eval/results/<name>.jsonl (one line per run) and <name>.md (summary tables).
"""
import argparse
import json
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPOS = HERE.parent / "test_repos"
WINE_PARTS = ["baseline", "samples", "data", "shared", "contracts", "dry_run_jobs", "tests", "agents"]
CONFIGS = ["rules", "llm_plain", "llm_memory", "llm_crag", "llm_sft", "llm_cascade"]
# llm_cascade = llm_sft + the API model when the fine-tuned model gives up, gives no valid answer, or its fix changes
# nothing (same stage and error) or breaks an earlier stage: it takes the remaining attempts (same budget of 3)
ESCALATION_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
SFT_MODEL = "qwen-builder-sft-v5"     # the fine-tuned Qwen (QLoRA), served by Ollama
FAULTS = ["clean", "runtime_dependency", "unpinned_version", "wrong_data_path", "broken_import"]
# v2: faults written AFTER the v2 fixes (never used to tune them): do the fixes generalise?
HELDOUT = ["removed_numpy_alias", "removed_pandas_append", "data_dir_path"]     # SEEN since v2's run (v3 fixes)
# v3: written AFTER the v3 fixes, never used to tune them
HELDOUT_V3 = ["removed_scipy_simps", "removed_numpy_product", "missing_data_file"]     # SEEN since v3's run (v4)
# FINAL (external): APIs the libraries' own release notes list as removed, turned into one-line uses (eval/
# external_pool.py, 33 candidates; np.bool dropped: it exists again in numpy 2). 6 drawn at random with the seed
# 20261021, before any run, with the agent frozen after v4. A case whose fault doesn't fail the build (no error
# under rules, 0 fixes) is invalid and left out of the score: decided before the run.
from agents.setup.eval.external_pool import FINAL_SEED, cascade_faults, final_faults  # noqa: E402
FINAL = final_faults()
CASCADE = cascade_faults()        # second unseen draw (seed 20261022), for the cascade test
BASE_REPOS = ["wine", "sk_root", "sk_src"]
SLOT_REPOS = ["sk_two_scripts", "sk_legacy"]      # the decide step can't settle the training script / target
# which (repo, fault) pairs each set runs; v1 is the set of team_eval_v1, unchanged
SETS = {"v1": [(r, f) for r in BASE_REPOS for f in FAULTS],
        "heldout": [(r, f) for r in BASE_REPOS for f in HELDOUT],
        "slots": [(r, "clean") for r in SLOT_REPOS],
        "heldout_v3": [(r, f) for r in BASE_REPOS for f in HELDOUT_V3],
        "final": [(r, f) for r in BASE_REPOS for f in FINAL],
        "cascade": [(r, f) for r in BASE_REPOS for f in CASCADE]}
# the right choices of the slot repos (ground truth written with the repos): a build with another choice fails
SLOT_TRUTH = {"sk_two_scripts": {"train_script": "train_model.py", "target": "species"},
              "sk_legacy": {"train_script": "train_v2.py"}}

# what each fault expects: fixable or not, and the packages a fix may install
EXPECTED = {
    "clean": {"fixable": True, "packages": set()},
    "runtime_dependency": {"fixable": True, "packages": {"emoji"}},
    "unpinned_version": {"fixable": True, "packages": {"scikit-learn"}},
    "wrong_data_path": {"fixable": True, "packages": set()},
    "broken_import": {"fixable": False, "packages": set()},
    # np.float: removed in numpy 1.24. pandera (the agent's own schema check) refuses numpy < 1.24.4 at import,
    # so an older pandera is part of the fix (checked by hand: numpy==1.23.5 + pandera==0.20.4 builds sk_root)
    "removed_numpy_alias": {"fixable": True, "packages": {"numpy", "pandera"}},
    # DataFrame.append: removed in pandas 2.0; pandas 1.5.3 needs numpy < 2, and pandera an older version
    # (checked by hand: pandas==1.5.3 + numpy==1.26.4 + pandera==0.20.4 builds sk_root, exactly 3 fixes)
    "removed_pandas_append": {"fixable": True, "packages": {"pandas", "numpy", "pandera"}},
    "data_dir_path": {"fixable": True, "packages": set()},
    "removed_scipy_simps": {"fixable": True, "packages": {"scipy"}},       # scipy.integrate.simps: gone in 1.14
    "removed_numpy_product": {"fixable": True, "packages": {"numpy"}},     # np.product: gone in numpy 2.0
    "missing_data_file": {"fixable": False, "packages": set()},           # the file is nowhere: report
}
# the final set (ground truth checked with the pins applied directly, no agent: sk_root, and wine / sk_src where
# noted): scipy.misc.face -> scipy==1.13.1 (sk_root, wine); read_csv(squeeze) and DataFrame.iteritems ->
# pandas==1.5.3 + numpy==1.26.4 (sk_root); BaggingClassifier(base_estimator) -> scikit-learn==1.3.2 + numpy==1.26.4
# (sk_root, sk_src); np.NaN -> numpy==1.26.4 (sk_root); load_boston -> scikit-learn==1.1.3 (+ numpy==1.26.4 on
# sk_root; alone on wine)
EXPECTED.update({
    "scipy_misc_face": {"fixable": True, "packages": {"scipy"}},
    "pandas_read_csv_squeeze": {"fixable": True, "packages": {"pandas", "numpy", "pandera"}},
    "sklearn_bagging_base_estimator": {"fixable": True, "packages": {"scikit-learn", "numpy"}},
    "numpy_NaN": {"fixable": True, "packages": {"numpy"}},
    "pandas_df_iteritems": {"fixable": True, "packages": {"pandas", "numpy", "pandera"}},
    "sklearn_load_boston": {"fixable": True, "packages": {"scikit-learn", "numpy"}},
})
assert set(FINAL) <= set(EXPECTED), "the final draw changed: its ground truth must be checked again"
# the cascade set (checked with the pins applied directly: sk_root, and wine for binom_test / int alias)
EXPECTED.update({
    "scipy_trapz": {"fixable": True, "packages": {"scipy"}},                       # scipy==1.13.1
    "pandas_error_bad_lines": {"fixable": True, "packages": {"pandas", "numpy", "pandera"}},  # 1.5.3 + 1.26.4
    "scipy_binom_test": {"fixable": True, "packages": {"scipy", "numpy"}},         # 1.11.4 (+ numpy 1.26.4)
    "numpy_unicode_": {"fixable": True, "packages": {"numpy"}},                    # 1.26.4
    "numpy_int_alias": {"fixable": True, "packages": {"numpy", "pandera"}},        # 1.23.5
    "numpy_asfarray": {"fixable": True, "packages": {"numpy"}},                    # 1.26.4
})
assert set(CASCADE) <= set(EXPECTED), "the cascade draw changed: its ground truth must be checked again"
# wine's requirements pin pandas==2.2.3: an older pandas conflicts with the team's pin, so the agent must report
# (checked: pandas==1.5.3 on wine -> ResolutionImpossible at install)
EXPECTED_FOR = {(("wine", f)): {"fixable": False, "packages": {"pandas", "numpy", "pandera"}}
                for f in ("removed_pandas_append", "pandas_read_csv_squeeze", "pandas_df_iteritems",
                          "pandas_error_bad_lines")}
TRAIN = {"wine": "baseline/train.py", "sk_root": "train.py", "sk_src": "src/irisclf/train.py",
         "sk_two_scripts": "train_model.py", "sk_legacy": "train_v2.py"}
DATA_PATH = {"wine": ('data_path = sys.argv[1] if len(sys.argv) > 1 else "data/winequality-red.csv"',
                      'data_path = "winequality-red.csv"'),
             "sk_root": ('"data/iris.csv"', '"iris.csv"'), "sk_src": ('"data/iris.csv"', '"iris.csv"')}
MISSING_PATH = {"wine": ('data_path = sys.argv[1] if len(sys.argv) > 1 else "data/winequality-red.csv"',
                         'data_path = "data/winequality-white.csv"'),
                "sk_root": ('"data/iris.csv"', '"data/iris_v2.csv"'), "sk_src": ('"data/iris.csv"', '"data/iris_v2.csv"')}
TYPO = {"wine": ("import ElasticNet", "import ElasticNett"),
        "sk_root": ("import LogisticRegression", "import LogisticRegresion"),
        "sk_src": ("import RandomForestClassifier", "import RandomForestClassifer")}


def copy_repo(name: str, dst: Path) -> Path:
    repo = dst / name
    if name == "wine":
        repo.mkdir()
        for part in WINE_PARTS:
            shutil.copytree(ROOT / part, repo / part, ignore=shutil.ignore_patterns(
                "__pycache__", "logs", "memory", ".cache", ".tools", "mlruns", "results"))
        for f in ("requirements.txt", ".gitignore"):
            shutil.copy(ROOT / f, repo / f)
    else:
        shutil.copytree(REPOS / name, repo)
    return repo


def prepend(path: Path, code: str) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    at = 1 if lines and lines[0].startswith('"""') and lines[0].count('"""') == 2 else 0
    if lines and lines[0].startswith('"""') and lines[0].count('"""') == 1:        # skip a multi-line docstring
        at = next(i for i, line in enumerate(lines[1:], 1) if '"""' in line) + 1
    while at < len(lines) and lines[at].startswith("from __future__ import"):    # must stay first (team train.py)
        at += 1
    path.write_text("".join(lines[:at]) + code + "".join(lines[at:]), encoding="utf-8")


def inject(repo: Path, name: str, fault: str) -> None:
    train = repo / TRAIN[name]
    if fault == "runtime_dependency":       # loaded by name at run time: the scan can't see it
        prepend(train, 'import importlib\nimportlib.import_module("emoji")\n')
    elif fault == "unpinned_version":       # `squared` was removed from mean_squared_error in scikit-learn 1.6
        prepend(train, "from sklearn.metrics import mean_squared_error as _mse\n"
                       "_mse([1.0, 2.0], [1.0, 2.5], squared=False)\n")
    elif fault == "wrong_data_path":
        if name == "wine":                  # reads a root-relative path, but the Makefile runs from baseline/
            text = train.read_text(encoding="utf-8")
            text = re.sub(r'data_path = sys\.argv\[1\] if len\(sys\.argv\) > 1 else "data/winequality-red\.csv"',
                          'data_path = "data/winequality-red.csv"', text)
            train.write_text(text, encoding="utf-8")
        elif name == "sk_root":             # the script expects to run from its own folder
            (repo / "model").mkdir()
            shutil.copy(repo / "data/iris.csv", repo / "model/iris.csv")
            (repo / "model/train.py").write_text(train.read_text().replace("data/iris.csv", "iris.csv"))
            train.unlink()
        else:                               # sk_src: a path relative to the script's folder
            train.write_text(train.read_text().replace('"data/iris.csv"', '"../../data/iris.csv"'))
    elif fault == "removed_numpy_alias":    # np.float was removed in numpy 1.24: fix numpy<1.24 (e.g. 1.23.5)
        prepend(train, "import numpy as _np\n_np.float(1.0)\n")
    elif fault == "removed_pandas_append":  # DataFrame.append was removed in pandas 2.0: pandas<2 (+ numpy<2)
        prepend(train, "import pandas as _pd\n_pd.DataFrame({'a': [1]}).append(_pd.DataFrame({'a': [2]}))\n")
    elif fault == "data_dir_path":          # reads the bare file name; the file is in data/: run train from data/
        old, new = DATA_PATH[name]
        text = train.read_text(encoding="utf-8")
        assert old in text, (name, old)
        train.write_text(text.replace(old, new, 1), encoding="utf-8")
    elif fault == "removed_scipy_simps":    # scipy.integrate.simps was removed in scipy 1.14: scipy<1.14
        prepend(train, "from scipy.integrate import simps as _simps\n_simps([1.0, 2.0, 3.0])\n")
    elif fault == "removed_numpy_product":  # np.product was removed in numpy 2.0: numpy<2
        prepend(train, "import numpy as _np\n_np.product([1, 2])\n")
    elif fault == "missing_data_file":      # the code reads a file that is nowhere in the repo: the team's to fix
        old, new = MISSING_PATH[name]
        text = train.read_text(encoding="utf-8")
        assert old in text, (name, old)
        train.write_text(text.replace(old, new, 1), encoding="utf-8")
    elif fault in FINAL or fault in CASCADE:   # the external faults: prepend the one-line use of the removed API
        from agents.setup.eval.external_pool import code_of
        prepend(train, code_of(fault))
    elif fault == "broken_import":          # a typo in the team's own code: not the agent's to fix
        old, new = TYPO[name]
        train.write_text(train.read_text().replace(old, new, 1))


def make_agent(config: str, repo: Path, memdir: Path):
    from agents.setup.agent import SetupAgent
    from agents.setup.builder_agent.llm.api import make_embedder
    from agents.setup.builder_agent.memory import FixMemory
    from agents.setup.builder_agent.memory.learned import memory_docs
    from agents.setup.builder_agent.llm.api import make_client
    common = dict(root=str(repo), commit=False, learn=False, memory_dir=str(memdir))
    if config == "rules":
        return SetupAgent(client=None, diagnosis_client=None, crag=False, memory=None, **common)
    if config in ("llm_sft", "llm_cascade"):
        sft = make_client(os.environ.get("BUILDER_DIAGNOSIS_MODEL") or SFT_MODEL, provider="ollama")
        sft.digest                       # fails here when the fine-tuned model isn't in Ollama: no silent base model
    else:
        sft = None
    common["diagnosis_client"] = sft or make_client(provider="ollama")     # base model: the same in every config
    common["escalation_client"] = None                                      # no cascade unless asked
    if config == "llm_cascade":                 # fails here without an NVIDIA key: no silent cascade-less run
        common["escalation_client"] = make_client(os.environ.get("BUILDER_ESCALATION_MODEL") or ESCALATION_MODEL,
                                                  provider="api")
    if config == "llm_plain":
        return SetupAgent(crag=False, memory=None, **common)
    memory = FixMemory.build(make_embedder(), docs=memory_docs("base"), keywords_if_unreachable=True)
    if config == "llm_memory":
        return SetupAgent(crag=False, memory=memory, **common)
    return SetupAgent(crag=True, memory=memory, **common)


def installed_by(fix: str) -> str | None:
    m = re.match(r"(?:install|add_dependency|pin_package)\s+([A-Za-z0-9_.\-]+)", fix)
    return m.group(1).split("==")[0].lower() if m else None


def run_one(name: str, fault: str, config: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="team-eval-") as tmp:
        repo = copy_repo(name, Path(tmp))
        inject(repo, name, fault)
        agent = make_agent(config, repo, Path(tmp) / "memory")
        t0 = time.time()
        try:
            msg = agent.step()
            error = None
        except Exception as e:  # noqa: BLE001 - a crash is a result too
            msg, error = None, f"{type(e).__name__}: {e}"
        minutes = round((time.time() - t0) / 60, 2)
    exp = EXPECTED_FOR.get((name, fault), EXPECTED[fault])
    ev, payload = (msg.evidence, msg.payload) if msg else ({}, {})
    fixes = [f for f in ev.get("fixes", []) if not f.startswith("recreate the dry-run")]
    built = bool(msg) and msg.kind == "pipeline.setup_completed"
    unrelated = sorted({p for f in fixes if (p := installed_by(f)) and p not in exp["packages"]})
    diagnoses = ev.get("diagnoses", [])
    chosen = {d["item"]: d.get("value") for d in ev.get("decisions", [])}
    truth = SLOT_TRUTH.get(name, {})
    slots_right = all(chosen.get(k) == v for k, v in truth.items())
    success = (built if exp["fixable"] else (msg is not None and not built)) and not unrelated and slots_right
    return {"repo": name, "fault": fault, "config": config, "success": success, "built": built,
            "expected": "build" if exp["fixable"] else "report", "fixes": fixes, "attempts": len(fixes),
            "unrelated_installs": unrelated, "stage": payload.get("stage"), "error_class": payload.get("error_class"),
            "why": ev.get("why"), "llm_calls": sum(d.get("calls", 0) for d in diagnoses) + ev.get("decide_calls", 0),
            "decide_calls": ev.get("decide_calls", 0),
            "slots_model": (getattr(agent._client, "last_model", None) or getattr(agent._client, "model", None))
            if agent._client not in ("auto", None) else None,
            "escalated": sum(1 for d in diagnoses if d.get("escalated")),
            "models": [d.get("model") for d in diagnoses],
            "rejected": sum(len(d.get("rejected", [])) for d in diagnoses),
            "crag": [(d.get("crag") or {}).get("action") for d in diagnoses],
            "decisions": {d["item"]: d["source"] for d in ev.get("decisions", [])},
            "chosen": chosen, "slots_right": slots_right if truth else None,
            "minutes": minutes, "error": error, "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _table(rows: list[dict]) -> list[str]:
    out = ["| Config | Success | Fixable built | Unfixable reported | Mean attempts | Unrelated installs | LLM calls | Minutes |",
           "|---|---|---|---|---|---|---|---|"]
    for config in CONFIGS:
        rs = [r for r in rows if r["config"] == config]
        if not rs:
            continue
        fx = [r for r in rs if r["expected"] == "build"]
        un = [r for r in rs if r["expected"] == "report"]
        out.append(f"| {config} | {sum(r['success'] for r in rs)}/{len(rs)} "
                   f"| {sum(r['built'] for r in fx)}/{len(fx)} | {sum(r['success'] for r in un)}/{len(un)} "
                   f"| {sum(r['attempts'] for r in rs) / len(rs):.2f} | {sum(len(r['unrelated_installs']) for r in rs)} "
                   f"| {sum(r['llm_calls'] for r in rs)} | {sum(r['minutes'] for r in rs):.1f} |")
    return out


def invalid_cases(rows: list[dict], cases) -> set:
    """Final set: a fixable case where the fault didn't fail the build (rules built it with no fix) is invalid
    (e.g. np.NaN on wine, whose numpy stays < 2): left out of the score, as decided before the run."""
    return {(r["repo"], r["fault"]) for r in rows if r["config"] == "rules" and (r["repo"], r["fault"]) in cases
            and r["expected"] == "build" and r["built"] and r["attempts"] == 0 and r["fault"] != "clean"}


def summary(rows: list[dict]) -> str:
    out = []
    for name, cases in SETS.items():
        bad = invalid_cases(rows, cases) if name in ("final", "cascade") else set()
        rs = [r for r in rows if (r["repo"], r["fault"]) in cases and (r["repo"], r["fault"]) not in bad]
        if not rs:
            continue
        out += [f"## {name} ({len(cases) - len(bad)} cases)", ""]
        if bad:
            out += ["Invalid (the fault doesn't fail the build), left out: "
                    + ", ".join(f"{a}/{b}" for a, b in sorted(bad)), ""]
        out += _table(rs) + [""]
        out += ["| Repo | Fault | " + " | ".join(CONFIGS) + " |", "|---|---|" + "---|" * len(CONFIGS)]
        for repo, fault in cases:
            cells = []
            for config in CONFIGS:
                r = next((x for x in rs if (x["repo"], x["fault"], x["config"]) == (repo, fault, config)), None)
                slots = "" if r is None or r.get("slots_right") is None else \
                    " [" + ", ".join(f"{k}={r['chosen'].get(k)} ({r['decisions'].get(k)})"
                                     for k in SLOT_TRUTH.get(repo, {})) + "]"
                cells.append("" if r is None else ("✅" if r["success"] else "❌") + f" {r['attempts']}"
                             + (f" ({', '.join(r['fixes'])[:60]})" if r["fixes"] else "") + slots)
            if any(cells):
                out.append(f"| {repo} | {fault} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m agents.setup.eval.team_eval")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--sets", default=",".join(SETS), help="case sets: " + ", ".join(SETS))
    ap.add_argument("--cases", default="", help="only these faults (default: all of the sets)")
    ap.add_argument("--repos", default="", help="only these repos (default: all of the sets)")
    ap.add_argument("--name", default=datetime.now().strftime("team_eval_%Y%m%d_%H%M"))
    args = ap.parse_args()
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    jsonl = out / f"{args.name}.jsonl"
    done = {(r["repo"], r["fault"], r["config"]) for r in map(json.loads, jsonl.read_text().splitlines())} \
        if jsonl.exists() else set()                     # resume an interrupted run
    cases = [c for s in args.sets.split(",") for c in SETS[s]
             if (not args.repos or c[0] in args.repos.split(",")) and (not args.cases or c[1] in args.cases.split(","))]
    for config in args.configs.split(","):
        for name, fault in cases:
            if (name, fault, config) in done:
                continue
            print(f"[{config}] {name} / {fault} ...", flush=True)
            row = run_one(name, fault, config)
            print(f"   {'OK  ' if row['success'] else 'FAIL'} {row['attempts']} fixes, {row['minutes']} min, "
                  f"{row['fixes'] or ''} {row['why'] or ''} {row['error'] or ''}"
                  + (f" chosen={row['chosen']}" if row.get("slots_right") is not None else ""), flush=True)
            with open(jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    (out / f"{args.name}.md").write_text(summary(rows), encoding="utf-8")
    print(summary(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
