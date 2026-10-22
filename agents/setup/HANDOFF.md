# Setup Agent (P1): handoff, state on 2026-10-17

This file is for the next conversation with Claude. Read it before doing anything: it says what exists, how it
works, what was decided and why, what is left, and the rules to follow. Last updated 2026-10-17 10:45 (Tunis).
**Day 2 (2026-10-17): full Corrective RAG built and verified on a real run (§5.6, §6b); then the agent completed
against the roadmap (§12): layout aligned, team image + compose, the two test repos build, new fix actions, logs, README.**

---

## 0. Read this first: rules for working on this project

- **Never modify teammates' files** (anything outside `agents/setup/`, `samples/`, `contracts/examples/`, and
  the two one-line `.gitignore` appends listed below). Read their branches with `git show <branch>:<path>`, never check them out.
- **Never overwrite an existing file in the team repo.** The agent itself follows this too: it only creates
  missing files and reports differences.
- **Never push** without Jack saying so. Everything is committed **locally only** on branch `setup-agent`.
- **Never commit `.env`.** It holds Jack's NVIDIA key. Check with `git check-ignore -v .env` before any push.
- **Don't tune or train on the held-out project.** The wine project is the team's main project. The Builder's evaluations
  ran on the taxi repo; never use the main project for prompt tuning or training data.
- **Explain simply.** Jack is new to MLOps, prefers plain words and tables, wants one decisive choice per
  question based on real-world practice (no time for comparing options), and wants the numbers traced to their source.
- **Be honest about evidence.** Say which model produced which number. Don't claim Gemma results that were measured on Qwen.

## 1. The project in one paragraph

ESPRIT, ADSP course, Topic 4: a six-agent autonomous MLOps pipeline built around an existing ML project.
The team's main project is the **Wine Quality MLOps baseline** (UCI red wine, 1,599 rows, `;` separated,
target `quality`), in the team repo **Pi-Mlops** (GitHub `KTHIRI-Chedi/Pi-Mlops`, described as public).
The team plan is the PDF "Agent Build Roadmap v2" (4 weeks, in the repo). Jack is **P1, the Setup Agent**
(his earlier standalone project called it "the Builder"). Other agents: P2 Reviewer (Chedi), P3 Checker,
P4 Trainer, P5 Deployer, P6 Monitor.

**The teacher grades on:** the LLM, prompt strategies, fine-tuning and with which method, evaluation metrics, and
the agent's tools and memory. At the suivi on 2026-10-16 the questions were all about the LLM (and fine-tuning) and
the RAG (and its type). Reaction was neutral.

## 2. Where things are (Jack's Windows PC)

| What | Path |
|---|---|
| Team repo | `C:\Users\mdain\OneDrive\Bureau\project\Pi-Mlops` (branch `setup-agent`) |
| Jack's original Builder (standalone, evaluations of record) | `C:\Users\mdain\OneDrive\Bureau\adsp-project\builder-agent` |
| Taxi dev repo (where the Builder was evaluated) | `C:\Users\mdain\OneDrive\Bureau\adsp-project\taxi-trip-regression` |
| Python to use on Windows | `C:\Users\mdain\OneDrive\Bureau\adsp-project\builder-agent\.venv\Scripts\python.exe` (Python 3.12; plain `python` is the Microsoft Store stub) |
| GPU | RTX 4070 Laptop, 8 GB; Ollama runs qwen2.5-coder:7b and nomic-embed-text |

From Claude's side: the PC is reached through the device bridge (Linux VM, folders mounted at
`$HOME/mnt/Pi-Mlops`, `$HOME/mnt/builder-agent`, `$HOME/mnt/taxi-trip-regression`). **The VM and the cloud
container cannot reach `integrate.api.nvidia.com`** (proxy 403) **nor Jack's Windows Ollama**. Any real LLM call
must be run by Jack in PowerShell; Claude writes the script and reads his pasted output.

**Git from the VM:** use `git -c core.autocrlf=true` and author
`-c user.name=fedimdaini -c user.email=fedi.mdaini@esprit.tn`. Files are CRLF (Windows clone): write with
`newline="\r\n"`, and when patching with Python read with `newline=""` and normalise `\r\n` before `str.replace`
(an edit silently did nothing once because of this). Commit messages end with the
`Co-Authored-By` / `Claude-Session` lines.

## 3. Commits on `setup-agent` (local, not pushed), oldest first

| Commit | What |
|---|---|
| `dd93a72` | Week 1: `samples/agents.yaml`, `samples/schema.py` (Pandera WineBatch), stub, contract examples, db healthcheck in `baseline/docker-compose.yaml` |
| `1ae808a` | Builder copied into `agents/setup/builder_agent` (Python 3.10, Chroma replaced by numpy cosine) |
| `3003258` | NVIDIA NIM client (`llm/api.py`): chat + embeddings, same interface as Ollama |
| `97ce3bd` | **Team mode**: complete the wine project with the missing files, never overwrite |
| `d56dd45` | **Agent loop end to end**: dry run, rule fixes, builder branch, board messages |
| `e2ee7d1` | **LLM diagnosis** for errors no rule fixes + memory write-back; `.env` added to `.gitignore` |
| `24c22cd` | Accept the team's provider name `nvidia_nim`; `check_llm.py` |
| `b0ba954`, `b5ebb02` | `check_key.py` (dead key vs model unavailable vs busy) |
| `8084194` | **Local LLM switch** (`BUILDER_LLM_PROVIDER` in `.env` beats `agents.yaml`) |
| `7ccb5e7` | `pandas` in `agents/setup/requirements.txt` |
| `2cfe990` | Validator: a module the error shows as found counts as installed (from the first real diagnosis) |
| `b7ce0e3` | This HANDOFF.md |
| `9df0f47` | **Corrective RAG** (`memory/crag.py`), prompt `diagnose_team_v2`, wired into the agent |
| `146183f` | CRAG refinement keeps explanations, drops echoes of the error; ≤3 sentences per page |

`main` is at `c69a965`. Tests: **454 passed, 56 skipped** (`python3 -m pytest -q -p no:cacheprovider agents/setup/tests tests`
from the Pi-Mlops root, in the VM).

**Teammate files touched (only these, all tiny):** `baseline/docker-compose.yaml` (db healthcheck uncommented, roadmap
asks for it), `.gitignore` (appended `agents/setup/logs/`, `.cache/`, `.tools/`, `memory/`, then `.setup/`, then `.env`).

**Not in git (ignored, on Jack's PC only):** `Pi-Mlops\.env` (Jack's own NVIDIA key, `TAVILY_API_KEY` (Jack's free Tavily key, 1,000 credits/month, no card), local switch),
`agents\setup\memory\web_cache\` (cached Tavily results, replayed by evaluations),
`Pi-Mlops\baseline\.env` (MinIO/MySQL defaults for the baseline compose: `minio`/`minio123`, `mlflow_user`/`mlflow`, etc.),
`data\winequality-red.csv`.

## 4. Layout agreed with the team

Code stays in `baseline/`, data in `data/` at the repo root (roadmap 5.2). Commands run from the Pi-Mlops root.
(P5's branch moved the baseline to the root; the team was to be told the agreed layout. Not confirmed yet.)

```
Pi-Mlops/
  baseline/            the wine project (team code: train.py, app/main.py FastAPI, Dockerfiles, compose)
  data/                winequality-red.csv (gitignored; `make data` downloads it)
  samples/             agents.yaml + schema.py (Jack's hand-made samples, week 1)
  shared/              messages.py (board envelope), depmap.py   [team]
  agents/reviewer/     Chedi (on his branch: shared/llm.py too)
  agents/setup/        THE SETUP AGENT (everything below)
    agent.py           the 7-step loop
    dry_run_jobs/build.py
    builder_agent/     the Builder (scan, decide, llm, render, sandbox, fix, memory, team, ...)
    prompts/           slots_v1-3, diagnose_v1-5, diagnose_team_v1, ...
    contracts.yaml     Builder contracts + integration fields (incl. integration.layout)
    check_llm.py, check_key.py
    tests/
```

## 5. How the Setup Agent works now (with the wine project)

### 5.1 The loop (`agents/setup/agent.py`, roadmap 3.2 / 5.8)

`SetupAgent(repo="baseline", root=".", commit=True, client="auto", memory="auto", learn=True)`; `step()` runs one turn:

1. **observe**: first run, or a pending `request.rebuild` (from the Reviewer, topic `pipeline:setup`).
   `handle(msg)` answers "accepted" immediately; idempotent; unknown kinds ignored.
2. **analyze**: no LLM. Rebuild reason + changed paths, or "first run".
3. **decide**: rule decision → `dry_run_build` (menu: write_files 1, dry_run_build 1, apply_fix 2, commit_setup 1, give_up_and_report 1).
4. **dry_run + act**: team mode generates the files → dry run → fix loop (below) → on success writes the missing files
   through `RepoWriter` on a new branch `builder/<date>`, commits them there (author from `contracts.yaml`
   `reviewer.builder_commit_author`) and **switches back** to the branch the repo was on. `commit=False`: files left uncommitted on the builder branch.
5. **verify**: posts `pipeline.setup_completed` (payload: repo, commit, branch, files{Makefile, ci, schema, agents_yaml,
   dependency_map}, agents_yaml, schema_path, depmap_seeded; evidence: fixes, diagnoses, created, verified, findings)
   or `pipeline.setup_failed` to `human` (payload: stage, error_class, attempts, log_ref; evidence: why, diagnoses). Reply_to/incident_id set from the trigger.
6. **remember**: every fix the dry run checked is written to `builder_agent/memory/learned/` (via `memory/learned.py write_back`)
   with its outcome; retrieval uses only the passed ones ("learned" memory set = base + passed). Plus `self.history`.
   Each learned entry records in `source.retrieval` what the LLM was shown (CRAG action, graded/kept cases, web
   query, web URLs; `fixed_by` llm/rule): traceable ("learned with web help from X, verified by the dry run"),
   never indexed or shown in prompts, so the memory snapshot hash is unchanged. Web text itself is never stored as
   knowledge; it only lives in `memory/web_cache/` (raw search results, for replay).

Thresholds come from `agents.yaml` (`agents.setup.thresholds`: max_fix_attempts 3, dry_run_sample_rows 200, dry_run_timeout_s 600),
falling back to `samples/agents.yaml`.

### 5.2 Team mode (`builder_agent/team/`): complete, never overwrite

`python -m agents.setup.builder_agent.team` (plan only) / `--write`. Paths come from `contracts.yaml` `integration.layout`
(code_dir baseline, data_path data/winequality-red.csv, data_url UCI, config_sample samples/agents.yaml, makefile Makefile,
ci .github/workflows/pipeline.yml, schema_file pipeline/schema.py, depmap pipeline/dependency_map.yaml, agents_yaml agents.yaml,
tests_dir tests; all `confirmed: false` until the team confirms).

`team_plan()` scans `baseline/` and the data and finds: Python 3.10, requirement files (`baseline/requirements.txt`,
`baseline/app/requirements.txt`; the MLflow server's own file excluded), train script `baseline/train.py` (takes the data path
as argv[1]), serve target `baseline.app.main:app`, separator `;`, target `quality`, 11 feature columns with ranges, which code files name which features.
**Findings it reports (doesn't fix):** scikit-learn imported but undeclared; uvicorn and boto3 installed only in
`baseline/app/Dockerfile`; `train.py` saves no model (P4's file). The Makefile installs those three.

Generated (only if missing; existing ones are compared by digest and reported "same"/"differs"):
- `Makefile`: install / data / sample / train / serve / test / all. `make all SAMPLE=1` = first 200 rows (the dry run).
- `.github/workflows/pipeline.yml`: ubuntu-24.04, Python 3.10, `make install`, ruff `--select E9,F63,F7,F82`, `make test SAMPLE=1`.
- `pipeline/schema.py`: Pandera WineBatch (0.1–99.9 percentiles ±10%, widened to cover every row), CLI `python pipeline/schema.py <csv>`.
- `pipeline/dependency_map.yaml`: owners (code→reviewer, feature/dataset→checker, model→trainer, deployment→deployer) and links (computes, column_of, trained_on, serves).
- `agents.yaml`: from `samples/agents.yaml` + scanned facts.
Templates: `render/templates/team_*.j2`. `check_files()` checks YAML, schema compiles, agents.yaml keys, `make -n` via stdin (writes nothing).

### 5.3 The dry run (`dry_run_jobs/build.py`, option c)

Copies the repo to a temp folder (skips .git, venvs, mlruns, caches, logs), writes the generated files there, runs
`make all SAMPLE=1` with `DRY_RUN=1`, `MLFLOW_TRACKING_URI=file:` in the copy. **No Docker** (the Builder's 9-stage Docker
sandbox stays optional). Uses its own venv **outside the repo** (`~/.cache/setup-agent/dry-run-venv`, override `SETUP_AGENT_CACHE`), because a venv in OneDrive got corrupted once.
Classifies errors by regex: BrokenEnvironment, ModuleNotFoundError, ImportError, ResolutionImpossible, SchemaError, FileNotFoundError, OutOfMemory, TestFailure, SyntaxError, Timeout; stage from make's `[target] Error`.
Real run: the whole pipeline goes green in ~8 s with a warm venv.

### 5.4 The fix loop (max 3, in `act()`)

1. **Rules first** (`SetupAgent.fix_for`): ModuleNotFoundError → install the dist (IMPORT_TO_DIST map), once per package;
   BrokenEnvironment → recreate the venv, once.
2. **LLM** if no rule (`builder_agent/team/fix.py diagnose_team`), reusing the Builder's `fix.diagnose()`:
   - prompt `prompts/diagnose_team_v1.md` = diagnose_v5 adapted (no Docker image, Makefile only);
   - one-shot + **Corrective RAG** (§5.6) over the memory's top 3 (`FixMemory.advanced`: BM25 + vectors, RRF k=60, stage bonus);
     without CRAG (`crag=False`) the top 3 go in as they are (prompt `diagnose_team_v1`);
   - strict JSON schema; **team menu = add_dependency, pin_package, give_up** (`validate_fix(..., allowed=...)` rejects the rest);
   - validator retry with reasons, max 3 calls; modules named in `cannot import name X from 'M'` count as installed (so `add_dependency M` is rejected → pin_package);
   - **regression guard**: a fix that makes an earlier stage fail (order install, data, sample, train, test) is reverted and the LLM is told.
3. give_up / no valid answer / no LLM → `setup_failed` with the reason.

### 5.5 Which LLM (`builder_agent/llm/api.py make_client / make_embedder`)

Order: explicit `--provider` > `BUILDER_LLM_PROVIDER` / `BUILDER_EMBED_PROVIDER` in `.env` (local switch, this PC only) >
`agents.yaml llm.provider` (team: `nvidia_nim`, alias of `api`) > NVIDIA if a key is set > Ollama.
With Ollama, a NIM model name falls back to `BUILDER_OLLAMA_MODEL` or **qwen2.5-coder:7b**.
**Current state on Jack's PC: Ollama** (`.env` has `BUILDER_LLM_PROVIDER=ollama`, `BUILDER_EMBED_PROVIDER=ollama`; delete those two lines to use NVIDIA).
**Two LLMs (since 2026-10-20):** the **slots** (decide step) use the client above, `SetupAgent.llm()`: the external
API model by design (Gemma on NVIDIA; Ollama while NVIDIA is stuck). The **diagnosis** (fix loop) uses
`SetupAgent.diagnosis_llm()`: `BUILDER_DIAGNOSIS_MODEL` (Jack's `.env`: `qwen-builder-sft-v5`, the fine-tuned Qwen)
on `BUILDER_DIAGNOSIS_PROVIDER` (default ollama). Unset, or a client passed to the agent (tests, rules-only), or the
model not pulled in Ollama (printed) → the slots LLM does both. Learned fixes record the diagnosis model.
⚠️ `qwen-builder-sft-v5` was trained on the **Builder's** Docker-mode prompt (`diagnose_v3`, taxi repo, menu
add_dependency / add_system_package / pin_package / set_env_var / set_python_version), not on the team prompt
(`diagnose_team_v2`, which adds fix_path / change_entry_command). In team mode it is out of its training
distribution: the eval config `llm_sft` measures what it gives there.
NVIDIA: `google/gemma-4-31b-it` chat (JSON schema, falls back to json_object on 400/422, temp 0, seed 42), `nvidia/nv-embedqa-e5-v5` embeddings.

### 5.6 Corrective RAG (`builder_agent/memory/crag.py`, Yan et al. 2024, arXiv 2401.15884)

Chosen because the teacher asked about the RAG type and Jack wants a type whose definition is met **fully**
(not "similar to"). Literature check done: Naive ✅ (v1), **Advanced ✅ fully** (pre-retrieval signature, hybrid
retrieval, post-retrieval RRF rerank), Modular ⚠️ framework view, RAG-Fusion ❌ (needs LLM-generated multiple
queries; we fuse retrievers, not queries: never call it that), Self-RAG ❌ (trained reflection tokens),
Adaptive RAG ❌. CRAG implemented in full:
1. **Retrieve**: advanced retriever top 3.
2. **Retrieval evaluator** (`relevance`, rules, deterministic, no model call, like the paper's small T5):
   0.25 same exception class + 0.5 shared package/module names (`entities`: quoted names, `module named X`,
   dotted `mlflow.x`, `libffi.so` → libffi; keywords like "named" excluded) + 0.25 word overlap; an error that
   names no package: 0.25 class + 0.75 overlap. Option `LlmEvaluator` (LLM grades yes/partly/no as JSON).
3. **Action** by thresholds UPPER 0.6 / LOWER 0.35: **Correct** (any ≥ 0.6: keep, refined, no web),
   **Incorrect** (all < 0.35: discard all, rewrite query, web search), **Ambiguous** (keep ≥ 0.35 + web).
4. **Knowledge refinement (decompose-then-recompose)**: a case → strips error / verified fix / cause (cause kept only if
   it names what the error names: fights copied reasons); web pages → sentences scored by `strip_score`
   (0 for code/traceback/error echoes; needs cause-or-fix words), top 6, ≤ 3 per page, put back in page order.
5. **Query rewrite** (`rewrite_query`, deterministic; the paper uses an LLM): "python <error signature without paths/hex/line numbers> fix".
6. **Web search**: `TavilySearch` (POST api.tavily.com/search, Bearer key, 5 results, basic), cache by query hash in
   `agents/setup/memory/web_cache/`; `CRAG_SEARCH` = `cache` (default: search on a miss) / `replay` (never search: evaluations) / `off`.
   Web search replaces nothing in the definition: the paper's Incorrect action is web search, kept as is.
7. Prompt `diagnose_team_v2`: "RETRIEVED KNOWLEDGE (Corrective RAG: <action>)", memory marked "our build verified",
   web marked "NOT verified: hints only", rule "a version you pin must exist".
Evaluator behaviour on the base memory: markupsafe → incorrect; `No module named 'sklearn'` → ambiguous (keeps gen-002);
exact cases (xgboost, libffi, numpy dtype, mlflow 404) → correct. Tests: `tests/test_crag.py`, `test_setup_llm.py`.
The VM can't reach Tavily (blocked); Jack's PC can.

## 14. Team-mode evaluation v1, and v2 (2026-10-20): read this first

### 14.1 v1 result (`eval/results/team_eval_v1.*`, Jack's PC, in the container, base qwen2.5-coder:7b)

| Config | Success | Fixable built | Unfixable reported | Unrelated installs | LLM calls | Minutes |
|---|---|---|---|---|---|---|
| rules | **11/15** | 8/12 | 3/3 | 0 | 0 | 4.9 |
| llm_plain | 9/15 | 8/12 | 1/3 | 4 | 35 | 10.9 |
| llm_memory | 9/15 | 8/12 | 1/3 | 4 | 30 | 10.8 |
| llm_crag | 8/15 | 8/12 | 0/3 | 4 | 34 | 11.1 |

The LLM configs did WORSE than rules only. Reported as it is. The slots never needed the LLM (all decided):
every failure is the **diagnosis**. What the logs show: a pin of the version already installed (scikit-learn
1.7.2), pins of versions that don't exist (2.9.2, 1.6.2), pins of packages the error doesn't name (mlflow, numpy:
pip's own lines name every package), `fix_path baseline` accepted although train already runs there (the check
compared with "." instead of the Makefile's folder), a sweep 1.7.2 → 1.7.1 → 1.7.0, and no-op pins on a typo
(broken_import) counted as unrelated installs.

### 14.2 What changed for v2 (generic: no rule names a fault of the evaluation)

1. **Two LLMs** (`7c16997`): slots = API model (`SetupAgent.llm()`), diagnosis = `BUILDER_DIAGNOSIS_MODEL`
   (`qwen-builder-sft-v5`, the fine-tuned Qwen, on Ollama) via `SetupAgent.diagnosis_llm()`. See §5.5.
2. **The dry run reports what it installed** (`installed`: `pip list` of the dry-run venv) and, for
   "cannot import name X from M", the names M really has that look like X (`import_hint`, difflib 0.85).
3. **Validator** (`team/fix.py team_checks`): never a fix already tried (rules' and LLM's, `fix_key`); a pin
   must change the installed version, exist on PyPI (`pypi_releases`, pre-releases and yanked excluded; PyPI
   unreachable → check skipped, never guessed), and be of a package the error names (`packages_in_error`, pip's
   lines excluded); no version change for a misspelt name (→ give_up); add_dependency of an installed package
   rejected; fix_path compared with the folder train really runs in (`current_cwd`).
4. **Prompt `diagnose_team_v3`**, the same for every LLM config (only the retrieval differs): adds
   AVAILABLE VERSIONS (installed version + newest patch of each minor on PyPI, for the packages the error names)
   and rule 9 (pin from the list, never the installed one; a removed function/argument → an older minor).
5. **Path rule** (`fix_for`): the missing relative path is searched from the script's folder up to the repo root
   (nearest wins). This fixes wine/wrong_data_path **by rule**, so the rules config gains a case too.
6. **Decide step bug** found while writing the slot repos: with several top-level `train*.py`, the first one was
   taken and called "the only script that trains a model" (a silent guess). Now all are candidates; exactly one
   that saves a model → decided by rule; otherwise needs_llm, with "saves a model / saves no model" in the facts.
   The target's facts now include the chosen training script's lines that use each candidate column.

### 14.3 The v2 evaluation (`eval/team_eval.py`, 26 cases × 5 configs)

* Sets: `v1` (the same 15 cases), `heldout` (written AFTER the fixes, never used to tune them:
  `removed_numpy_alias` np.float, `removed_pandas_append` DataFrame.append, `data_dir_path` bare file name with
  the file in data/; × wine, sk_root, sk_src), `slots` (`sk_two_scripts`, `sk_legacy`, ground truth `SLOT_TRUTH`:
  a build with the wrong choice fails).
* Configs: rules, llm_plain, llm_memory, llm_crag (all diagnose with base qwen2.5-coder:7b, forced, so `.env`
  can't change them), **llm_sft** = llm_crag with the fine-tuned model for the diagnosis. Compare llm_crag vs
  llm_sft for what the fine-tuning adds. The slots use the slots LLM in every LLM config.
* Ground truth checked by hand with a scripted LLM: numpy case = numpy==1.23.5 + pandera==0.20.4 (2 fixes);
  pandas case on sk_root = pandas==1.5.3 + numpy==1.26.4 + pandera==0.20.4 (3 fixes, the maximum). On wine the
  pandas case is unfixable (the team pins pandas==2.2.3): the agent must report.
* **Finding:** pandera (the agent's own schema check, `tools` target, unpinned) refuses numpy < 1.24.4 at import.
  A project that needs an older numpy/pandas also needs an older pandera: the fix loop has to find it. Not changed
  (found on a held-out case: changing it now would tune on it). A later fix: pin the tools to versions that match
  the project's.
* ⚠️ `qwen-builder-sft-v5` was trained on the Builder's Docker-mode prompt (§5.5): team mode is outside its
  training data (fix_path / change_entry_command are new to it).
* Run (Jack's PC, Ollama running, ~1.5–2 h; resumable: the same command continues):
  `docker compose -f docker-compose.agents.yml run --rm setup python -m agents.setup.eval.team_eval --name team_eval_v2`
  Subsets: `--sets heldout`, `--configs rules,llm_sft`, `--repos sk_root`.

### 14.4 v2 result (`eval/results/team_eval_v2.*`, Jack's PC, container, 2026-10-20 evening)

| Set | rules | llm_plain | llm_memory | llm_crag | llm_sft |
|---|---|---|---|---|---|
| v1 cases (15) | 12/15 | 12/15 | 12/15 | **15/15** | **15/15** |
| held-out (9) | 1/9 | 1/9 | 2/9 | 1/9 | 1/9 |
| slots (2) | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 |

* Same 15 cases as v1: llm_crag 8/15 → 15/15, unrelated installs 4 → 0. The 3 sklearn cases are solved by CRAG
  and by the fine-tuned model (1.6.1 fails, then 1.5.2 builds); plain and memory configs still sweep versions.
* Slots: the LLM chose right 2/2 (target `species` from the script's lines; `train_v2.py`); the rule's guess 0/2.
  (The "LLM calls" column counts only diagnosis calls: the slot calls are not counted, a reporting gap.)
* Held-out: the fixes don't generalise yet. Causes from the logs:
  1. numpy (0/3): every LLM config found numpy==1.23.5 (right), then the build failed on the agent's OWN check
     tool (pandera refuses numpy < 1.24.4, §14.3). The agent's bug, not the diagnosis.
  2. pandas on sk_root/sk_src (0/2): needs 3 linked pins (pandas<2, numpy<2, older pandera); the LLM mostly pins
     pandas 2.x (it doesn't know append was removed in 2.0; the web pages say "use concat", a code change).
  3. data_dir_path (1/3 configs×repos solved once, llm_memory sk_root): the LLM is not told where the file is;
     it pins pandas because the traceback goes through pandas, and the validator allows it.
  4. wine pandas (unfixable): reported correctly by every config.
* Fine-tuned vs base (llm_sft vs llm_crag): equal on the v1 cases (15/15), equal on held-out (1/9, 3 vs 1
  unrelated installs). No measurable gain in team mode, expected: it was trained on the Builder's Docker prompt.
* The held-out cases are now SEEN: any change made because of them must be measured on a new held-out set.

### 14.5 v3 (2026-10-20 night): three changes from the v2 held-out failures (those 9 cases are now SEEN)

1. **The agent's check tools can't break the project** (`team_makefile.j2`, `tools`): after installing pandera,
   the Makefile checks that `pa.DataFrameSchema` imports; if not (newest pandera refuses numpy < 1.24.4 or
   pandas < 2.1), it installs `pandera==$(PANDERA_FALLBACK)` (0.20.4, checked with numpy 1.23.5 + pandas 2.3.3
   and numpy 1.26.4 + pandas 1.5.3).
2. **A missing file** (`team/fix.py`): prompt `diagnose_team_v4` adds WHERE THE MISSING FILE IS (`where_text`:
   the files of that name in the repo, and the folder from which the code's path opens it) and rule 10; the
   validator rejects pin_package / add_dependency when the error is a missing relative file.
3. **Reporting**: the slot LLM calls are counted (`decide_calls` in the evidence; `llm_calls` includes them).

New held-out set **`heldout_v3`**, written after these changes: `removed_scipy_simps` (scipy.integrate.simps,
gone in scipy 1.14 → scipy==1.13.1), `removed_numpy_product` (np.product, gone in numpy 2.0 → numpy==1.26.4),
`missing_data_file` (the code reads a file that is nowhere → report), × wine, sk_root, sk_src. Ground truth
checked by hand with a scripted LLM (sk_root, and wine for scipy). Also checked: the v2 cases now need fewer
fixes (numpy alias: numpy==1.23.5 alone; pandas on sk_root: pandas==1.5.3 + numpy==1.26.4; data_dir_path:
fix_path data).

Run (only rules, llm_crag, llm_sft, to spare the PC):
`docker compose -f docker-compose.agents.yml run --rm setup python -m agents.setup.eval.team_eval --name team_eval_v3 --configs rules,llm_crag,llm_sft`
Report: `v1` and `heldout` are development cases now; `heldout_v3` is the honest generalisation number.

### 14.6 v3 result (`eval/results/team_eval_v3.*`, 2026-10-21 morning; rules, llm_crag, llm_sft)

| Set | rules | llm_crag | llm_sft |
|---|---|---|---|
| v1 cases (15, development) | 12/15 | **15/15** | **15/15** |
| v2 held-out (9, SEEN since v2) | 1/9 | 6/9 | **8/9** |
| slots (2) | 0/2 | 2/2 | 2/2 |
| **heldout_v3 (9, new)** | 4/9 | 4/9 | 4/9 |

* On the new held-out set the LLM adds nothing over rules yet. Of its 4 successes, 3 are the unfixable
  `missing_data_file` (rules "succeed" too: no rule fires, so they report) and 1 is **an invalid case**: on wine
  `np.product` doesn't fail (the team's mlflow 2.9.2 keeps numpy < 2, where np.product still exists). So the
  real fixable score on new faults is 0/5 for every config.
* Why, from the logs: for scipy.simps the LLM pins NEWER scipy (1.18.1, 1.17.1, 1.16.3), which need Python 3.11
  (install fails, reverted); for np.product it pins numpy 2.2.x / 2.3.5 although the error says "removed in the
  NumPy 2.0 release". The versions list shows versions that can't install on the project's Python, and the 7B
  models pick from its top.
* Good: missing_data_file reported 3/3 with the right reason; zero unrelated installs everywhere. One wasted fix:
  llm_crag tried `fix_path data` when no file of that name exists.
* Fine-tuned vs base (fair on every set: neither model was tuned on any case): 8/9 vs 6/9 on the v2 held-out set,
  with fewer calls (19 vs 31): it picks older versions (numpy 1.23.5, pandas 1.5.3) where the base model picks
  2.x; equal elsewhere. Small samples.
* Candidate v4 fixes (generic): show and accept only versions that install on the project's Python
  (`requires_python` is in the PyPI JSON per file: scipy 1.16.3 needs >= 3.11); a fix_path must make the missing
  path exist. The v3 held-out cases are SEEN after this.

### 14.7 v4, then the agent is FROZEN; the final external test set (2026-10-21)

**v4** (from the v3 held-out failures, so those cases are SEEN):
1. `pypi_releases(name, python=...)`: only versions whose files' `requires_python` accept the build's Python
   (the dry run now reports `python`, e.g. 3.10.12). AVAILABLE VERSIONS shows only those, and a pin of a version
   that doesn't install on it is rejected with the list (checked: scipy 1.16.3 rejected, then 1.13.1 builds).
   Limit: `requires_python` is permissive for old releases (scipy 1.6.1 says >= 3.7 but has no 3.10 wheel).
2. A `fix_path` must make the missing path exist from that folder.

**Freeze**: decided with Jack before the final run: no change to the agent after v4, whatever the result.

**Final set (`final`, 6 faults × 3 repos = 18 cases)**: `eval/external_pool.py`. The faults are not written by
us: each is one use of an API that the library's own release notes list as removed (numpy 1.24 / 2.0, pandas 2.0,
scikit-learn 1.2 / 1.4 / 1.6, scipy 1.12 / 1.14). Pool of 33, filtered by one objective test before the draw (must
fail on the newest versions for Python 3.10: np.bool dropped, it exists again in numpy 2) → 32; the faults used in
development are not in it. Draw: `random.Random(20261021).sample(sorted(POOL), 6)` →
scipy_misc_face, pandas_read_csv_squeeze, sklearn_bagging_base_estimator, numpy_NaN, pandas_df_iteritems,
sklearn_load_boston. Disclosure: a first trial draw on an unsorted scratch list gave other faults; the committed
code's draw is the one used, chosen before running anything on either.
Rule fixed before the run: a case whose fault doesn't fail the build (rules build it with no fix, e.g. np.NaN on
wine where numpy stays < 2) is invalid and left out (`invalid_cases`).
Ground truth checked by applying the pins directly (no agent): see the comment above `EXPECTED.update` in
`team_eval.py`; the 3 wine pandas cases must be reported (the team pins pandas==2.2.3: ResolutionImpossible).

**Known limitation, found while checking that ground truth and NOT fixed (the agent is frozen):** for a
TypeError raised at the call site (`read_csv() got an unexpected keyword argument 'squeeze'`,
`BaggingClassifier.__init__() got an unexpected keyword argument 'base_estimator'`) the traceback has no frame
inside the library, so the error text never names pandas / scikit-learn, and the validator rejects the right pin
("not named in the error output"). Expect those cases to fail; a later version could map the called name
(`_pd.read_csv`, the imported class) back to its package.

Run (rules, llm_crag, llm_sft; final set only, ~1 h):
`docker compose -f docker-compose.agents.yml run --rm setup python -m agents.setup.eval.team_eval --name team_eval_final --configs rules,llm_crag,llm_sft --sets final`
Optional, same frozen agent on the development sets (~1.5 h, shows v4 didn't break them):
`... --name team_eval_v4_dev --configs rules,llm_crag,llm_sft --sets v1,heldout,slots,heldout_v3`

### 14.8 FINAL result (`eval/results/team_eval_final.*`, 2026-10-21, frozen v4 agent, run once)

17 cases (wine/numpy_NaN invalid, left out by the rule fixed before the run):

| Config | Success | Fixable built | Unfixable reported | Unrelated installs | LLM calls | Minutes |
|---|---|---|---|---|---|---|
| rules | 2/17 (12%) | 0/15 | 2/2 | 0 | 0 | 3.3 |
| llm_crag (base Qwen + CRAG) | 8/17 (47%) | 6/15 | 2/2 | 0 | 47 | 13.9 |
| **llm_sft (fine-tuned + CRAG)** | **10/17 (59%)** | **8/15** | 2/2 | 0 | 50 | 15.3 |

95% intervals, cluster bootstrap over the 6 faults (10,000 resamples): rules 0–24%, llm_crag 22–76%,
llm_sft 39–83%; llm_crag − rules +6% to +71% (excludes 0: the LLM beats rules on unseen faults);
llm_sft − llm_crag +0% to +35% (touches 0): the fine-tuned model is better on 2 cases and worse on none, both
from one fault (load_boston, wine and sk_root): a direction, not a proof.
Solved: scipy.misc.face 3/3 (both), np.NaN 2/2 (both), BaggingClassifier on sk_src (both), load_boston 2/3
(sft only). Failed: read_csv(squeeze) and DataFrame.iteritems on sk_root / sk_src (the LLM pins pandas 2.x;
squeeze also hits the known limitation of §14.7), BaggingClassifier on wine / sk_root (the known limitation:
the pin is rejected, the model gives up), load_boston on sk_src (swept 1.6 → 1.4, never reached < 1.2).
Every unfixable case (wine pandas, pinned by the team) was reported; zero unrelated installs in every config.

### 14.9 Metrics (2026-10-21): intervals and RAG metrics

* `eval/team_intervals.py` → `results/team_intervals.md`: cluster-bootstrap 95% intervals (over faults) for every
  team_eval table. Final set: llm_crag − rules +6% to +71%, llm_sft − rules +17% to +81% (both exclude 0);
  llm_sft − llm_crag +0% to +35% (includes 0). Development sets: wide intervals (3–5 faults each).
* `eval/rag_metrics.py` → `results/rag_metrics.{json,md}` (Jack's PC, container, no LLM: a scripted client replays
  the verified fixes so the loop meets the errors a right LLM meets; 49 diagnoses over 42 cases). Relevance rule
  fixed before computing: a memory case is relevant if its fix acts on a package the error is about, or its error
  line is the same message.
  - **Recall@3: 82% (14/17)**. The 3 misses are the first load_boston error, where the label itself is weak (numpy
    counts as "named" only because the traceback passes through numpy); without them 14/14.
  - **CRAG evaluator** (keep/discard of 147 retrieved cases): accuracy 95%, **precision 100%** (7/7: it never kept
    an irrelevant case; 133/133 irrelevant discarded), **recall 50%** (7/14): it keeps only a near-identical error
    (the numpy binary error, score 1.0) and discards same-package cases with another message (np.float /
    np.product / np.NaN vs gen-003, scores 0.04–0.27). Conservative by design (thresholds 0.6 / 0.35).
  - **Action** vs ideal: 86% exact (42/49); all 7 errors are "incorrect" where a related case existed (it went to the
    web instead); it never chose "correct" wrongly.
  - Memory coverage: only 17 of 49 diagnoses had any relevant case (10 Docker-era cases): the case for a growing
    memory. Wilson intervals assume independent diagnoses; the same error on 3 repos is not independent.

### 14.10 Real-project acceptance test (2026-10-21), and three setup fixes

On a clone of the real repo (scratch copy, VM): the agent ran for real (no LLM needed): `builder/2026-10-21` with
5 new files, committed, back on the original branch, no team file touched. Then, on the builder branch:
`make -C baseline install train test` on the **full data** (1599 rows) passes; the schema catches 3 planted errors
(negative value, empty cell, quality = 42); the CI file is valid, its ruff lint passes, the team tests pass (5);
`agents.yaml` has exactly the sample's keys; the 5 findings are all true (checked in the files: scikit-learn not in
any requirements file; uvicorn / boto3 only in app/Dockerfile; mlflow 2.9.2 in app/requirements vs 2.14.1 on the
server; train.py saves no model). `make serve` can't start without the team's stack: app/main.py loads a hard-coded
MLflow run id at startup (the trainer's file; the no_model_output finding says so). CI on GitHub not run (no push).

Fixed (`384f1b6`), outside the evaluated diagnosis, so the frozen results stand:
1. A fresh clone (data/ not in git) crashed the agent: `ensure_data` now downloads `data_url` to `data_path` (whole
   file or nothing), or fails with the reason.
2. The generated CI ran `make test` only: it now runs `make -C baseline all SAMPLE=1` (trains too).
3. A missing line break in the Makefile's install comment.
4. (`8961458`) `TeamLayout.detect` fell back to the small-repo layout on a fresh clone (no data yet) and crashed:
   it now keeps the team layout when `baseline/` exists and `data_url` is set.

**Confirmed on Jack's PC (container, 2026-10-21):** fresh clone of `setup-agent` (no data/) → the agent downloads the
data, writes and commits its 5 files on `builder/…` → `make -C baseline all` on the full data: install OK, "Trained
baseline ElasticNet model successfully on 1599 rows", "passes WineBatch". Note: `make install` in the image's own
Python replaced the agents' mlflow 2.14.1 by the project's 2.9.2 (the version_mismatch finding, live): run the
pipeline in its own venv (CI does: a fresh runner).

**CI confirmed on GitHub (2026-10-21):** `setup-agent` and `builder/2026-10-21` pushed to `KTHIRI-Chedi/Pi-Mlops`
(the push of a workflow file needed a token with the `workflow` scope: browser login fixed it). PR #2, base
`setup-agent` ← `builder/2026-10-21` (5 files, main untouched). Workflow `pipeline / lint-and-test` succeeded in
1 min 5 s: Install 34 s, Lint 0 s, Pipeline on a sample (install, train, test, with the UCI download) 6 s, Team tests
12 s; 1 warning annotation: Node.js 20 deprecated for checkout v4.2.2 / setup-python v5.6.0 (forced onto Node 24,
harmless). Fixed in the template (`3a63126`: checkout v5.0.0, setup-python v6.0.0); the agent regenerated its files
on `builder/2026-10-21-2` (commit 6319650), PR #3 (base setup-agent): green in 1 min 7 s, both versions resolved,
no annotation. PR #2 to close (replaced by #3). Nothing merged; merging setup-agent into main is a team decision.

### 14.11 CRAG evaluator on the 3 Builder test faults never looked at (2026-10-21)

The leakage check promised in the old §9 (thresholds 0.6 / 0.35 calibrated while looking at memory cases):
`results/crag_heldout_builder.json`. Retrieval here is keyword-only (no embeddings reachable from the VM); the
evaluator is rule-based, so its scores are the same as with embeddings. Relevant = a memory case of the same family.

| Fault | Same-family case in the top 3 | Evaluator score of it | Action |
|---|---|---|---|
| gen-010 flask 2.0.3 (incompatible pin) | gen-003 | 0.0 (discarded) | incorrect → web |
| gen-011 Python 3.13 (python version) | gen-004 | 0.0 (discarded) | incorrect → web |
| gen-013 empty tracking URI (env var) | gen-012 | 0.558 (kept) | ambiguous (memory + web) |

Same picture as the team-mode RAG metrics: never a confident wrong "correct", but low recall: same-family cases
with a different error message score near 0. On gen-013 the two mlflow 404 cases (another family) score 0.55 too:
the evaluator can't separate them from the env-var cases, so "ambiguous" is the honest answer there.

### 14.12 Slots fallback and the LLM cascade (2026-10-21, after the final test; new code, measured on new faults)

* **Slots fallback** (`SetupAgent.llm()`, `llm/api.py FallbackClient`): an API slots model (e.g. Nemotron 3 Ultra via
  `BUILDER_LLM_PROVIDER=api` + `NVIDIA_MODEL`) is wrapped as API (timeout `BUILDER_SLOTS_TIMEOUT`, 60 s) → local Ollama
  model → the rules' guess ("assumed"). `BUILDER_SLOTS_FALLBACK=off` disables it. The decide log records which model
  answered (`model`).
* **Cascade** (`SetupAgent.escalation_llm()`, `BUILDER_ESCALATION_MODEL` [+ `_PROVIDER`, default api]): the diagnosis
  model (fine-tuned Qwen) answers first; the escalation model takes over (a) in the same attempt when the first gives
  up or gives no valid answer, (b) for the remaining attempts when the first model's fix changes nothing (same stage
  and error class) or breaks an earlier stage. Same budget of 3 fixes; every diagnosis records `model`, and
  `escalated` when it came from the second model. No cascade unless asked (never with a client passed in).
* **Test set `cascade`** (`external_pool.cascade_faults()`, seed 20261022, fixed before the draw): 6 faults from the 26
  never drawn: scipy_trapz, pandas_error_bad_lines, scipy_binom_test, numpy_unicode_, numpy_int_alias,
  numpy_asfarray, × 3 repos. Ground truth checked with the pins applied directly (comments in `team_eval.py`).
  Config `llm_cascade` = `llm_sft` + Nemotron 3 Ultra as the escalation model.
* Run (Jack's PC, ~50 min): `docker compose -f docker-compose.agents.yml run --rm setup python -m agents.setup.eval.team_eval --name team_eval_cascade --configs rules,llm_sft,llm_cascade --sets cascade`
  then `python -m agents.setup.eval.team_intervals team_eval_cascade` for the intervals.

**Result (`results/team_eval_cascade.*`, Jack's PC, 2026-10-21, run once):** 16 valid cases (wine/np.unicode_ and
wine/np.asfarray invalid: wine keeps numpy < 2).

| Config | Success | 95% interval | Fixable built | Unfixable reported | Unrelated installs |
|---|---|---|---|---|---|
| rules | 1/16 (6%) | 0–19% | 0/15 | 1/1 | 0 |
| llm_sft (fine-tuned alone) | 7/16 (44%) | 22–71% | 6/15 | 1/1 | 0 |
| **llm_cascade** (+ Nemotron 3 Ultra) | **10/16 (62%)** | 31–93% | 9/15 | 1/1 | 0 |

Cascade − fine-tuned alone: +19%, 95% +0% to +41% (includes 0): better on 4 cases, worse on 1. Read honestly:
* 3 wins come from the escalation (wine and sk_src scipy.trapz: Nemotron pinned 1.13.1 / 1.11.4 after the
  fine-tuned model's 1.14.1 changed nothing; sk_src np.int: Nemotron found 1.23.5).
* 1 win is NOT the cascade: sk_root np.int was solved by the fine-tuned model alone in the cascade run but given
  up in the llm_sft run: the local model varies between runs even at temperature 0 (GPU non-determinism).
* 1 loss: wine np.int: after the fine-tuned model's 1.25.2 changed nothing, Nemotron gave up ("a bug in the code"),
  while llm_sft alone kept going down to 1.23.5. Escalating on "no change" can hand over a case that was converging.
* **New finding — import-check false positive:** scipy.stats.binom_test failed 0/3 for every config. The import check
  (difflib 0.85) finds `binomtest` close to `binom_test` and calls it a misspelling, so the validator forces give_up;
  but binom_test is a REMOVED API (scipy 1.12), fixable with scipy==1.11.4. A renamed function looks like a typo.
  Not fixed (seen on this test). Fix later: before calling it a misspelling, check whether an older release had the
  name (PyPI versions + an import in a throwaway venv), or only flag names absent from every release.
* sk_root read_csv(error_bad_lines): the known limitation again (pandas not named in a call-site TypeError);
  Nemotron itself explained the validator refusal.


## 13. Fresh review (2026-10-17 midday), commit `f28b581`: what was wrong and what changed

Read the code as a stranger; checked the suspicions by running them. Tests after: 471 passed, 57 skipped.

**Fixed:**
1. **The dry run could pass with a missing dependency** (proved: sk_src without pyyaml passed). Two causes: the venv had
   `--system-site-packages` (the machine's packages leaked in) and `make install` also installed the agent's tools
   (pytest, pandera, **pyyaml**). Now: an isolated venv per install set (no system packages, keyed by the hash of every
   Makefile + requirements file of the copy, reused while unchanged, 3 kept, in `~/.cache/setup-agent/venv-*`), and
   a `tools` target that `test` runs after `train`. Proved: without pyyaml or pandas → fails at train with the module name.
   Cost: a wine venv is a few hundred MB, the first dry run downloads (pip cache helps). The sandbox VM's disk filled once
   (TMPDIR is `~/tmp`; pytest keeps 3 runs of tmp folders with venvs): clean `~/tmp/pytest-of-*` and `~/.cache/setup-agent/venv-*`.
   A venv with a missing compiled library (install cut short) → `BrokenEnvironment` → recreated once.
2. **A rebuild could never change anything** (never-overwrite + files already merged). Now on `request.rebuild` the agent
   may rewrite **its own Makefile and CI** (first line = the MARKER, `RepoWriter.update_own`, on the builder branch) and
   the dry run checks those updated files; `agents.yaml` (board), `schema.py` (Checker), dependency map (Reviewer/Trainer)
   stay first drafts. Test: merge, board edits agents.yaml, train.py needs emoji → rebuild updates only baseline/Makefile.
3. **Supply chain**: LLM `add_dependency` / `pin_package` must name a package mentioned in the error output, already installed
   by the project, or a known import's pip name (`team.fix.known_packages`); otherwise rejected with the reason and retried.
   (pip install runs code; a name from the LLM or from a web page CRAG showed it must not be trusted alone.)
4. **Scope**: `make test` ran the team's `tests/` (agents' tests): a teammate's broken test failed the setup. Now
   `integration.layout.tests_dir` = `baseline/tests` (the ML project's own; none today → schema only) and CI has a separate
   "Team tests" step (`pip install -r requirements.txt`, `pytest tests`).
5. Small: a reverted fix now also restores `train_cwd` / `train_style`; learned records name the real prompt (`diagnose_team_v2`)
   and retriever (`crag`); board messages carry diagnosis summaries (full prompts only in `logs/llm_calls.jsonl`);
   `dry_run_sample_rows` is read from agents.yaml.

**Checked and fine:** openai 3.33.0 still has `OpenAI().chat.completions.create(response_format, temperature, max_tokens)`
(Chedi's code works in the image); keys never in git or in the image; learned memory and web cache are gitignored.

**Not fixed, reported (decide later):**
- **Documentation said things that weren't true** (now corrected in §11 and §9.2): team mode has no slot filling; the 57% is
  the Builder; the CRAG thresholds were calibrated on examples that are evaluation cases.
- The schema check at setup is a tautology (schema built from the data it then checks): its value is for future batches (Checker).
- The findings (e.g. mlflow mismatch → deployer) are only in `setup_completed` evidence, not sent to the owners as messages.
- The agents' container runs as root: on a Linux teammate's machine files it writes in the mounted repo are root-owned.
- `git config --system safe.directory '*'` in the image: acceptable in a container, not outside it.
- For small repos, the generated `agents.yaml` still carries the wine sample's units/groups.
- Tests write `runs.jsonl` / `llm_calls.jsonl` into the real repo's `agents/setup/logs/` (gitignored).
- **FLAKE4DOCK**: downloadable (github.com/flakidock/docker-flakiness-study: dataset/, configs/, ...), but it is **Docker image
  build** failures (apt/pip/npm inside `docker build`), 798 flaky Dockerfiles of which **100 have repairs**. Team mode's dry run
  builds no image, so it fits only the Builder's Docker sandbox. Recommendation: don't add it to the team agent; if wanted,
  load the 100 repaired cases as an external, unverified source for the Builder mode only. CRAG's web search already gives
  the team agent external knowledge.

### 14.13 Integration with the team branch `p3/data-checker` (2026-10-21; setup code, not the evaluated diagnosis)

Checked on a scratch merge of `setup-agent` + `p3/data-checker` (3 conflicts: `requirements.txt` take theirs + our
extras rank-bm25, pyflakes, packaging, tomli, python-dotenv, SQLAlchemy; `.gitignore` union; `samples/schema.py`
theirs). Changed in commit `6111c92` so the agent fits the integrated project:
- **Team schema:** when `samples/schema.py` exists (the Data Checker's), `make test` validates the data with it (any
  pandera `DataFrameSchema` in the module; `normalize_columns()` applied if the module has one) and the agent no
  longer writes `pipeline/schema.py`; `agents.yaml` `schema_path` points to the team's file.
- **`--data`:** a train script with an argparse `--data` option gets `--data $(abspath $(RUN_DATA))`
  (the Trainer's `train.py` moved off `sys.argv[1]`).
- **Tests** no longer assume findings the team has since fixed; new tests for both changes. All suites pass
  (494 + 7 + 5, skips are Docker/Ollama).

Result on the merge: `--plan` creates 4 files (no second schema) and runs `train.py --data …`. The dry run then
**fails honestly**: the Trainer's `train.py` holds out a fixed 400 rows, larger than the 200-row sample
(`ValueError: test_size=400 … number of samples 200`). Before, it "passed" only because the data path was ignored and
the full file was used. With a 600-row sample the whole `make all SAMPLE=1` is green (setup_completed, 0 fixes).
**Decision (Jack, 2026-10-21): the sample is now 600 rows** (our default `SAMPLE_ROWS` and `samples/agents.yaml`),
a deliberate change from roadmap 7.3's 200 so the Trainer's fixed 400-row holdout fits (400 test + 200 train).
Checked on the merge: setup_completed, 3 files created, 0 fixes. Still worth suggesting to the Trainer:
`test_size=0.25` instead of 400 would work with any sample size.
Left for the team:
1. The root `agents.yaml` (P3's) has no `agents.setup` section, so our thresholds in `samples/agents.yaml` are
   ignored after the merge (defaults apply). Add `setup: {thresholds: {max_fix_attempts: 3, dry_run_sample_rows: …,
   dry_run_timeout_s: 600}}` under `agents:`.

Also (commit `c153fd1`): the Trainer's service is in `docker-compose.agents.yml` as P4 wrote it (own image
`agents/trainer/Dockerfile`, `./runs:/app/runs` for board.db, joins the external `example-network` of the baseline
stack, so start that first); `.dockerignore` no longer ignores `data/`, because the Trainer's Dockerfile copies it
(its build would fail otherwise). Not built here (no Docker in the VM): check on Jack's PC after the merge with
`docker compose -f docker-compose.agents.yml build trainer`. The README now lists the manual steps the agent replaces.

### 14.14 Demo page for the presentation (2026-10-21; optional, removable)

`agents/setup/demo/` (+ `tests/test_demo.py`): a local web page in English with buttons (Jack asked for English), same look and same idea as
P3's `harness/demo_ui.py` (port 8766; the Checker's is 8765). **Jack may change or remove it**: the agent never
imports it; deleting the folder and the test removes it. Four steps: *Analyse* (the plan of the real project, read
only), *Build* (`step()` on a temp copy), *Break and repair* (5 errors injected into the copy's train.py:
`hidden_dependency` → a rule fixes it; `numpy_float`, `sklearn_load_boston` → need the LLM; `team_typo`,
`missing_data_file` → the right outcome is to report), *Results* (sections of `team_intervals.md` and
`rag_metrics.md`). Switch "The rules" (no LLM, no memory: the eval's `rules` config) / "The LLM" (the agent as
configured in `.env`, `learn=False`). The demo only listens: it subclasses the agent to report dry runs and rules,
and wraps `resolve_with_llm` / `diagnose_team` during a run (put back after). Every finished run is saved to
`demo/recordings/<action>-<fault>-<mode>.json`; "Replay the recording" replays it, marked as a replay.
Faults are inserted after the docstring and `from __future__` imports (the team's new train.py has one; the eval's
`prepend` would give a SyntaxError there, not changed since the eval is frozen). Checked in the VM, rules mode, on
`setup-agent` and on the scratch merge: every error really breaks the build in both states (`sklearn_squared` was
dropped: scikit-learn 1.5.1 is pinned after the merge, so it no longer breaks); the page was looked at with
Playwright (desktop and phone width). **Not checked here: the LLM mode** (no Ollama in the VM): Jack runs
`numpy_float` and `sklearn_load_boston` once in LLM mode on his PC, which also records them for the replay.
Run (container: make + Linux): `docker compose -f docker-compose.agents.yml run --rm -p 127.0.0.1:8766:8766
-e DEMO_HOST=0.0.0.0 setup python -m agents.setup.demo`. Click *Build* once before presenting: the first
dry run in a new container installs the packages (minutes); later ones reuse them.

### 14.15 Check of the newest team merge (2026-10-21 evening)

`p3/data-checker` at `7b7892f` (Checker P3 + Trainer P4, 3 new commits: Trainer gate fix, Checker docs). Jack's folder
`Bureau\newest merge\Pi-Mlops` is checked out on `main`; the merge is on that branch. In a clean venv with the
team's pins: 133/133 team tests pass; Checker scenario C PASS; Checker evaluation (rules) 100% of faults right,
1 alarm per 15 normal batches (target ≤ 1); Trainer solo D/E/F = retrain → Deployer / refuse / reject, as documented;
the long-running Trainer on board.db with fake partners (scenario D) ends with "candidate v2 sent to the Deployer"
(train + register the first model first, as its guide says). LLM paths not run live (Groq). `setup-agent` merges on
top with the same 3 conflicts (resolved as planned); after it 135 team tests + 498 of ours pass and
`python -m agents.setup --no-commit` gives setup_completed, 3 files created, 0 fixes. `p5-deployer-baseline`
(15 Oct) is not merged into it yet.

### 14.16 Branch `data-checker+builder` (2026-10-21, local, not pushed)

Jack's request: his work added to the team's merge. Branch `data-checker+builder` = `p3/data-checker` (7b7892f) +
merge of `setup-agent` (commit d3a93cb). Conflicts: `samples/schema.py` and `requirements.txt` theirs (+7 packages),
`.gitignore` theirs + ours; `baseline/docker-compose.yaml` kept as theirs (they rely on the MySQL image's healthcheck;
our week-1 test now accepts that, 7d26ee4). Checked on that commit: 135 team tests, 498 + 5 of ours pass; agent run:
setup_completed, 3 files, 0 fixes. To share: `git push origin data-checker+builder`, then a PR into `p3/data-checker`.
After the merge, Jack's `.env` needs `BUILDER_LLM_PROVIDER=api` (the root agents.yaml says groq, which our agent
doesn't support; keep NVIDIA/Nemotron: no new tests needed).

### 14.17 Whole-project test of `data-checker+builder` before the push (2026-10-21 night)

Fresh clone of the branch, new venv from the merged `requirements.txt` + `requirements-agents.txt` (pip check clean).
Tests: 135 team + 498 + 7 + 5 Setup Agent (skips: research-repo files, Docker). Two test-only fixes made here:
three tests in `test_setup_agent.py` put a line before the Trainer's `from __future__` import (SyntaxError):
helper `add_first`; `team_eval.prepend` skips `from __future__` lines too (same behaviour on the old train.py, so
the frozen results are unchanged). Also found: the VM disk filled up again (pytest temp + venvs), cleaned.
Agents: Checker scenario C PASS, evaluation 100% (60 normal: 0.5 alarms/15); Trainer solo D/E/F/bad_candidate/none as
documented; Trainer long-running + fake partners D ends "candidate v2 sent to the Deployer"; Setup Agent `--plan`
(3 files, agents.yaml exists, 0 findings) and run: setup_completed, 0 fixes; the generated `make -C baseline all
SAMPLE=1` passes with a local MLflow store (a real `make train` needs the MLflow server up).
Interfaces: Checker demo (8765): page, /api/status, /api/cases, /api/scenario (pass, 27 messages), /api/evaluate
(100%); Setup demo (8766): page, status, analyse, results, build + 3 faults (rules), right outcomes; wine API
(uvicorn, file store): /model loaded v1 champion, /predict 5.33 on the README wine, bad input 400, /metrics.
Not tested here: Docker builds, live LLMs (no keys/Ollama in the VM). For P4: `agents.trainer.fake_partners`
inserts 5 values into `batches`; if the real Checker created the shared 11-column table first, it crashes
(`table batches has 11 columns but 5 values were supplied`); use named columns. Not a pipeline bug (their
Checker+Trainer integration test passes).

## 12. Day 2, completing the agent against the roadmap (read this; it updates §4, §5.2–5.4)

Commits (after `ddadc26`): `5334840` check_key compares with served models · `62e2b0a` handoff (Gemma queue) ·
`de1a5d7` **layout aligned** · `56e201d` **team image + compose** · `41fbea2` **test repos + fix_path / change_entry_command** ·
`7719890` **logs, compose check, README**. Tests: **469 passed, 57 skipped**.

**Layout (changes §4/§5.2):**
- The pipeline Makefile is **`baseline/Makefile`** (contracts `integration.layout.makefile`), paths relative to it,
  run as `make -C baseline <target>`; CI uses `make -C baseline ...`. Reason: roadmap 5.2 gives the **root
  `Makefile` to the team runner** (`contract-test`, `solo`, `scenario`, `experiment`). The week-1 fixture already said baseline/Makefile.
- The dry-run job is **`dry_run_jobs/build.py` at the repo root** (roadmap 5.2; Chedi's `dry_run_jobs/feature_diff.py` is there too).
  Input `makefile` → `make -C <dir> -f <name> all SAMPLE=1`.
- `make test` runs the team's tests **from the repo root** (`cd $(ROOT) && pytest tests`): they open `samples/...` relatively.
- Bug found and fixed: the "install the missing module" rule had `pip install`ed the unrelated PyPI package
  **`shared`** for the team's `shared/` (a namespace package without `__init__.py`, which an installed package
  shadows). The rule now never installs a module that is a folder/file of the project. (Removed it from the dry-run venv.)

**Team image (P1's shared piece, roadmap 4.1), new files at the root:** `Dockerfile.agents` (python:3.10-slim + make + git,
`safe.directory '*'`, `OLLAMA_HOST=http://host.docker.internal:11434`), `docker-compose.agents.yml` (six services
`python -m agents.<name>`, repo mounted at `/app`, `.env` read at run time `required: false`, `host.docker.internal:host-gateway`,
MLflow/API/Prometheus URLs on the host), `requirements.txt` (one pinned set for all agents, resolved together with pip
on Python 3.10; **mlflow 2.14.1 = the server**, openai 3.33.0, packaging 24.2, ...), `.dockerignore` (`.env`, data, .git, logs, memory).
`python -m agents.setup` (`agents/setup/__main__.py`): first run, `--rebuild REASON --changed ...`, `--no-commit`, `--plan`;
messages appended to `agents/setup/logs/outbox.jsonl` until `shared/board.py` exists. Ollama client reads `OLLAMA_HOST`.
**Not yet built on Jack's PC** (asked him to run: `docker compose -f docker-compose.agents.yml build setup`, then `run --rm setup python -m agents.setup --plan`, then the tests inside).

**Any small repo (roadmap 7.1 MVP):** `TeamLayout.detect(root)` = the team layout if `baseline/` + the wine data exist, else
`for_repo` (code = the repo, data = its first CSV, Makefile at its root, no download URL → `data` only checks the file).
A `train*.py` with top-level training code counts as the training script (the scanner only takes guarded/CLI scripts;
not changed, to keep the Builder's measured behaviour). `make test` skips pytest when there is no tests folder.
Practice repos in `agents/setup/test_repos/`: **sk_root** (train.py/predict.py at root, requirements.txt) and **sk_src**
(package `src/irisclf/` with a relative import, no requirements file); iris data, target `species`.
Real dry runs, rules only: **sk_root 0 fixes, sk_src 1 (run as module), a 3rd layout "sk_dir" 1 (run from the script's folder), wine 0**: all ≤ 3 attempts.

**New fix actions (roadmap 7.4 menu), team menu only:** `fix_path {target: workdir, path}` and
`change_entry_command {stage: train, style: script|module}`, in their own answer schema `TeamAnswer` (`team/fix.py`) so the
Builder's `Fix` union and the schema its evaluation sends are **unchanged** (comparability with the 36%→57% runs).
`validate_fix(extra_check=...)` / `diagnose(answer=..., extra_check=...)` hooks; `team_checks` rejects a non-existent folder,
`..`, a no-op, or module style for a script outside a package. Rules: relative-import error → module; FileNotFoundError whose
file sits next to the script → workdir. Learned memory stores them as actions; `MemoryDoc.fix_text` falls back for team actions.
Still not in the menu (need Docker, not in the light dry run): `change_base_image`, `add_system_package`.

**New finding in team mode:** `baseline/app/requirements.txt` pins **mlflow==2.9.2**, the server runs **2.14.1** (roadmap 5.3 says the same version;
the fault-001 family). Reported (owner deployer), not changed. Also: `sk_src`'s pickle output is now recognised (no false "no model output").

**Definition of done extras:** `logs/runs.jsonl` (one line per turn), `logs/llm_calls.jsonl` (every call incl. rejected, full prompt/answer,
CRAG action), `docker compose config` check recorded in evidence (skipped where docker is absent), `agents/setup/README.md`.

**Things found in Chedi's branch to raise with the team (not touched):** root `agents.yaml` lists features with **spaces**
("fixed acidity") but units with underscores; the roadmap/our samples use underscores everywhere. `shared/` has no `__init__.py`
(namespace package, shadowed by any installed package named `shared`).

## 6b. Day 2 result: the first real CRAG run (2026-10-17, Jack's PC, `check_llm.py`)

| | 2026-10-16, Advanced RAG | 2026-10-17, Corrective RAG |
|---|---|---|
| Memory in the prompt | gen-006, gen-003, gen-002 (all unrelated) | all 3 graded 0.29 / 0.0 / 0.03 → **discarded** (incorrect) |
| External knowledge | none | real Tavily search, rewritten query, refined sentences |
| LLM calls | 2 (1 rejected) | **1** |
| Fix | `pin_package markupsafe==2.1.1` (**wrong**) | **`pin_package markupsafe==2.0.1` (right)**, true reason ("soft_unicode deprecated in 2.1.0") |

Same model (qwen2.5-coder:7b, Ollama), same error, same validator: only the RAG changed. It's one case, not an
evaluation; the evaluation (§9.3) must measure it. After that run, refinement was improved (5 of the 6 first kept
sentences only echoed the error); replayed on the cached real pages: 6 explanatory sentences from 4 sources.

## 6. What happened on day 1 with the LLMs (important for the team talk)

- **The team's shared NVIDIA key (from Chedi) is dead**: `GET /models` works but every chat call returns 403
  "Authorization failed", on every model. Evidence it worked before: Chedi's commit `9a04bfe` (Oct 13) loosened a test
  to accept Gemma's natural wording. Since then it was revoked/expired/out of credits.
- **Chedi's `shared/llm.py`** (on `origin/feature/p2-reviewer-agent`): `openai` package, NVIDIA base URL, Gemma,
  `json_object`, 1 retry, then **silently falls back to `rule_decision`** with only a print. So his Reviewer stays green
  with a dead key; nobody noticed. Suggested (not done; his file): record `source: llm|rule` in decisions. Not touched.
- **Jack created his own key** (build.nvidia.com), now in `Pi-Mlops\.env`. The key is accepted (no 403), but
  **Gemma didn't answer within 5 minutes, twice**, and even a 5-token call timed out. Two other listed models returned
  404 "Function not found for account" (listed but not served to free accounts; bad test picks).
- **2026-10-17 morning, conclusive:** `check_key.py` again: Gemma no answer in 120 s. NVIDIA's own playground on
  build.nvidia.com (Jack's account): "hello" hangs with reasoning on; with reasoning off it shows
  **"There are 416 requests in the queue…"** then **"Error: Retries exhausted 3/3"** (screenshots taken). Gemma 4 31B
  is overloaded on the free tier; not the key, not the code, not the network. In ~12 h of trying it never answered;
  local Ollama answered every call in 4–7 s. `check_key.py` now compares with widely served models (Llama 3.3 70B,
  Qwen2.5-Coder 32B, ...) to tell "Gemma busy" from "account not served".
- **Decision (Jack): keep Ollama for now**, discuss with the team. Jack's leaning: NVIDIA everywhere except where the fine-tuned model works.
- Ollama crashed once with "CUDA error: illegal memory access" (GPU had 0.8/8 GB used); **restarting Ollama fixed it**.
- **First real end-to-end run on Ollama (`check_llm.py`) passed**: chat 4.1 s; embedding 768-d; diagnosis of
  `cannot import name 'soft_unicode' from 'markupsafe'`: memory showed gen-006, gen-003, gen-002; call 1
  `add_dependency markupsafe` **rejected by the validator** ("already installed; use pin_package"); call 2
  `pin_package markupsafe==2.1.1` accepted. 2.1.1 is still wrong (soft_unicode was removed in 2.1.0; right: 2.0.1).
  In the real loop the dry run would fail it and the next attempt would be told. Good teacher example of three checks: schema → validator → dry run.

## 7. Evaluation numbers and where they come from (don't mix them up)

All from `builder-agent/docs/RESULTS.md`. **Every run used qwen2.5-coder:7b on local Ollama** (temp 0, seed 42) on the
taxi repo, fixes verified by real Docker sandbox rebuilds. Nothing was measured with Gemma.

Leave-one-out (section 12): 14 fault cases, 3 configs × 5 seeds, 252 runs total:

| Metric | v1 (no memory) | v3 + advanced RAG | v5 + advanced RAG |
|---|---|---|---|
| First fix passes (main metric) | 36% (25/70) | 57% (40/70) | 57% (40/70) |
| Fixed within 3 | 61% | 86% | 81% |
| Action accuracy | 29% | 50% | 51% |

Decision rule: no configuration *shown* better (case-cluster bootstrap); v3 + advanced stays default.

**Fine-tuning (QLoRA SFT on Qwen2.5-Coder-7B, Colab T4, rank 16; LoRA on q_proj: 12.8M → 114,688 trainable, −99.1%):**
section 6 (SFT on v3 data) **tied** the base model; section 9 (SFT on v5 data, `qwen-builder-sft-v5`) 3/4 vs 2/4 first fix
on only 4 faults, one run each, one pass copied another fault's answer, one needed the validator, and it lost gen-011.
RESULTS.md: "the gain is thin… default not changed". **The fine-tuned model was never in the 252-run evaluation.**
No DPO (only 37 preference pairs).

**Can Gemma be fine-tuned?** Not through the free NVIDIA API (no weights; it serves only the base model). NVIDIA's guide
(full fine-tune with NeMo AutoModel) needs 8× A100/H100 and self-hosting. QLoRA on Gemma 31B 4-bit is trainable on a
rented/Colab Pro A100 but can't be deployed through the API. No fine-tuning method (LoRA, QLoRA, prompt/P-tuning, adapters)
works without weights. The API-only alternatives are prompt adaptation: dynamic few-shot via RAG (already done) and prompt optimisation (v1→v3→v5, already measured).

## 8. Decisions taken (one choice per criterion)

| Criterion | Choice |
|---|---|
| Dry run | Option (c): `make all SAMPLE=1` on 200 rows, no Docker; Docker sandbox optional |
| LLM | Team: Gemma via NVIDIA for both calls (slot filling, diagnosis). **Now on Jack's PC: Ollama qwen2.5-coder:7b** until the team decides |
| Fine-tuned Qwen | Research result, not production (unless the evaluation in §9 shows it wins) |
| Embeddings | NVIDIA or Ollama, keyword-only fallback (BM25) if unreachable |
| Prompt strategy | One-shot + RAG (dynamic few-shot) + strict JSON schema + validator retry with reasons |
| Fine-tuning method | QLoRA SFT; no DPO |
| Metrics | Task success (first fix passes ≈ SWE-bench resolve rate, sandbox-verified) + faithfulness (true vs copied reasons ≈ RAGAS) |
| RAG | Advanced hybrid (BM25 + vectors, RRF), measured → **full Corrective RAG on top (DONE 2026-10-17)**, real web search via Tavily with replay cache; FLAKE4DOCK optional extra external source later |
| Memory | Only sandbox/dry-run-verified fixes are retrievable |

## 9. What is left (rewritten 2026-10-21; the old list is in git history)

Done since the old list: team-mode evaluation v1→v4 + final external test (§14), RAG metrics and intervals (§14.9),
real-project acceptance + CI green on GitHub (§14.10), push of `setup-agent`, PR #3, presentation prep doc
(claude.ai artifact "Setup Agent — Presentation Prep").

**Matters for the grade**
1. **Builder leave-one-out with CRAG and the fine-tuned model**: `builder-agent/experiments/eval_loo/run.py`, same 14
   faults, configs (a) v3 + advanced base (the existing 57%), (b) + CRAG base, (c) + CRAG `qwen-builder-sft-v5`.
   Needs a CRAG retriever wired into the Builder's fix loop (see the old item 2 in git history). Docker Desktop, hours.
2. **The 50 hand labels** (Jack): `experiments/eval_loo/labels/to_label.csv`; the reason metrics stay preliminary until then.
3. ~~Measure the API model~~ **DONE (2026-10-21)**: `probe_models.py` (container): 3 of 39 chat models answer the key with
   right JSON: `nvidia/nemotron-3-ultra-550b-a55b` (1.6 s), `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (1.5 s),
   `openai/gpt-oss-20b` (19 s); Gemma 4 31B still times out, most others 404. Slots run with Nemotron 3 Ultra as the
   slots LLM (`-e BUILDER_LLM_PROVIDER=api -e NVIDIA_MODEL=nvidia/nemotron-3-ultra-550b-a55b`,
   `results/slots_api_nemotron.*`, `slots_model` recorded per row): **2/2 right** (target `species`, `train_v2.py`),
   1 LLM call per repo. Open: make it the default slots model (Jack's `.env`; `agents.yaml` is the team's).
4. **LoRA adapters** in Jack's Downloads (`qwen-builder-sft-extras.zip`, `qwen-builder-sft-v5-extras.zip`, ~173 MB each):
   move into `builder-agent/experiments/finetune/v3` and `v5`, show `adapter_config.json` (rank, alpha, target modules).

**Team (needs the teammates)**
5. Team message: PR #3 + green CI; train.py saves no model (trainer); mlflow 2.9.2 vs 2.14.1 (deployer); scikit-learn,
   uvicorn, boto3 missing from requirements; dead shared NVIDIA key (own keys, never in the chat); confirm the layout.
6. Close PR #2 (replaced by #3); merging `setup-agent` into `main` = team decision; week-3 pair session Reviewer + Setup.

**Set aside on purpose (future work, in the presentation doc)**
7. LLM cascade (fine-tuned first, API model + web when it fails), measured on a fresh draw from the 26 unused external faults.
8. "Improves with use": learn on wine, test on sk_root / sk_src (~30–40 min) — the only way to claim it.
9. Final-test weaknesses (frozen agent): pin refused when the error doesn't name the library; version bisection instead of
   stepping down; when an API was removed (release notes).
10. FLAKE4DOCK: Docker-only, possible for the Builder's Docker mode, not team mode.
11. Optional: remove the silent Ollama fallback in `make_client` once NVIDIA works; slides if the teacher wants them.

## 10. Commands (PowerShell, from `C:\Users\mdain\OneDrive\Bureau\project\Pi-Mlops`)

```powershell
$py = "C:\Users\mdain\OneDrive\Bureau\adsp-project\builder-agent\.venv\Scripts\python.exe"
ollama list                                   # qwen2.5-coder:7b and nomic-embed-text must be there
& $py agents\setup\check_llm.py               # 3 real calls: chat, embedding, diagnosis (prompt + RAG + validator)
& $py agents\setup\check_key.py               # is the NVIDIA key alive / is Gemma answering
& $py -m agents.setup.builder_agent.team      # team mode plan: findings + files it would create (writes nothing)
```
In the VM (Claude): `python3 -m pytest -q -p no:cacheprovider agents/setup/tests tests`; agent end to end on a copy:
`python3 -c "from agents.setup.agent import SetupAgent; m=SetupAgent(commit=False).step(); print(m.kind, m.summary)"` (run in a temp git copy, never the real repo).
On Windows there is no `make` by default: the dry run itself runs in the VM/Linux or CI.

Run the scripts **from the Pi-Mlops folder** (`cd C:\Users\mdain\OneDrive\Bureau\project\Pi-Mlops`): they read `.env` from there.
If Ollama fails with a CUDA error: tray icon → Quit Ollama → start it again.
In a new session, git commits from the VM need delete permission on the Pi-Mlops folder (git removes its own lock files);
without it a stale `.git/HEAD.lock` is left. Ask with device_request_delete_permission, then `rm -f .git/*.lock .git/objects/maintenance.lock`. If `pandas`/other module missing: `& $py -m pip install -r agents\setup\requirements.txt`.

## 11. Lines for the teacher

- **CORRECTED 2026-10-17 (fresh review): be precise about which system each claim is about.** The Builder (create mode,
  Docker sandbox, taxi repo) uses the LLM for **slot filling and diagnosis** and is the one **measured** (36%→57%, 252 runs).
  The **Setup Agent in the team repo (team mode)**, since commit `d1f884d` (2026-10-17 afternoon), uses the LLM for
  **(1) the decide step**: Python version / target column / training script are "decided" by rules when clear, else
  "needs_llm" → the LLM picks one candidate (`team/decide.py`, `prompts/team_slots_v1.md`, enum schema, validated, 1 retry;
  no LLM → rule guess kept, marked "assumed", reported as a finding); and **(2) diagnosis** with CRAG. On wine, sk_root and
  sk_src every choice is decided by rules (no LLM call); the LLM is asked only for genuinely ambiguous repos (tested with a
  repo that has two training scripts and two possible targets). LLM query rewriting for CRAG was considered and
  **rejected**: an error line is already a precise query (the real run found the right pages); an LLM rewrite adds latency,
  can inject guesses, and breaks the replay cache. The rule rewrite still satisfies CRAG's "rewrite before web search".
  Also fixed with it: the schema hard-coded the wine target (int 0..10); the target's check now comes from the data.
  Say "measured on the Builder's fix loop" for the
  numbers, never "the team agent scores 57%". The team agent's own numbers come from the next evaluation (§9.2).
- **LLM:** "Same agent code for local Qwen2.5-Coder-7B (Ollama) and hosted Gemma 4 31B (NVIDIA); temperature 0, strict JSON; the provider is one setting."
- **Prompt strategy:** "One-shot with the error, project facts and the 3 most similar verified fixes (RAG as dynamic few-shot), a JSON schema, and a validator that sends rejected answers back with the reason."
- **Fine-tuning:** "QLoRA SFT on Qwen 7B (fits a free T4; rank-16 LoRA trains ~99% fewer weights). Measured against the base model on held-out faults with sandbox-verified fixes: no clear gain, so production uses base model + RAG. Gemma can't be fine-tuned through the hosted API (no weights; NVIDIA's guide needs 8× A100 and self-hosting)."
- **Metrics:** "First fix passes, verified by actually rebuilding (like SWE-bench resolve rate): 36% without memory → 57% with advanced RAG, 252 runs, Qwen 7B. Plus faithfulness: are the reasons true or copied from memory."
- **Tools and memory:** "Menu of actions with levels; dry run before acting; regression guard; memory keeps only fixes the build confirmed; hybrid retrieval BM25 + vectors with RRF."
- **RAG type:** "Advanced RAG (signature query, hybrid BM25 + vectors, RRF rerank), measured 36% → 57%, then **Corrective RAG** on top:
  an evaluator grades each retrieved fix; Correct → keep refined; Incorrect → discard, rewrite the query, web search; Ambiguous → both;
  decompose-then-recompose keeps only useful pieces; web results are cached so the evaluation is repeatable. Real example: markupsafe,
  plain RAG showed 3 unrelated cases and the model pinned the wrong version; CRAG discarded them, searched, and the model pinned 2.0.1 in one call."
- **Three checks on every LLM answer:** schema → validator → dry run (tonight's markupsafe run shows all three).
