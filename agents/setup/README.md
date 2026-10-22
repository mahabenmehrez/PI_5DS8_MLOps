# Setup Agent (P1)

Turns an ML repository into a pipeline the other agents can plug into, and fixes its own build errors
(at most 3 attempts). Runs once at setup, then on `request.rebuild` (roadmap section 7).
Full state, decisions and history: [HANDOFF.md](HANDOFF.md).

## What it does, in one turn (`SetupAgent.step()`)

1. **Look / understand**: first run, or a `request.rebuild` from the Reviewer (answered "accepted" at once).
2. **Plan + decide**: scan the code and the data (requirements, undeclared imports, scripts, web app, feature
   ranges). Each choice the plan needs (Python version, target column, training script) is **decided** by a rule
   when the evidence is clear, or marked **needs_llm** with the reason and the candidates; only those go to the
   LLM, which must pick one candidate (`prompts/team_slots_v1.md`, validated, one retry). No LLM: the rule's guess
   is kept and reported as an assumption. Findings are reported, never fixed in other people's files.
3. **Generate what is missing**, never overwrite a team file: `baseline/Makefile` (install, tools, data, sample, train,
   serve, test, all; `install` = only the project's packages), `.github/workflows/pipeline.yml` (ruff + pytest), `pipeline/schema.py` (Pandera `WineBatch`, only when the
   team has no schema: if `samples/schema.py` exists, `make test` uses it),
   `pipeline/dependency_map.yaml`, `agents.yaml`. Existing files are compared and reported.
4. **Dry run** (`dry_run_jobs/build.py`): `make -C baseline all SAMPLE=1` on a copy, 600 rows (roadmap: 200; the Trainer holds out 400), no Docker, in an
   isolated venv (no system packages): a dependency the Makefile forgets fails here.
5. **Fix** (max 3): rules first (missing package, relative import, file next to the script, broken venv), then the
   LLM: one-shot prompt + **Corrective RAG** (memory graded; web search when it doesn't fit) + strict JSON + a
   validator that sends rejected answers back. Menu: `add_dependency`, `pin_package`, `fix_path`,
   `change_entry_command`, `give_up`. A package the LLM names must appear in the error output, be installed by the
   project or be a known import (pip install runs code). A fix that breaks an earlier stage is reverted.
6. **Write and commit** on a new `builder/<date>` branch, then back to the branch the repo was on. On a rebuild it may
   also rewrite its own Makefile and CI (marked first line); agents.yaml, schema.py and the dependency map are first
   drafts that the board, the Checker, the Reviewer and the Trainer own.
7. **Post** `pipeline.setup_completed` or `pipeline.setup_failed` (stage, error class, log); **remember** the fixes
   the dry run verified (`builder_agent/memory/learned/`).

## The manual steps it replaces

Running the wine project by hand takes these steps. Each row says what went wrong when we did it by hand and
what the agent does instead.

| # | Step by hand | What went wrong | What the agent does |
|---|---|---|---|
| 1 | Install the packages: `pip install -r baseline/requirements.txt` | Packages the code imports were not declared (scikit-learn, uvicorn, boto3), and the app pinned another mlflow version than the server | `make install` installs exactly the project's packages, plus the missing ones it found in the code; the gaps are reported to the file's owner |
| 2 | Get the data into `data/winequality-red.csv` | The data is not in git, so a fresh clone cannot train | `make data` downloads it when it is missing |
| 3 | Start the stack: `cd baseline && docker compose up -d --build` | MLflow could start before MySQL was ready | A healthcheck on the database, so MLflow waits for it; the agent checks that `docker compose config` passes (it does not start the stack) |
| 4 | Train and register the model: `python baseline/train.py` | The data path and the MLflow address have to be given by hand | `make train` runs the script with the data path (`--data` or the first argument, whichever the script reads) |
| 5 | Make the API serve the new model (`POST /reload`; the model is found by its `champion` alias) | With a run id instead of an alias, the id had to be copied and the app restarted | Not the agent's job (the Deployer moves the alias); `make serve` starts the API on port 8000 |
| 6 | Check the data and run the tests | No one command did it | `make test` validates the data with the team's schema, then runs the unit tests; the CI runs it on every push |
| 7 | Check that all of it works before committing | Only by doing all of it again | The dry run: `make all SAMPLE=1` on 600 rows, in a fresh environment, with up to 3 fixes, before anything is committed |

## Run it

From the repository root:

```bash
python -m agents.setup --plan                 # what it would create and what it found; changes nothing
python -m agents.setup                        # first run: generate, dry-run, fix, commit on builder/<date>
python -m agents.setup --no-commit            # same, files left uncommitted on the builder branch
python -m agents.setup --rebuild "train.py changed" --changed baseline/train.py
```

In the team image (needs Docker; Ollama on the host):

```bash
docker compose -f docker-compose.agents.yml build setup
docker compose -f docker-compose.agents.yml run --rm setup python -m agents.setup --plan
```

On any small repo (the code is the repo, the data its first CSV): `python -m agents.setup --root path/to/repo`.
Practice repos: `test_repos/sk_root`, `test_repos/sk_src`.

Checks without the build: `python agents/setup/check_llm.py` (3 real LLM calls: chat, embedding, a diagnosis
with Corrective RAG) and `python agents/setup/check_key.py` (is the NVIDIA key alive, is Gemma answering).

Tests: `python -m pytest -q agents/setup/tests` (the dry-run tests need `make`).

## Demo page (`demo/`, optional)

A local web page with buttons, in English, like the Data Checker's (`harness/demo_ui.py`): analyse the project,
build the pipeline, break and repair (5 errors to choose from, with the rules alone or with the LLM), and the
evaluation results. Each click runs the real agent on a temporary copy of the project; the real project is never
written. Every run is saved in `demo/recordings/` and can be replayed (marked "replay") if a live run fails.

```bash
docker compose -f docker-compose.agents.yml run --rm -p 127.0.0.1:8766:8766 -e DEMO_HOST=0.0.0.0 setup python -m agents.setup.demo
# then open http://127.0.0.1:8766
```

The agent never uses this folder: deleting `demo/` (and `tests/test_demo.py`) removes the demo and nothing else.

## Settings (no hard-coded values)

| Where | What |
|---|---|
| `agents.yaml` → `agents.setup.thresholds` | `max_fix_attempts` 3, `dry_run_sample_rows` 600 (default too), `dry_run_timeout_s` 600 |
| `agents.yaml` → `llm` | provider (`nvidia_nim` / `api` / `ollama`), model |
| `.env` (never committed) | `NVIDIA_API_KEY`, `TAVILY_API_KEY`; local switch `BUILDER_LLM_PROVIDER` / `BUILDER_EMBED_PROVIDER` (wins over agents.yaml on this machine), `BUILDER_OLLAMA_MODEL`; `BUILDER_DIAGNOSIS_MODEL` (+ `BUILDER_DIAGNOSIS_PROVIDER`, default ollama): the fine-tuned model for the diagnosis only, the slots keep the API model |
| environment | `OLLAMA_HOST` (the Ollama server), `CRAG_SEARCH` = cache / replay / off, `SETUP_AGENT_CACHE` (dry-run venv) |
| `contracts.yaml` → `integration.layout` | where things are in the team repo (code `baseline/`, data `data/`, generated file paths) |
| `builder_agent/memory/crag.py` | CRAG thresholds: Correct >= 0.6, Incorrect < 0.35 |

## Logs

`agents/setup/logs/` (not in git): `runs.jsonl` (one line per turn: decision, fixes, attempts, checks, outcome),
`llm_calls.jsonl` (every LLM call: prompt, answer, latency, tokens, valid or the validator's reasons),
`setup/<time>-<stage>.log` (the failed dry run's output), `outbox.jsonl` (messages, until `shared/board.py`).

## Known limits

- **No board yet**: messages go to `logs/outbox.jsonl`; `shared/board.py` doesn't exist on main.
- **Private helpers**: the roadmap (13.1.5) wants the shared LLM helper, memory and dry run; this agent has its own
  (Corrective RAG, validator, verified memory). To agree with the team once `shared/` lands.
- **The dry run has no Docker**: `docker compose config` is checked only where docker is installed; the Builder's
  Docker sandbox (9 stages) is still in `builder_agent/sandbox/` for full builds.
- **Windows**: no `make`, so the dry run runs in the team image or Linux/CI.
- **Disk**: each isolated dry-run venv is a few hundred MB (3 kept in `~/.cache/setup-agent/`); the first dry run downloads.
- **What the numbers measure**: the 36% → 57% results are the Builder's Docker fix loop (create mode, LLM for slot filling
  and diagnosis). This team mode uses the LLM for the unsure choices (decide step) and for diagnosis; its own
  evaluation is still to do.
- **LLM**: Gemma on NVIDIA's free tier was unavailable during development (queue full); measured with
  qwen2.5-coder:7b on Ollama. Web search (Tavily) works only where the network allows it.
- **Fixes it can't make**: anything in other people's code or data (it reports them as findings or gives up).
