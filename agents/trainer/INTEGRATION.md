# Integrating the Trainer agent (P4)

This page is for the rest of the team: what the Trainer needs from your piece, what it sends you, and how to run it inside the full pipeline.

The Trainer is a long-running agent. Every 2 seconds it reads its messages on `board.db`, sends a heartbeat, and checks whether a retrain was requested or is due. When one is, it decides whether retraining is worth it, retrains in a dry run, and checks the new model against the live one. Then it either hands the new model to the Deployer or explains why it rejected it. It never imports another agent: everything goes through the board, the tables in `board.db` and MLflow (roadmap 3.4).

**Scoring:** the model is scored by MAE (mean absolute error, in quality points), and **lower is better**.
- "Gain" always means how much lower the MAE gets.
- "MAE rise" means the live model's MAE went up, so it got worse.

## 1. Quick start (3 terminals)

```bash
pip install -r requirements-agents.txt

# terminal 1: the baseline stack (MLflow :5000, API :8000, Prometheus, Grafana)
cd baseline && docker compose up -d --build && cd ..
python baseline/train.py                      # registers wine-quality v1 with alias champion

# terminal 2: the Trainer agent
python -m agents.trainer --db runs/demo/board.db --mlflow http://localhost:5000

# terminal 3: fake partners (replay, Data Checker, Monitor, Deployer) until the real ones are merged
python -m agents.trainer.fake_partners --db runs/demo/board.db --scenario D

# anywhere: watch the board
python -m shared.standins tail runs/demo/board.db
```

Scenario D should end with these messages:
- `training.decision` (retrain);
- `request.deploy_candidate` (candidate v2, error about 5% lower);
- the Deployer stub's `accepted`;
- the answer `Retrain request done`.

The other scenarios end differently:
- F ends with `model.candidate_rejected`: it is better overall but worse for high-alcohol wines.
- E ends with a refuse: the drift is on a feature the model does not use.

Use a new `--db` folder for each scenario run.

Or run the Trainer in Docker next to the baseline stack:

```bash
docker compose -f agents/trainer/docker-compose.yml up -d --build
docker exec trainer python -m agents.trainer.fake_partners --db /app/runs/board.db --scenario D
docker logs -f trainer
```

## 2. Where the Trainer sits

```
 replay / label feeder ──► batches, labels tables ◄── Data Checker (status passed / repaired / quarantined)
                                    │
 Monitor ── request.retrain ──────► TRAINER ── training.decision ──────────► Monitor
 Data Checker ── request.do_not_use ─►  │   ── request.deploy_candidate ─────► Deployer
 Reviewer ── request.expected_change ─► │   ── model.candidate_rejected ────► Monitor, human
                                        │
                     MLflow: reads alias champion, registers a new version with alias candidate
```

## 3. Messages the Trainer reads

All messages use the roadmap 5.5 envelope (`shared/messages.py`). The Trainer is subscribed to `dataset:*`, `model:*` and `expected_change:*`, and also gets every message with `"trainer"` in `to`.

| kind (type) | from | payload the Trainer uses | what it does |
| --- | --- | --- | --- |
| `request.retrain` (request) | Monitor | `reason`, optional `window`, `affected_features`, `expected_change_id`. Put the `incident_id` on the message: the Trainer reuses it. | Analyses, then decides and answers `done`, `refused` or `failed` (see 4). If nothing new came in since its last training, it answers right away with that result instead of training the same data twice. |
| `request.do_not_use` (request) | Data Checker | `batch_ids: [...]` | Never trains or evaluates on those batches. Answers `done`. |
| `request.expected_change` (request) | Reviewer | `change_id` | Next retrain uses only batches after the change. Answers `accepted` now and `done` (with the candidate version) after the retrain. |
| `test.ping` (question) | harness | none | Answers `pong`: the M1 "hello board" check. |

Example: [contracts/examples/request.expected_change/example.json](../../contracts/examples/request.expected_change/example.json).

**Minimal `request.retrain` from the Monitor:**

```python
board.post({"type": "request", "kind": "request.retrain", "sender": "monitor", "to": ["trainer"],
            "topic": "model:wine-quality", "incident_id": "inc-1a2b3c4d", "level": 1,
            "summary": "Live MAE up 18% over 3 windows",
            "evidence": {"mae_now": 0.61, "mae_ref": 0.52},
            "payload": {"reason": "label_mae_rise", "window": {}, "affected_features": [], "expected_change_id": None}})
```

## 4. Messages the Trainer sends

Examples of every one are in `contracts/examples/<kind>/` and in `agents/trainer/fixtures/`.

| kind (type) | to | when | payload keys |
| --- | --- | --- | --- |
| `training.decision` (finding) | monitor | Every decision. | `decision` (`retrain` or `refuse`), `expected_gain` (% MAE reduction expected), `threshold`, `data_window` (`batch_ids`, `eval_batch_ids`, `excluded`), `reason_code`, `decided_by` (`llm` / `rules` / `llm_error` / `llm_invalid`), and `candidate_run_id` when it retrained. |
| `request.deploy_candidate` (request) | deployer | The candidate passed the gate: MAE at least `min_gain_pct` lower, and no alcohol or acidity group of wines worse by more than `max_group_worse_pct` (quality bands come from the label, so they are reported but never block). | `model_uri` (`models:/wine-quality@candidate`), `version`, `champion_version`, `metrics.overall` (`mae_candidate`, `mae_champion`, `delta_pct`), `metrics.by_group`, `comparison_ref` (MLflow path of `comparison.json`). |
| `model.candidate_rejected` (finding) | monitor, human | The candidate failed the gate. | `version` (`run:<mlflow run id>`), `overall_delta`, `failing_groups` (`group`, `band`, `delta_pct`, CI), `comparison_ref`. |
| answer to `request.retrain` | monitor | Always, exactly once per request. | `decision`: `done` (candidate sent, `detail` names the version), `refused` (not worth it, or not enough clean data), or `failed` (candidate rejected, or MLflow / board.db could not be read). |

## 5. What each teammate's piece must give the Trainer

**P6, board (`shared/board.py`)**
- Implement the roadmap 5.7 API: `post`, `read(agent)`, `answer(original, **fields)`, `thread`, `heartbeat`, `ask`.
- The Trainer uses messages as dicts (`msg["kind"]`, `msg["payload"]`) and passes the same dict back to `answer()`. If `read()` returns pydantic `Message` objects, return `msg.model_dump(mode="json")` instead, or tell me and I will adapt.
- `python -m agents.trainer` picks `shared.board.Board` up automatically when the file exists; it tries `Board(db_url="sqlite:///…")`, then `Board()`. Until then it uses `SqliteBoard` in `shared/standins.py`. That stand-in uses the same tables (`messages`, `cursors`, `heartbeats`), so the other agents can already run on it. Delete it once `board.py` is merged.

**P3, base agent and clock (`shared/base_agent.py`, `shared/clock.py`)**
- The Trainer subclasses `BaseAgent` when it exists and overrides `step()`.
- It needs `self.board`, plus `safe_handle(msg)` that skips message ids it has already handled.
- It calls `shared.clock.now()` for time when that exists, so replay time works.

**P3, replay script and label feeder: tables in `board.db`**
- `batches(batch_id TEXT PRIMARY KEY, path TEXT, status TEXT, checked_at TEXT, reason TEXT)`:
  - Batch ids must sort in arrival order (`batch-0001`, `batch-0002`, …).
  - `path` points to a CSV or parquet file with a `wine_id` column and the 11 features in snake_case (`fixed_acidity`, …, `alcohol`).
- `labels(wine_id INTEGER PRIMARY KEY, quality REAL, available_at TEXT)`:
  - The Trainer only uses labels with `available_at <= now`.
- The Trainer creates these tables with `CREATE TABLE IF NOT EXISTS` if they are missing (`agents/trainer/data.py`). If yours come first, keep at least these columns.

**Data Checker**
- Set `batches.status` to `passed`, `repaired` or `quarantined`. The Trainer only trains on `passed` and `repaired`.
- Send `request.do_not_use` with `batch_ids` for anything it must skip.

**Monitor**
- Send `request.retrain` (example above) and read the answer and `training.decision`.
- The Trainer also retrains on a schedule, every `retrain_every_batches` new clean batches, without being asked.

**Deployer (P5)**
- Handle `request.deploy_candidate`: answer `accepted` or `refused`, then shadow or canary and move the `champion` alias.
- The candidate is already registered in MLflow as `wine-quality@candidate`. It has the `reference/` files (`psi_bins.json`, `feature_importance.json`, `group_metrics.json`) the Trainer and Monitor need when it becomes champion.

**Reviewer (P2)**
- `request.expected_change` with `payload.change_id`, `to` including `trainer`.
- The Trainer answers `accepted`, and `done` after the post-change retrain.
- Note for Chedi:
  - `shared/llm.py` (`decide()`) is the helper the roadmap gives P4. If your `decide_with_llm` should move into it, let's merge them.
  - `baseline/app/main.py` now loads `models:/wine-quality@champion` and has `/model` and `/reload` endpoints. Please check it is OK with you.

**Setup (P1)**
- Copy the `trainer` service in `agents/trainer/docker-compose.yml` into `docker-compose.agents.yml` and share the `runs/` volume (where `board.db` lives) with the other agent services.

## 6. Configuration

**`agents.yaml`** (repo root, optional; defaults are in `agents/trainer/agent.py`):

```yaml
db_url: sqlite:///runs/board.db
mlflow: {tracking_uri: http://localhost:5000, model_name: wine-quality}
llm: {provider: groq}                       # or ollama, nvidia_nim, off
agents:
  trainer:
    subscriptions: ["dataset:*", "model:*", "expected_change:*"]
    thresholds: {min_gain_pct: 2, max_group_worse_pct: 10, retrain_every_batches: 10}
```

**Environment / `.env`** (git ignores `.env`; never commit keys):

| variable | meaning |
| --- | --- |
| `GROQ_API_KEY`, `NVIDIA_API_KEY` | Keys for the hosted LLMs. |
| `LLM_PROVIDER` | Overrides `agents.yaml`: `ollama`, `groq`, `nvidia_nim` or `off`. |
| `LLM_MODEL`, `GROQ_MODEL` | Model names. |
| `OLLAMA_BASE_URL` | Where Ollama runs (`http://host.docker.internal:11434` from Docker). |
| `MLFLOW_TRACKING_URI` | MLflow server. `--mlflow` overrides it. |
| `RUN_ID` | Experiment run id put on every message (default `dev`). |

**Command line:**
- `python -m agents.trainer --db PATH --mlflow URL --llm config|off|groq|ollama|nvidia_nim --interval 2 --once --rounds N`
- `--once` runs one round and exits, which is good for CI.

**When the LLM fails:** with no key, a timeout or a bad answer, the Trainer uses its rule decision and keeps going (`decided_by: llm_error`). The gate is code: the LLM can choose to refuse a retrain, but it can never forward a candidate that failed the gate.

## 7. Files

| path | what |
| --- | --- |
| `agents/trainer/__main__.py` | The long-running agent: board choice, loop, logging, Ctrl+C. |
| `agents/trainer/agent.py` | `TrainerAgent`: handle, observe, analyze, decide, dry-run retrain, gate, forward or reject. |
| `agents/trainer/analysis.py` | PSI, MAE rise with bootstrap, expected gain, window choice, gate. No LLM. |
| `agents/trainer/data.py` | `batches`, `labels` and `trainer_runs` tables. |
| `agents/trainer/fake_partners.py` | Replay, Data Checker, Monitor and Deployer fakes on a real `board.db`. |
| `agents/trainer/solo.py` | Same scenarios in one process with an in-memory board (fast, used by tests). |
| `agents/trainer/stub.py` | Fake Trainer for the other five agents' own solo runs (`STUB_FIXTURE=retrain`, `refuse` or `reject`). |
| `agents/trainer/Dockerfile`, `docker-compose.yml` | The Trainer as a Docker service on the baseline network. |
| `shared/standins.py` | `SqliteBoard`, `InMemoryBoard`, `BaseAgent` and `new_message` stand-ins until P6's and P3's pieces merge. |
| `shared/llm.py`, `shared/dry_run.py`, `dry_run_jobs/train_compare.py` | LLM helper, dry-run runner and the retrain job (P4's shared pieces). |

## 8. Tests

```bash
python -m pytest -q                          # whole repo, about 45 s
python -m pytest -q agents/trainer           # Trainer only
```

`agents/trainer/tests/test_agent_loop.py` runs the real agent loop on a `board.db`:
- the board API and cursors across restarts;
- no double answers after a restart;
- the full scenario D round trip with the fake partners;
- no second model when a request comes in right after a scheduled retrain.
