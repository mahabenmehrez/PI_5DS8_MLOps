# Trainer agent (P4) — weeks 1 and 2

What is in place after week 2, how to run it, and what is still a stand-in.

**Teammates: to plug the Trainer into the pipeline, read [INTEGRATION.md](INTEGRATION.md)** (messages in and out, tables, what your piece must provide, Docker service).

## Files

| Path | What it is |
| --- | --- |
| `baseline/train.py` | `prepare_features_train`, `train()`, CLI. Pipeline(StandardScaler, ElasticNet alpha=0.01 l1_ratio=0.5), seed 42, 400-row holdout. Registers `wine-quality` and sets alias `champion`. |
| `data/winequality-red.csv`, `data/winequality-white.csv` | UCI wine quality data, original semicolon format (1,599 and 4,898 rows). |
| `samples/reference/` | `reference.parquet` (400 holdout rows: wine_id, 11 features in snake_case, quality, prediction, 3 group bands), `feature_importance.json`, `psi_bins.json`, `group_metrics.json`. |
| `shared/llm.py` | `decide(agent, context, menu, fallback)`: JSON answer checked against the menu, one retry, then the rule decision (`llm_invalid` / `llm_error`). Providers: `ollama`, `nvidia_nim` (any OpenAI-style API, key from `llm.api_key_env`), `fake`, `off`; `api` is a placeholder until the API is chosen. Logs every call to `llm_calls`. |
| `shared/dry_run.py` | `dry_run(kind, inputs, timeout_s=120)` runs `dry_run_jobs/<kind>.py` in a subprocess in a temp folder with `DRY_RUN=1`; `guard_real_write()` for `shared/tables.py`. |
| `shared/standins.py` | In-memory board, `SqliteBoard` (the same API on board.db, for several processes) and a minimal base loop until P6's `board.py` and P3's `base_agent.py` merge. Delete it then. `python -m shared.standins tail board.db` prints the board live. |
| `agents/trainer/__main__.py` | `python -m agents.trainer`: the long-running agent on board.db (round every 2 s, heartbeat, logging, Ctrl+C). |
| `agents/trainer/fake_partners.py` | Replay, Data Checker, Monitor and Deployer fakes that talk to the running agent through board.db. |
| `agents/trainer/Dockerfile`, `agents/trainer/docker-compose.yml` | The agent as a Docker service on the baseline network. |
| `agents/trainer/agent.py` | `TrainerAgent`: reads requests, measures drift and MAE rise, decides retrain or refuse (LLM with a rule fallback), retrains in a dry run, applies the gate, then sends `request.deploy_candidate` or `model.candidate_rejected`. Honours `request.do_not_use` and `request.expected_change`. |
| `agents/trainer/analysis.py` | PSI and weighted PSI, gain estimate v0, MAE rise with a bootstrap bound, window choice, candidate vs live comparison with a paired bootstrap, the gate. No LLM. |
| `agents/trainer/data.py` | `TrainerStore`: reads `batches` and `labels` from board.db, writes `trainer_runs`. |
| `dry_run_jobs/train_compare.py` | The retrain job: train on the window, predict new and live on the eval batches, compare, gate, log `comparison.json` to MLflow. |
| `agents/trainer/solo.py` | Solo runs against stubs: replays 40 batches with a fault file, the Monitor stub asks for a retrain, the real Trainer answers. |
| `harness/faults/model/` | `concept_drift_gradual`, `concept_drift_sudden`, `group_damage_high_alcohol`, `bad_candidate` and their inject functions. |
| `agents/trainer/stub.py` | Fake Trainer for the other five: `STUB_FIXTURE=retrain` (D) / `refuse` (E) / `reject` (F). |
| `contracts/examples/` | One or more examples for each kind the Trainer sends. |

## Run

```bash
pip install -r requirements-agents.txt

# Against the docker compose stack (MLflow on :5000)
cd baseline && docker compose up -d --build && cd ..
python baseline/train.py

# Without the stack
python baseline/train.py --tracking-uri file:./mlruns_local

# The real agent, long-running on board.db (terminal 1), and its fake partners (terminal 2)
python -m agents.trainer --db runs/demo/board.db --mlflow http://localhost:5000
python -m agents.trainer.fake_partners --db runs/demo/board.db --scenario D

# Solo scenarios in one process (no docker needed, MLflow is a local file store)
python -m agents.trainer.solo --scenario D          # also E, F, bad_candidate, none
python -m agents.trainer.solo --scenario D --llm ollama

# Tests (about 40 s)
python -m pytest -q
```

## LLM

Free and local: [Ollama](https://ollama.com) with `qwen2.5:3b` (about 2 GB, runs on a laptop CPU).

```bash
ollama pull qwen2.5:3b
```

Or the team's hosted model, NVIDIA NIM `google/gemma-4-31b-it` (the Reviewer's choice): put `NVIDIA_API_KEY=...` in `.env` at the repo root (git ignores it) and run with `--llm nvidia_nim`, or `--llm config` to use whatever `agents.yaml` names.

Or Groq (free key from console.groq.com/keys): put `GROQ_API_KEY=...` and optionally `GROQ_MODEL=...` (default `qwen/qwen3.8-27b`) in `.env`, and run with `--llm groq`.

Ollama is the default in `shared/llm.py`; `LLM_PROVIDER`, `LLM_MODEL` and `LLM_BASE_URL` override it. When Ollama is not running or answers badly twice, the Trainer uses its rule decision, so it never blocks on the LLM. The gate is code: the LLM can choose to refuse a retrain, but it cannot forward a candidate that failed the gate.

## Week 2 results

The model is scored by MAE (mean absolute error, in quality points): **lower is better**. "Gain" always means how much lower the MAE gets; "MAE rise" means the live model's MAE went up (got worse).

One retrain request per scenario, rules only, seeds 0 to 4:

| Scenario | Fault | Trainer does | Result |
| --- | --- | --- | --- |
| D | high-alcohol wines scored 1 point lower, ramping in | retrains and forwards (new model's MAE about 5% lower) | 5 of 5 seeds |
| E | residual_sugar x1.5 (the model gives it weight 0) | refuses: retraining would lower MAE by only about 0.7% | 5 of 5 |
| F | low-alcohol labels shift, high-alcohol rows dropped | retrains, rejects: MAE about 35% lower overall but higher for alcohol >11.5 | 3 of 3 |
| bad_candidate | D plus scrambled labels | retrains, rejects (MAE higher overall) | 3 of 3 |
| none | no fault | refuses | 5 of 5 |

## Calibration (changes from the roadmap's starting values)

- **Group rule.** Roadmap 10.1 starts from "upper 95% bound of the group difference below +5%". On unchanged data that rule flagged a worse group in 8 of 9 candidates. A group now fails when it is worse by more than `max_group_worse_pct` of its live MAE and the lower 95% bound is above 0.
- **Only input groups can block a candidate.** `alcohol_band` and `acidity_band` can make it fail; `quality_band` is built from the label itself, so it is only reported (`label_groups_worse` in `comparison.json`). When every wine is scored one point lower, wines change quality band and the old model looks good on band 6 by chance: with quality_band as a veto, that candidate (42-45% lower error, 3 seeds) was rejected. Scenario `sudden` covers it.
- **`max_group_worse_pct` is 10**, not 5. Noise alone moves a 100-wine group by about 5%.
- **`eval_batches` is 5** (500 rows), so the alcohol >11.5 band has at least 50 wines.
- **Gain estimate.** `a*weighted_psi + b*mae_rise + c*log1p(new_rows/1000)` with a=10, b=0.5, c=0.3. It uses the lower 95% bound of the MAE rise, measured on the newest 15 batches (`rise_batches`). With the point estimate on 500 rows, a no-fault replay sometimes read +5% and triggered a retrain.

All of these live in `agents.yaml` under `agents.trainer.thresholds`.

The holdout MAE of version 1 is 0.502 quality points. A constant prediction scores 0.669.

## Known limits

- MinIO's images were removed from Docker Hub. `baseline/docker-compose.yaml` still uses them; if `docker compose up` fails to pull `minio/mc`, the fix is to drop MinIO and let MLflow serve artifacts itself (`--serve-artifacts`). The Trainer's solo runs and tests do not need docker.

- The API (`app/main.py`) still loads a hard-coded run id. Switching it to `models:/wine-quality@champion` is P5's week-1 fix.
- The model is logged without an input signature on purpose: the API sends plain lists, and integers such as `citric acid = 0` would fail a strict float schema.
- `psi_bins.json` cut points come from the 1,199 training rows. A feature with many tied values can get fewer than 10 bins.
- `llm.provider: api` is not wired up yet.
- Scenario E's fault file is a stand-in in `solo.py`; the real one belongs to the Data Checker owner.
- `request_clean_window` is in the menu but the Trainer reads batch status straight from board.db; it asks the Data Checker only once that agent exists.
