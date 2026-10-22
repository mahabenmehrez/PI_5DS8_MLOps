# Data Checker (P3)

Decides for every incoming batch: **pass, pass with a warning, warn (valid shift), repair or
quarantine**, so bad data never reaches training or the live model (roadmap chapter 9).

## Run

```bash
python -m agents.checker --db board.db           # the agent loop (--llm off = rules only, --once)
python -m harness.scenario_runner C              # scenario C solo: real Checker, fakes for the others
python -m harness.evaluate_checker               # all data faults + false alarms on normal batches
python -m harness.fault_runner list              # the fault library
pytest tests/test_checker.py
```

## How it decides

| Step | What | Where |
|---|---|---|
| observe | next `pending` row of `batches` | `agent.py` |
| analyze | Pandera `WineBatch` (physical limits), nulls, learned ranges, unit changes (median **and** quartiles move by 10/100/1000), renames, duplicates by `wine_id`, PSI corrected for small-batch noise **and** confirmed by KS, Isolation Forest on batch statistics | `analysis.py`, `checks.py` |
| decide | LLM (`prompt.md`) with `rule_decision()` as fallback and as the "rules only" system | `agent.py` |
| dry run | `repair_check`: repair on a copy, all checks pass, PSI < 0.25 after | `dry_run_jobs/repair_check.py` |
| act | one function per action, table first, then the message | `actions.py` |

Rule decision:
- error that also reaches the live API (unit change, missing column, type change, feature
  invalid on most rows) or that no repair can fix → **quarantine** + `request.do_not_use`,
  and for live errors `incident.claim` + `request.hold` (once per incident);
- other errors (duplicates, a few bad rows, small null spike, renamed column) → **repair** (level 2);
- significant drift only → **warn** (`data.drift_trend`), never quarantine;
- other warnings → **pass_with_warning**; nothing that matters → **pass** (no LLM call).

Weighting: a drift counts by the feature's importance (`feature_importance.json`); a data
error on a model feature counts at least `error_floor` (0.5), because it also corrupts live
predictions and every future training set. The baseline ElasticNet uses almost only `alcohol`,
so without this floor a unit error on sulfur dioxide would be ignored.

Questions and requests handled: `query.data_changed` (window vs the 5 batches before),
`request.clean_window`, `request.expected_change` (new schema version, checked first).

## Thresholds (`agents.yaml` → `agents.checker.thresholds`)

`psi_warn` 0.10 · `psi_alert` 0.25 · `ks_p_max` 0.05 · `error_floor` 0.5 ·
`decision_min_score` 0.5 · `isolation_min_batches` 10 · `compare_last_batches` 5 ·
`unit_factors` [10, 0.1, 100, 0.01, 1000, 0.001]. Defaults in `settings.py`.

## Measured (rules, 3 seeds, the Trainer's reference)

8 data faults × 3 seeds: **100%** decided correctly, white-wine batch included;
60 normal batches: 0 blocked or repaired, 2 passed with a warning → **0.5 alarms per 15**
(targets: ≥ 90% and ≤ 1 per 15). With the real LLM (`shared/llm.py`, Groq `qwen/qwen3.8-27b`):
same decisions as the rules on the demo cases, reasons quoting the right numbers.

## Integrated with the Trainer (P4)

Built on top of `P4_Trainer_agent`; uses its shared pieces. Details in `INTEGRATION.md` §2.

| Piece | From | Used for |
|---|---|---|
| `shared/llm.py` | P4 | decisions; calls logged in `llm_calls` of the same `board.db` |
| `shared/dry_run.py` | P4 | repairs tried in a subprocess (`DRY_RUN=1`) |
| `shared/standins.SqliteBoard` | P4, until P6's board | both agents on one `board.db`, across processes |
| `samples/reference/` | P4 | reference profile and feature importance |
| experience log | P5 (not yet) | `shared.testing.InMemoryMemory` |

The Trainer trains on `batches.path`: after a repair it points to the repaired file (original in
`raw_path`). Claims are kept in the `claims` table, so a restarted Checker never claims twice.
`tests/test_integration_trainer.py` runs both real agents on one board.

## Known limits

- Leaving the learned range inside the schema's physical limits is a `range_shift` (new
  population, warning); impossible values are caught by the schema (`schema_range`, error).
- Experience memory is in-memory until P5's `shared/memory.py`.
- `release_state` in scenario end states is checked only once the Deployer runs.
