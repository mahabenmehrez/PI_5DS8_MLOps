# PROJECT CONTEXT & ARCHITECTURE STATE: CODE REVIEWER (P2)
**Project**: Wine Quality MLOps Multi-Agent System (4-Week Roadmap v2)  
**Assigned Agent Role**: **Agent 2: Code Reviewer (`Reviewer`) — Owner P2**  
**Last Updated**: October 13, 2026 (Week 1 Foundation Complete)  
**Status**: Week 1 Foundation & Reviewer Stub Complete (All 5 Tests Passing)  

---

## 1. Executive Summary & Architecture Overview

The project sits on top of the wine quality prediction baseline (`FastAPI + MLflow + MinIO + Prometheus + Grafana`) predicting wine quality score (0 to 10) from 11 chemical features.
Rather than a traditional static pipeline or a monolithic agent, **six autonomous agents** interact asynchronously via an SQLite message board (`board.db`):
1. **P1: Setup Agent (`setup`)** - Pipeline runnable & build repair.
2. **P2: Code Reviewer (`reviewer`) [OUR ROLE]** - Commit review, training-serving feature drift/skew detection, migration coordination, rebuild requests.
3. **P3: Data Checker (`checker`)** - Data validation, drift, quarantine & repair.
4. **P4: Trainer (`trainer`)** - Retraining decisions, fair group gates, candidate registration.
5. **P5: Deployer (`deployer`)** - Canary rollouts, traffic split, automatic rollbacks.
6. **P6: Monitor (`monitor`)** - Early/late drift & error monitoring, upstream blame attribution.

### Core Architectural Principles:
- **Zero Direct Import**: Agents never import each other's code. Communication is strictly through messages in `board.db`.
- **Messages Announce, Tables Hold State**: Durable states live in SQLite tables; messages notify agents of events/queries.
- **7-Step Agent Loop** (`shared/base_agent.py`):
  1. `observe()` -> 2. `analyze()` -> 3. `decide()` -> 4. `dry_run()` -> 5. `act()` -> 6. `verify()` -> 7. `remember()`.
- **Action Levels**: Level 0 (Info), Level 1 (Autonomous safe), Level 2 (Risky, requires dry run), Level 3 (Critical, requires human CLI approval).
- **Time Representation**: All agents strictly use `shared.clock.now()` (UTC / replay time speedup 1s = 12m).

---

## 2. Current Status & Deliverables Summary (Week 1 Complete)

| Component | Status | Location / Artifact |
|---|---|---|
| **Baseline Repository** | Cloned & Refactored | `baseline/` |
| **Dataset Source** | Downloaded | `data/winequality-red.csv` (1,599 rows) |
| **Dependency Map API** | Complete & Tested | [`shared/depmap.py`](file:///c:/Users/chedi/Desktop/pi%20mlops/shared/depmap.py) |
| **Dependency Map Seed** | Complete & Tested | [`samples/dependency_map.yaml`](file:///c:/Users/chedi/Desktop/pi%20mlops/samples/dependency_map.yaml) |
| **Clean Feature Sample** | Complete | [`samples/feature_sample.parquet`](file:///c:/Users/chedi/Desktop/pi%20mlops/samples/feature_sample.parquet) (500 rows) |
| **Pure Feature Serving** | Extracted | `prepare_features_serve(values)` in [`baseline/app/main.py`](file:///c:/Users/chedi/Desktop/pi%20mlops/baseline/app/main.py) |
| **Pure Feature Training** | Implemented & Tested | `prepare_features_train(df)` in [`baseline/train.py`](file:///c:/Users/chedi/Desktop/pi%20mlops/baseline/train.py) |
| **Message Model** | Implemented | [`shared/messages.py`](file:///c:/Users/chedi/Desktop/pi%20mlops/shared/messages.py) (Pydantic v2) |
| **Contract Examples** | Authored | `contracts/examples/code.change_assessed/example.json`, `contracts/examples/request.expected_change/example.json` |
| **Reviewer Stub** | Implemented & Tested | [`agents/reviewer/stub.py`](file:///c:/Users/chedi/Desktop/pi%20mlops/agents/reviewer/stub.py) |
| **Unit Test Suite** | 5 / 5 Passing | [`tests/test_week1_foundation.py`](file:///c:/Users/chedi/Desktop/pi%20mlops/tests/test_week1_foundation.py), [`tests/test_reviewer_stub.py`](file:///c:/Users/chedi/Desktop/pi%20mlops/tests/test_reviewer_stub.py) |

---

## 3. Communication Contract for Code Reviewer

### Messages P2 Emits:
| Message Kind | Type | Topic | Target Receivers | Key Payload Fields |
|---|---|---|---|---|
| `code.change_assessed` | finding | `commit:<sha>` | subscribers (`checker`, `monitor`, `trainer`) | `commit`, `risk_score`, `verdict`, `affected_features[]`, `one_sided`, `feature_diff{}` |
| `request.expected_change` | request | `expected_change:<id>` | `checker`, `monitor`, `trainer`, `deployer` | `change_id`, `commit`, `changes[{name, old, new, factor}]`, `tasks{agent: task}` |
| `request.rebuild` | request | `pipeline:setup` | `setup` | `reason`, `changed_paths[]` |
| `query.code_changed` (answer) | answer | `feature:<name>` | asker (`monitor`, `checker`) | `commits[{sha, time, verdict, feature_changed, intended}]` |

### Messages P2 Consumes / Handles:
| Message Kind | Type | From | P2 Action / Handling |
|---|---|---|---|
| `query.code_changed` | question | `monitor`, `checker` | Check commit history affecting feature since timestamp; return answer. |
| `request.expected_change` (reply) | answer | `checker`, `monitor`, `trainer`, `deployer` | Track migration task completion (`done`). On timeout, escalate to human. |
| `incident.closed` | finding | incident owners | Update incident history weights per file for risk scoring. |

---

## 4. Next Step: Week 2 Solo Build
1. **Commit Intake & AST Diff Scanner** (`agents/reviewer/analysis.py`): Extract function line boundaries via Python `ast` from Git diffs without importing.
2. **Deterministic Risk Score Calculation**: Compute numeric score $0.0 \to 1.0$.
3. **Feature-Diff Dry-Run Job** (`dry_run_jobs/feature_diff.py`): Run AST-extracted functions on `samples/feature_sample.parquet` to calculate column shifts, identity ratios, and scale multipliers (e.g. $0.001\times$ detection).
4. **Intent Recognition & LLM Decision Loop**: Parse commit messages for explicit intent vs. unintentional bugs; trigger `approve_with_migration` vs `ask_author` / `block`.
