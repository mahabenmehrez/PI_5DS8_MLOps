# Wine Quality MLOps — Master Implementation Plan
**Agent Role**: Agent 2: Code Reviewer (`Reviewer`) — Owner P2  
**Baseline Repository**: `BaraaZ95/FastAPI-Prometheus-Grafana-MLOPs` (cloned in `baseline/`)  
**Architecture**: 6 Autonomous Agents communicating asynchronously over an SQLite message board (`board.db`)  
**Reference Document**: [Agent Build Roadmap v2 — Wine Quality MLOps (4 weeks).pdf](file:///c:/Users/chedi/Desktop/pi%20mlops/Agent%20Build%20Roadmap%20v2%20—%20Wine%20Quality%20MLOps%20(4%20weeks).pdf)

---

## 1. High-Level Architecture & Separation of Concerns

### Six Pipeline Agents
1. **P1 (Setup Agent)**: Scans repo with AST, generates CI workflow, Makefile, Pandera schema, repairs build errors in a dry run.
2. **P2 (Code Reviewer — US)**: Inspects commits, computes static risk score, runs AST-level feature diffs against sample data to detect training-serving skew, infers intent, issues migrations, answers code queries.
3. **P3 (Data Checker)**: Validates incoming batches against Pandera schema, checks PSI/KS drift, executes repair functions in dry runs, handles batch quarantine.
4. **P4 (Trainer)**: Evaluates retraining gain, trains in isolated dry runs, enforces subgroup fairness gates (alcohol, acidity, quality bands), registers candidates in MLflow.
5. **P5 (Deployer)**: Canary rollouts (`shadow` -> `10%` -> `50%` -> `100%`), hard-limit automated rollback, promotion via human approval card.
6. **P6 (Monitor)**: Evaluates input drift and error models, tracks live MAE, queries upstream agents (`checker`, `reviewer`, `deployer`) with deadlines to identify root causes.

### Core Non-Negotiable Contract Rules
- **No Agent-to-Agent Code Imports**: Agents only import `shared/` and their own directory `agents/<name>/`.
- **Messages Announce, Tables Hold State**: Ephemeral signals travel through `board.db` (`messages` table); persistent data sits in tables (`batches`, `predictions`, `labels`, `release_state`, `memory`).
- **Standardized 7-Step Base Loop**: `observe()` -> `analyze()` -> `decide()` -> `dry_run()` -> `act()` -> `verify()` -> `remember()`.
- **Uniform Clock**: Strict usage of `shared.clock.now()`. Replay time is scaled (1 real second = 12 simulated minutes).

---

## 2. Shared Tasks vs. Individual Tasks (P2 Ownership Breakdown)

### A. Shared Tasks Owned by P2
1. **`shared/depmap.py` & `samples/dependency_map.yaml`**:
   - Implements graph navigation API:
     - `depmap.upstream(node)`
     - `depmap.downstream(node)`
     - `depmap.owner(node)`
     - `depmap.add_link(src, dst, relation)`
   - Node format: `code:<path>`, `feature:<name>`, `dataset:<batch_id>`, `model:<name>@<version>`, `deployment:prod`.
2. **`samples/feature_sample.parquet`**:
   - 500 clean reference rows extracted and saved for all feature diff tests across agents.
3. **Baseline Code Refactoring (`baseline/app/main.py` & `baseline/train.py`)**:
   - Extract pure feature transformation functions:
     - `prepare_features_train(df)` in `train.py`
     - `prepare_features_serve(values)` in `app/main.py`
   - Maintain intentional duplication so the Code Reviewer can detect training-serving skew!

### B. Individual Tasks Owned by P2 (`agents/reviewer/`)
1. **Agent Implementation**:
   - `agent.py`: Subclass of `BaseAgent`. Implements the 7-step cycle.
   - `analysis.py`: Git commit intake, AST line-to-function mapper, 0–1 risk scoring formula.
   - `actions.py`: Concrete implementations of menu actions (`approve`, `run_extra_tests`, `ask_author`, `block`, `approve_with_migration`, `warn_agents`, `request_rebuild`).
   - `prompt.md`: System prompt forcing strict JSON decision output quoting calculated numerical ratios.
   - `stub.py` & `fixtures/`: Reviewer stub with pre-baked mock responses to allow teammates (P1, P3, P4, P5, P6) to build and test in isolation.
2. **Dry-Run Runner Job (`dry_run_jobs/feature_diff.py`)**:
   - Compares old vs. new commit functions on `samples/feature_sample.parquet`.
   - Compares training vs. serving feature outputs.
   - Calculates column dtypes, identical values ratio, median ratio $\frac{new}{old}$ without importing `app/main.py` (using `ast.parse` & compilation).
3. **Scenarios Owned & Tested by P2**:
   - **Scenario A (Accidental unit change in code)**: Serving divides `total_sulfur_dioxide` by 1000 ($mg \to g$), training does not. Reviewer detects ratio $\approx 0.001$, blocks or asks author.
   - **Scenario B (Intended unit change)**: Both sides, schema and commit tag updated. Reviewer emits `request.expected_change`, waits for 4 agents to report `done`.
   - **Scenario H (predict() bug fix)**: Named input fields + clean 422 error replacing `NameError`. Coordinates migration across pipeline.

---

## 3. Communication Matrix for Reviewer (P2)

### Outgoing Messages Emitted by Reviewer:
- `code.change_assessed` (finding to subscribers on `commit:<sha>`):
  - Payload: `{commit, risk_score, verdict, affected_features[], one_sided, feature_diff{}}`
- `request.expected_change` (request to `checker`, `monitor`, `trainer`, `deployer` on `expected_change:<id>`):
  - Payload: `{change_id, commit, changes[{name, old, new, factor}], tasks{agent: task}}`
- `request.rebuild` (request to `setup` on `pipeline:setup`):
  - Payload: `{reason, changed_paths[]}`
- `query.code_changed` (answer to asker on `feature:<name>`):
  - Payload: `{commits[{sha, time, verdict, feature_changed, intended}]}`

### Incoming Messages Handled by Reviewer:
- `query.code_changed` (from `monitor` or `checker`): Answers if feature code changed since a given timestamp.
- `request.expected_change` replies (from `checker`, `monitor`, `trainer`, `deployer`): Tracks migration completion.
- `incident.closed` (from incident owners): Updates historical incident counts per file to inform risk scoring.

---

## 4. Master 4-Week Step-by-Step Implementation Roadmap

```
Week 1: Contract & Foundation  ──>  Week 2: Solo Build (on Stubs)
              │                                    │
              ▼                                    ▼
Week 4: Experiments & Report   <──  Week 3: Integration (Real Agents)
```

### Phase 1: Week 1 — Contract Sprint & Shared Baseline
- **Team Milestone**: **M1 "Hello board"** (Six skeleton agents start from Docker image, send/receive test messages on `board.db`).
- [ ] **Step 1.1: Environment & Baseline Check**
  - Verify baseline dependencies: FastAPI, MLflow, Prometheus, Grafana.
  - Reproduce baseline `predict()` bug (observe `finally` block `NameError`).
- [ ] **Step 1.2: Shared Dependency Map (`shared/depmap.py`)**
  - Define `DependencyMap` class with YAML loader/serializer.
  - Implement query methods: `upstream(node)`, `downstream(node)`, `owner(node)`, and mutation `add_link(src, dst, relation)`.
  - Author and merge `samples/dependency_map.yaml` covering code, feature, dataset, model, and deployment links.
- [ ] **Step 1.3: Reference Feature Parquet (`samples/feature_sample.parquet`)**
  - Extract 500 clean rows from `winequality-red.csv`.
- [ ] **Step 1.4: Refactor Baseline Feature Functions**
  - In `baseline/train.py`: Create `prepare_features_train(df)`.
  - In `baseline/app/main.py`: Create `prepare_features_serve(values)`.
- [ ] **Step 1.5: Reviewer Skeleton & Stub**
  - Create directory `agents/reviewer/` with `agent.py` implementing `ReviewerAgent(BaseAgent)`.
  - Implement `agents/reviewer/stub.py` and initial mock fixtures in `agents/reviewer/fixtures/`.
  - Add contract examples to `contracts/examples/code.change_assessed/` and `contracts/examples/request.expected_change/`.

---

### Phase 2: Week 2 — Solo Build (Against Stubs)
- **Team Milestone**: **M2 "Solo green"** (Every agent passes solo unit and scenario tests against mock partners).
- [ ] **Step 2.1: Commit Intake & AST Parser (`agents/reviewer/analysis.py`)**
  - Poll Git commits or inspect branches with `git fetch` / `GitPython`.
  - Parse commit diffs and map modified line ranges to AST functions without executing module code.
- [ ] **Step 2.2: Deterministic Risk Scoring Formula**
  - Implement heuristic calculation:
    - Feature function touched: $+0.35$
    - One-sided edit (train vs. serve): $+0.25$
    - `requirements.txt` or `Dockerfile`: $+0.15$
    - `agents.yaml` or config: $+0.10$
    - Diffs $> 200$ lines: $+0.05$
    - Files with past incident history: $+0.10$
- [ ] **Step 2.3: AST Feature-Diff Dry-Run Job (`dry_run_jobs/feature_diff.py`)**
  - Load old vs. new function source via `git show <sha>:<path>`.
  - Isolate AST nodes and compile in a sandbox.
  - Evaluate on `samples/feature_sample.parquet`. Output column types, identical ratios, and median multiplier (detecting $mg \to g$ $0.001\times$ shift).
- [ ] **Step 2.4: Intent Extraction & Decision Engine (`prompt.md` & `agent.py`)**
  - Extract commit signals: tags like `intent: total_sulfur_dioxide mg -> g`, conventional commits (`feat:`, `fix:`), simultaneous schema edits.
  - Implement prompt in `prompt.md` enforcing JSON response with numerical evidence.
  - Implement `rule_decision()` safe fallback.
- [ ] **Step 2.5: Solo Scenarios Verification**
  - Pass **Scenario A Solo** (Accidental change blocked or author asked with $\approx 0.001$ reported).
  - Pass **Scenario B Solo** (Migration request emitted and mock responses collected).
  - Pass **Scenario H Solo** (API bug fix approved with migration).

---

### Phase 3: Week 3 — Integration Week (Pair Sessions to Full System)
- **Team Milestones**: **M3** (Scenarios A–D pass with all 6 real agents), **M4** (Scenarios A–H pass, feature freeze).
- [ ] **Step 3.1: Pair Session 1 — Reviewer + Setup Agent (`setup`) & Monitor (`monitor`)**
  - Connect with Setup Agent: Reviewer sends `request.rebuild` on build-breaking commits; CI updates flow to Reviewer.
  - Connect with Monitor: Reviewer handles real `query.code_changed` questions.
  - **Pass Scenario A End-to-End**.
- [ ] **Step 3.2: Pair Session 2 — Migration Plan Integration**
  - Coordinate multi-agent migration with Checker (schema), Monitor (silence drift alerts), Trainer (retrain on new data), and Deployer (model rollouts).
  - **Pass Scenario B and Scenario H End-to-End**.
- [ ] **Step 3.3: Full 6-Agent Smoke Testing & False Positive Tuning**
  - Run 10 harmless commits (docs, comments, harmless renames) to ensure zero false blocks.
  - Tune thresholds in `agents.yaml`.
  - Achieve **M4 Feature Freeze**.

---

### Phase 4: Week 4 — Experiments, Benchmark & Reporting
- **Team Milestone**: **M5 "Report and Demo"**
- [ ] **Step 4.1: Benchmark Execution**
  - Compare System 1 (Traditional pipeline: rules only, board off) vs. System 2 (6 Agents on `board.db`).
  - Run 3 repeats across all fault scenarios.
- [ ] **Step 4.2: Metric Extraction & Evaluation**
  - Detection rate, Time to detect (simulated minutes), Correct owner attribution, Time to recover, Unsafe actions, LLM token and latency cost.
- [ ] **Step 4.3: Final Documentation & Demo**
  - Author Code Reviewer section of the project report.
  - Prepare demonstration run of Scenario A, B, and H.

---

## 5. Directory Structure Target State

```
wine-agents/
├── baseline/                   # Cloned BaraaZ95 repo + pure feature refactor
│   ├── app/main.py             # FastAPI API (prepare_features_serve)
│   ├── train.py                # ElasticNet training (prepare_features_train)
│   └── docker-compose.yaml
├── data/                       # winequality-red.csv, winequality-white.csv
├── shared/                     # Frozen Contract Module
│   ├── depmap.py               # [P2 OWNED] Dependency graph traversal & API
│   ├── messages.py             # Pydantic envelope definitions
│   ├── catalogue.yaml          # Registered message kinds & schemas
│   ├── board.py                # SQLite message board (board.db)
│   ├── base_agent.py           # 7-step loop base class
│   ├── llm.py                  # Structured LLM invocation & retry fallback
│   ├── dry_run.py              # Isolated subprocess execution engine
│   ├── memory.py               # Incident memory & text-overlap retrieval
│   ├── tables.py               # SQLite schema & DB session helpers
│   ├── incidents.py            # Incident grouping & deduplication
│   ├── clock.py                # Simulated replay time clock
│   ├── config.py               # agents.yaml reader
│   └── testing.py              # Test runners & scenario harnesses
├── agents/reviewer/            # [P2 OWNED] Code Reviewer Agent
│   ├── agent.py                # ReviewerAgent(BaseAgent)
│   ├── analysis.py             # AST-based diff parser & risk scoring
│   ├── actions.py              # Action implementations
│   ├── prompt.md               # LLM system prompt
│   ├── stub.py                 # Mock stub for teammates
│   ├── fixtures/               # Mock message fixtures
│   └── tests/                  # Unit tests & solo scenario tests
├── dry_run_jobs/
│   └── feature_diff.py         # [P2 OWNED] AST function evaluator job
├── harness/
│   └── faults/code/            # [P2 OWNED] Code faults for Scenarios A, B, H
├── samples/
│   ├── dependency_map.yaml     # [P2 OWNED] Baseline dependency map seed
│   ├── feature_sample.parquet  # [P2 OWNED] 500 clean reference rows
│   └── ...
├── contracts/examples/         # Example JSON payloads per message kind
├── scenarios/                  # Scenario definitions (A.yaml to H.yaml)
├── agents.yaml                 # Master system configuration
├── Dockerfile.agents           # Unified Docker container for all agents
├── Makefile                    # Targets: contract-test, solo, scenario, experiment
├── PROJECT_CONTEXT.md          # Persistent engineering state
└── IMPLEMENTATION_PLAN.md      # This master plan
```
