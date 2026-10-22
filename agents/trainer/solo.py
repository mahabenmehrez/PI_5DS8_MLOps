"""Solo runs of the Trainer against fakes (roadmap 4.2 week 2, 13.1).

A small stand-in for the replay script and label feeder (owner P3): it cuts
the red-wine data into batches of 100, applies fault files, fills the
batches and labels tables of a fresh board.db, and marks every batch
passed (what the Data Checker stub does). Then a Monitor stub sends
request.retrain and the real Trainer handles it.

    python -m agents.trainer.solo --scenario D     # E, F, bad_candidate, none
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
FAULTS = ROOT / "harness" / "faults"
T0 = datetime(2026, 3, 1, tzinfo=timezone.utc)
FMT = "%Y-%m-%dT%H:%M:%SZ"

# Data-layer fault the Trainer needs for scenario E. The real one belongs to the
# Data Checker owner (harness/faults/data/); this stand-in goes when theirs lands.
E_RESIDUAL_SUGAR = {
    "id": "data.valid_shift.residual_sugar", "layer": "data", "target": {"column": "residual_sugar", "factor": 1.5},
    "start": {"after_batches": 10}, "duration": {"batches": 30}, "applies_to": ["batch"], "inject": "scale",
}

SCENARIOS = {
    # D: tasters score high-alcohol wines 1 point lower (labels change, inputs do not)
    "D": [FAULTS / "model" / "concept_drift_gradual.yaml"],
    # E: drift on a feature the model does not use
    "E": [E_RESIDUAL_SUGAR],
    # F: candidate better on average, worse for high-alcohol wines
    "F": [{**yaml.safe_load((FAULTS / "model" / "concept_drift_sudden.yaml").read_text()),
           "target": {"alcohol_le": 11.5, "delta": -1}},
          FAULTS / "model" / "group_damage_high_alcohol.yaml"],
    # a real drift asks for a retrain, but half the window has scrambled labels
    "bad_candidate": [FAULTS / "model" / "concept_drift_gradual.yaml", FAULTS / "model" / "bad_candidate.yaml"],
    # every wine scored 1 point lower (all quality bands move): must retrain and forward
    "sudden": [FAULTS / "model" / "concept_drift_sudden.yaml"],
    "none": [],
}


def _load_fault(spec) -> dict:
    return spec if isinstance(spec, dict) else yaml.safe_load(Path(spec).read_text())


def _inject(fault: dict):
    if fault["inject"] == "scale":
        col, factor = fault["target"]["column"], fault["target"]["factor"]
        return lambda b, step, target: b.assign(**{col: b[col] * factor})
    module, fn = fault["inject"].split(":")
    return getattr(importlib.import_module(module), fn)


@dataclass
class Replay:
    db_path: Path
    workdir: Path
    n_batches: int
    now: str
    faults: list[dict]


def build_replay(workdir: str | Path, faults: list, *, n_batches: int = 40, batch_size: int = 100,
                 seed: int = 0, labels_delay_batches: int = 0) -> Replay:
    """Write batch files and a board.db with batches (all passed) and labels."""
    from agents.trainer.data import TrainerStore

    from baseline import train

    workdir = Path(workdir).resolve()  # absolute batch paths: the dry run reads them from another folder
    (workdir / "batches").mkdir(parents=True, exist_ok=True)
    base = train.load_wines(train.DEFAULT_DATA).rename(columns=lambda c: c.replace(" ", "_")).drop(columns=["wine_id"])
    rng = np.random.default_rng(seed)
    faults = [_load_fault(f) for f in faults]

    store = TrainerStore(workdir / "board.db")
    store.create_tables()
    rows_b, rows_l = [], []
    for i in range(n_batches):
        batch = base.iloc[rng.integers(0, len(base), batch_size)].reset_index(drop=True)
        batch.insert(0, "wine_id", np.arange(i * batch_size, (i + 1) * batch_size))  # replay row id
        for fault in faults:
            start, dur = fault["start"]["after_batches"], fault["duration"]["batches"]
            if start <= i < start + dur:
                batch = _inject(fault)(batch, i - start, fault["target"])
        bid = f"batch-{i + 1:04d}"
        path = workdir / "batches" / f"{bid}.csv"
        batch.drop(columns=["quality"]).to_csv(path, index=False)
        t = T0 + timedelta(hours=i)
        rows_b.append((bid, str(path), "passed", t.strftime(FMT), None))
        avail = (t + timedelta(hours=labels_delay_batches)).strftime(FMT)
        rows_l += [(int(w), float(q), avail) for w, q in zip(batch["wine_id"], batch["quality"])]
    with store.connect() as con:
        con.executemany("INSERT INTO batches VALUES (?,?,?,?,?)", rows_b)
        con.executemany("INSERT INTO labels VALUES (?,?,?)", rows_l)
    now = (T0 + timedelta(hours=n_batches + labels_delay_batches)).strftime(FMT)
    return Replay(workdir / "board.db", workdir, n_batches, now, faults)


def setup_champion(tracking_uri: str) -> str:
    """Train and register wine-quality v1 (alias champion) on the original data."""
    from baseline import train

    return train.train(train.load_wines(train.DEFAULT_DATA), tracking_uri=tracking_uri,
                       reference_dir=Path(tempfile.mkdtemp(prefix="ref_"))).version


def has_champion(tracking_uri: str) -> bool:
    import mlflow

    try:
        mlflow.MlflowClient(tracking_uri=tracking_uri).get_model_version_by_alias("wine-quality", "champion")
        return True
    except Exception:
        return False


def run_scenario(name: str, workdir: str | Path, *, seed: int = 0, llm_client=None, llm_config=None,
                 settings: dict | None = None, tracking_uri: str | None = None) -> dict:
    """Replay a scenario, send one request.retrain from the Monitor stub, run the Trainer once.

    tracking_uri None: a private MLflow folder in workdir. Given (e.g. the Docker MLflow at
    http://localhost:5000): the Trainer uses that server, and its candidate shows up in the MLflow UI.
    """
    from agents.trainer.agent import SUBSCRIPTIONS, TrainerAgent
    from agents.trainer.data import TrainerStore
    from shared.standins import InMemoryBoard

    workdir = Path(workdir)
    if tracking_uri is None:
        tracking_uri = f"file:{(workdir / 'mlruns').resolve()}"
        if not (workdir / "mlruns").exists():
            setup_champion(tracking_uri)
    elif not has_champion(tracking_uri):
        setup_champion(tracking_uri)
    target = workdir / name
    shutil.rmtree(target, ignore_errors=True)  # a fresh replay each run
    if target.exists():  # Windows keeps files that something still holds open
        target = workdir / f"{name}-{uuid.uuid4().hex[:6]}"
    replay = build_replay(target, SCENARIOS[name], seed=seed)
    board = InMemoryBoard({"trainer": SUBSCRIPTIONS})
    agent = TrainerAgent(board, store=TrainerStore(replay.db_path),
                         settings={"tracking_uri": tracking_uri, **(settings or {})},
                         llm_client=llm_client, llm_config=llm_config or {"provider": "off", "db_url": None},
                         now=lambda: replay.now)
    board.post({"type": "request", "kind": "request.retrain", "sender": "monitor", "to": ["trainer"],
                "topic": "model:wine-quality", "incident_id": "inc-" + name.lower().ljust(8, "0")[:8], "level": 1,
                "summary": f"Scenario {name}: degradation detected", "expires_at": replay.now,
                "payload": {"reason": "degradation", "window": {}, "affected_features": [], "expected_change_id": None}})
    agent.step()
    return {"messages": board.messages, "log": agent.log, "replay": replay}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="D", choices=sorted(SCENARIOS))
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--llm", default="off",
                    help="off (rules only) | config (the provider in agents.yaml) | ollama | nvidia_nim | groq | fake")
    ap.add_argument("--mlflow", default=None, metavar="URL",
                    help="MLflow server to use, e.g. http://localhost:5000 (default: a private folder)")
    args = ap.parse_args()
    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="trainer_solo_"))
    from shared.llm import config_for, load_config

    provider = load_config()["provider"] if args.llm == "config" else args.llm
    llm_config = {**config_for(provider), "db_url": f"sqlite:///{workdir / 'llm_calls.db'}"}
    if provider not in ("off", "fake"):
        print(f"LLM: {provider} {llm_config.get('model')} (falls back to the rules if it fails)")
    if args.mlflow:
        print(f"MLflow: {args.mlflow}")
    out = run_scenario(args.scenario, workdir, llm_config=llm_config, tracking_uri=args.mlflow)
    print("(MAE = average prediction error in quality points: lower is better)")
    for m in out["messages"]:
        if m["sender"] == "trainer":
            print(f"{m['type']:8} {m['kind']:28} {m['summary']}")
    print("decided by:", ", ".join(f"{d['phase']}={d['source']}" for d in out["log"] if "source" in d))
    from shared.llm import CALL_LOG

    for rec in CALL_LOG:
        if rec.error:
            print(f"LLM {rec.outcome} ({rec.agent}, attempt {rec.attempt}): {rec.error}")
    print(json.dumps(out["log"], indent=1, default=str))


if __name__ == "__main__":
    main()
