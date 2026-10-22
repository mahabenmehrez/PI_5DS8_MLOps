"""
Experience : pipeline traditionnel contre agent Deployer (roadmap 13.6, partie du Deployer).

Lancer depuis la RACINE du depot :
    python -m evaluation.experiment                 # 4 scenarios x 2 systemes x 3 repetitions
    python -m evaluation.experiment --repeats 5

Les deux systemes voient le MEME candidat et la MEME simulation de trafic (60 requetes par minute,
horizon de 3 heures simulees).

  traditionnel : le nouveau modele est mis a 100 % tout de suite, aucun rollback automatique.
  agent        : le vrai DeployerAgent (regles, sans LLM) : shadow, 10 %, 50 %, dry run de charge,
                 rollback d'urgence, approbation humaine pour le 100 %.

Metriques (par scenario et par systeme, moyennes sur les repetitions) :
  - requetes defectueuses servies : requetes d'utilisateurs servies par un candidat defectueux
  - promu a 100 % un candidat defectueux (action dangereuse)
  - minutes avant promotion d'un bon candidat (le cout de la prudence)
  - cartes humaines
Limites : trafic et pannes simules ; un candidat "defectueux" l'est des le debut.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

from agents.deployer.replay import simulated_replay      # noqa: E402
from deployer_live import SCENARIOS, LiveHarness         # noqa: E402

HORIZON = 180      # minutes simulees


def run_traditional(scenario: str, seed: int, horizon: int = HORIZON) -> dict:
    h = LiveHarness(scenario, seed)
    bad = bool(h.sc.get("bad"))
    served = horizon * h.REQ_PER_MIN              # 100 % du trafic des la premiere minute
    return {"system": "traditionnel", "scenario": scenario, "seed": seed,
            "outcome": "promu", "bad_served": served if bad else 0,
            "promoted_bad": bad, "minutes_to_promote": 0, "human_cards": 0}


def run_agent(scenario: str, seed: int, horizon: int = HORIZON) -> dict:
    h = LiveHarness(scenario, seed, replay=simulated_replay())
    bad = bool(h.sc.get("bad"))
    h.send_candidate()
    outcome, cards, promote_minute = "bloque", 0, None
    for _ in range(horizon):
        if h.agent.pending:
            cards += 1
            h.human_decision(True)                # un humain approuve ; la carte est comptee
            outcome, promote_minute = "promu", h.minute
            break
        if h.release_state()["stage"] == "none":
            break
        h.tick()
    if outcome != "promu" and h.agent.history and h.agent.history[-1]["to_stage"] == "rolled_back":
        outcome = "rollback"
    h.settle()
    return {"system": "agent", "scenario": scenario, "seed": seed, "outcome": outcome,
            "bad_served": h.bad_served, "promoted_bad": bad and outcome == "promu",
            "minutes_to_promote": promote_minute, "human_cards": cards,
            "rollback_verified": any(r["ok"] for r in h.agent.recovered)}


def summarize(runs: list[dict]) -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        for system in ("traditionnel", "agent"):
            rs = [r for r in runs if r["scenario"] == scenario and r["system"] == system]
            promote = [r["minutes_to_promote"] for r in rs if r["minutes_to_promote"] is not None]
            rows.append({
                "scenario": scenario, "systeme": system, "repetitions": len(rs),
                "resultats": "/".join(sorted({r["outcome"] for r in rs})),
                "requetes_defectueuses_servies": round(statistics.mean(r["bad_served"] for r in rs)),
                "promus_a_100_defectueux": sum(r["promoted_bad"] for r in rs),
                "minutes_avant_promotion": round(statistics.mean(promote)) if promote else None,
                "cartes_humaines": round(statistics.mean(r["human_cards"] for r in rs), 1)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    args = ap.parse_args()
    runs = []
    for scenario in SCENARIOS:
        for i in range(args.repeats):
            runs.append(run_traditional(scenario, 42 + i, args.horizon))
            runs.append(run_agent(scenario, 42 + i, args.horizon))
    table = summarize(runs)

    print(f"\nHorizon : {args.horizon} min simulees, {args.repeats} repetitions par cas\n")
    head = ["scenario", "systeme", "resultats", "requetes_defectueuses_servies",
            "promus_a_100_defectueux", "minutes_avant_promotion", "cartes_humaines"]
    print(" | ".join(head))
    for r in table:
        print(" | ".join(str(r[k]) for k in head))

    out = Path("evaluation")
    out.mkdir(exist_ok=True)
    with open(out / "results_experiment.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (out / "results_experiment.json").write_text(
        json.dumps({"summary": table, "runs": runs}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nResultats enregistres dans evaluation/results_experiment.csv et .json")


if __name__ == "__main__":
    main()
