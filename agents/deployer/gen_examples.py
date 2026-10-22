"""
Genere un exemple JSON valide pour chaque message que le Deployer envoie
(contracts/examples/<kind>/deployer_<type>.json), en faisant tourner le vrai agent.

    python -m agents.deployer.gen_examples
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

from agents.deployer.replay import simulated_replay      # noqa: E402
from deployer_live import LiveHarness                    # noqa: E402
from shared.messages import Message, MessageType         # noqa: E402

KINDS = ["deployment.stage_changed", "deployment.rolled_back", "request.approval",
         "incident.closed", "request.deploy_candidate", "request.hold",
         "query.deployment_changed"]


def _collect() -> dict:
    found: dict = {}

    def take(messages):
        for m in messages:
            found.setdefault((m.kind, m.type.value), m)

    # 1. candidat sain jusqu'a la demande d'approbation humaine
    h = LiveHarness("Candidat sain", 42, replay=simulated_replay())
    h.send_candidate()
    take(h.agent.outbox)
    h.send_hold("Data Checker : unite suspecte sur total_sulfur_dioxide")
    take(h.agent.outbox)
    h.release_hold()
    for _ in range(200):
        if h.agent.pending:
            break
        h.tick()
    take(h.agent.outbox)
    h.agent.step([Message(type=MessageType.QUESTION, kind="query.deployment_changed",
                          sender="monitor", to=["deployer"], topic="deployment:prod",
                          level=1, summary="Le deploiement a-t-il change ?",
                          payload={"since": "2000-01-01T00:00:00+00:00"})])
    take(h.agent.outbox)

    # 2. candidat lent : rollback automatique puis verification (incident.closed)
    h2 = LiveHarness("Candidat lent (scenario G)", 42, replay=simulated_replay())
    h2.send_candidate()
    for _ in range(120):
        if h2.release_state()["stage"] == "none":
            break
        h2.tick()
    h2.settle()
    take(h2.agent.outbox)
    return found


def generate(out_dir: str = "contracts/examples") -> list[Path]:
    found = _collect()
    written = []
    for kind in KINDS:
        matches = [m for (k, _), m in found.items() if k == kind]
        if not matches:
            raise RuntimeError(f"aucun message de type {kind} produit par l'agent")
        for m in matches:
            path = Path(out_dir) / kind / f"deployer_{m.type.value}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(m.model_dump_json(indent=2), encoding="utf-8")
            written.append(path)
    return written


if __name__ == "__main__":
    for p in generate():
        print("ecrit :", p)
