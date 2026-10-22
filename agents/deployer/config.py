"""Lecture des seuils du Deployer dans agents.yaml (section agents.deployer.thresholds)."""
import os
from typing import Optional

import yaml

from agents.deployer.analysis import DEFAULTS

ALIASES = {"p95_ms": "limit_p95_ms", "error_rate": "limit_error_rate",
           "hard_minutes": "limit_minutes"}


def load_deployer_config(path: Optional[str] = None) -> dict:
    """Renvoie la config du Deployer. Un fichier absent ou vide donne un dict vide :
    l'agent utilise alors les valeurs par defaut de analysis.DEFAULTS."""
    path = path or os.getenv("AGENTS_YAML", "agents.yaml")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    thresholds = (((raw.get("agents") or {}).get("deployer") or {}).get("thresholds")) or {}
    cfg = {}
    for key, value in thresholds.items():
        name = ALIASES.get(key, key)
        if name not in DEFAULTS:
            raise ValueError(f"seuil inconnu dans agents.yaml : {key!r}")
        cfg[name] = value
    return cfg
