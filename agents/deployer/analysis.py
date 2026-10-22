"""Statistiques et decisions du Deployer. Pas de LLM ici (roadmap 11.2)."""
import numpy as np
from scipy.stats import mannwhitneyu

STAGES = ["none", "shadow", "10%", "50%", "100%"]
SHARES = {"none": 0.0, "shadow": 0.0, "10%": 0.10, "50%": 0.50, "100%": 1.0}

# valeurs de depart (a mettre dans agents.yaml)
DEFAULTS = {
    "shadow_min": 300, "shadow_min_labelled": 100,
    "stage_min_minutes": 30, "stage_min_requests": 150,
    "latency_ratio": 1.2, "latency_floor_ms": 10.0,
    "error_delta": 0.005, "alpha": 0.05,
    "limit_p95_ms": 300.0, "limit_error_rate": 0.05, "limit_minutes": 2,
}


def summarize(rows):
    """Resume une liste de lignes de predictions : n, p95_ms, error_rate."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "p95_ms": 0.0, "error_rate": 0.0}
    errors = sum(1 for r in rows if r["status_code"] >= 500)
    p95 = float(np.percentile([r["latency_ms"] for r in rows], 95))
    return {"n": n, "p95_ms": p95, "error_rate": errors / n}


def hard_limit_breached(minute_windows, cfg=None):
    """minute_windows : un resume par minute simulee, le plus recent en dernier.
    Renvoie le declencheur si la limite est depassee N minutes de suite."""
    cfg = {**DEFAULTS, **(cfg or {})}
    k = cfg["limit_minutes"]
    if len(minute_windows) < k:
        return None
    last = minute_windows[-k:]
    if all(w["n"] > 0 and w["error_rate"] > cfg["limit_error_rate"] for w in last):
        return {"metric": "error_rate", "value": last[-1]["error_rate"],
                "limit": cfg["limit_error_rate"]}
    if all(w["n"] > 0 and w["p95_ms"] > cfg["limit_p95_ms"] for w in last):
        return {"metric": "p95_ms", "value": last[-1]["p95_ms"],
                "limit": cfg["limit_p95_ms"]}
    return None


def errors_not_worse(cand_abs, champ_abs, alpha=0.05):
    """Mann-Whitney U unilateral : vrai si on ne prouve PAS que le candidat est pire."""
    if len(cand_abs) < 2 or len(champ_abs) < 2:
        return True
    _, p = mannwhitneyu(cand_abs, champ_abs, alternative="greater")
    return bool(np.isnan(p) or p >= alpha)


def evaluate_gate(stage, cand, champ, cand_abs, champ_abs, minutes_in_stage, cfg=None):
    """Renvoie {'ok', 'waiting': [...], 'bad': [...]}.
    waiting = pas assez de donnees ; bad = le candidat est mauvais."""
    cfg = {**DEFAULTS, **(cfg or {})}
    waiting, bad = [], []
    if stage == "shadow":
        if cand["n"] < cfg["shadow_min"]:
            waiting.append(f"{cand['n']} shadow predictions < {cfg['shadow_min']}")
        if len(cand_abs) < cfg["shadow_min_labelled"]:
            waiting.append(f"{len(cand_abs)} labelled < {cfg['shadow_min_labelled']}")
    elif stage in ("10%", "50%"):
        if minutes_in_stage < cfg["stage_min_minutes"]:
            waiting.append(f"{minutes_in_stage} min < {cfg['stage_min_minutes']}")
        if cand["n"] < cfg["stage_min_requests"]:
            waiting.append(f"{cand['n']} requests < {cfg['stage_min_requests']}")
        limit = max(cfg["latency_ratio"] * champ["p95_ms"],
                    champ["p95_ms"] + cfg["latency_floor_ms"])
        if cand["p95_ms"] > limit:
            bad.append(f"p95 {cand['p95_ms']:.0f} ms > {limit:.0f} ms")
        if cand["error_rate"] > champ["error_rate"] + cfg["error_delta"]:
            bad.append(f"error rate {cand['error_rate']:.3f} above champion")
    enough = min(len(cand_abs), len(champ_abs)) >= cfg["shadow_min_labelled"]
    if enough and not errors_not_worse(cand_abs, champ_abs, cfg["alpha"]):
        bad.append("absolute errors significantly worse than champion")
    return {"ok": not waiting and not bad, "waiting": waiting, "bad": bad}


def next_stage(stage):
    return STAGES[min(STAGES.index(stage) + 1, len(STAGES) - 1)]


def rule_decision(stage, gate, breach, holds):
    """Decision par regles : sert de repli quand le LLM echoue, et de systeme 'traditionnel'."""
    if breach:
        return {"action": "rollback_emergency", "level": 1,
                "params": {"reason": f"{breach['metric']} {breach['value']:.3f} > {breach['limit']}"}}
    if holds:
        return {"action": "hold", "level": 1,
                "params": {"reason": "hold requested: " + "; ".join(holds)}}
    if stage in ("none", "100%"):
        return {"action": "hold", "level": 1, "params": {"reason": "no rollout in progress"}}
    if gate["bad"]:
        return {"action": "hold", "level": 1, "params": {"reason": "; ".join(gate["bad"])}}
    if gate["waiting"]:
        return {"action": "hold", "level": 1, "params": {"reason": "; ".join(gate["waiting"])}}
    if stage == "50%":
        return {"action": "promote_100", "level": 3,
                "params": {"reason": "all gates passed at 50%"}}
    return {"action": "advance_stage", "level": 2,
            "params": {"to_stage": next_stage(stage), "reason": "gates passed"}}