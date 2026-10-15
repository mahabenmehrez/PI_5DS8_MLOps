"""Experience log : remember / similar / best_fixes (SQLite, repli en memoire)."""
import json
import os
import re
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.getenv("BOARD_DB", "board.db")
_fallback = []  # repli en memoire si SQLite est indisponible

SCHEMA = """CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT, agent TEXT, incident_id TEXT,
    error_class TEXT, feature TEXT, problem_text TEXT, evidence TEXT,
    action TEXT, params TEXT, level INTEGER,
    outcome TEXT, time_to_recover_s REAL)"""


def _now():
    try:
        from shared import clock
        return clock.now().isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def _connect():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(SCHEMA)
    con.row_factory = sqlite3.Row
    return con


def _norm(problem):
    """Accepte un dict {error_class, feature, text} ou une simple chaine."""
    if isinstance(problem, str):
        problem = {"text": problem}
    return (problem.get("error_class") or "", problem.get("feature") or "",
            problem.get("text") or "")


def _words(text):
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def remember(agent, incident_id, problem, evidence, action, params, level,
             outcome, time_to_recover_s=None):
    """Enregistre un cas. outcome : 'success' ou 'failure'. Retourne mem_id."""
    if os.getenv("DRY_RUN") == "1":
        return None  # un dry run n'ecrit pas dans les vraies tables
    ec, feat, txt = _norm(problem)
    row = (_now(), agent, incident_id, ec, feat, txt,
           json.dumps(evidence or {}), action, json.dumps(params or {}),
           level, outcome, time_to_recover_s)
    try:
        con = _connect()
        cur = con.execute(
            "INSERT INTO memory (time, agent, incident_id, error_class, feature,"
            " problem_text, evidence, action, params, level, outcome,"
            " time_to_recover_s) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", row)
        con.commit()
        mem_id = f"mem-{cur.lastrowid:04d}"
        con.close()
        return mem_id
    except sqlite3.Error:
        _fallback.append(row)
        return f"mem-fb{len(_fallback):04d}"


def _all_rows(agent=None):
    rows = []
    try:
        con = _connect()
        q = "SELECT * FROM memory" + (" WHERE agent=?" if agent else "")
        rows = [dict(r) for r in con.execute(q, (agent,) if agent else ())]
        con.close()
    except sqlite3.Error:
        pass
    keys = ["time", "agent", "incident_id", "error_class", "feature",
            "problem_text", "evidence", "action", "params", "level",
            "outcome", "time_to_recover_s"]
    for r in _fallback:
        d = dict(zip(keys, r))
        if agent is None or d["agent"] == agent:
            rows.append(d)
    return rows


def similar(problem, agent=None, k=5):
    """Cas proches : meme classe d'erreur, meme feature, mots en commun."""
    ec, feat, txt = _norm(problem)
    q_words = _words(txt)
    scored = []
    for r in _all_rows(agent):
        score = 0.0
        if ec and r["error_class"] == ec:
            score += 3
        if feat and r["feature"] == feat:
            score += 2
        r_words = _words(r["problem_text"] or "")
        if q_words and r_words:
            score += len(q_words & r_words) / len(q_words | r_words)
        if score > 0:
            scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:k]]


def best_fixes(problem, agent=None):
    """[(action, params, taux_de_succes, n)] trie par taux puis par n."""
    stats = {}
    for r in similar(problem, agent, k=50):
        key = (r["action"], r["params"])
        ok, n = stats.get(key, (0, 0))
        stats[key] = (ok + (r["outcome"] == "success"), n + 1)
    out = [(a, json.loads(p or "{}"), ok / n, n) for (a, p), (ok, n) in stats.items()]
    out.sort(key=lambda t: (t[2], t[3]), reverse=True)
    return out