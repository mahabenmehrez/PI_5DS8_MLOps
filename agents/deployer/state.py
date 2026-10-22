"""Tables du Deployer dans board.db : release_state, predictions, labels."""
import os
import sqlite3
from datetime import datetime, timezone

DEFAULT_STATE = {"champion_version": "1", "candidate_version": None,
                 "stage": "none", "candidate_share": 0.0, "shadow_on": 0}

SCHEMAS = [
    """CREATE TABLE IF NOT EXISTS release_state (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        champion_version TEXT, candidate_version TEXT, stage TEXT,
        candidate_share REAL, shadow_on INTEGER, updated_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS predictions (
        request_id TEXT PRIMARY KEY, wine_id INTEGER, time TEXT,
        model_version TEXT, variant TEXT, features TEXT, prediction REAL,
        latency_ms REAL, status_code INTEGER)""",
    """CREATE TABLE IF NOT EXISTS labels (
        wine_id INTEGER PRIMARY KEY, quality REAL, available_at TEXT)""",
]


def _now():
    try:
        from shared import clock
        return clock.now().isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def connect():
    con = sqlite3.connect(os.getenv("BOARD_DB", "board.db"), timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    for schema in SCHEMAS:
        con.execute(schema)
    con.row_factory = sqlite3.Row
    return con


def get_release_state():
    con = connect()
    row = con.execute("SELECT * FROM release_state WHERE id = 1").fetchone()
    con.close()
    state = dict(DEFAULT_STATE)
    state["updated_at"] = None
    if row:
        for key in list(DEFAULT_STATE) + ["updated_at"]:
            state[key] = row[key]
    return state


def set_release_state(**fields):
    """Met a jour release_state. Refuse d'ecrire pendant un dry run."""
    if os.getenv("DRY_RUN") == "1":
        return None
    state = get_release_state()
    state.update(fields)
    state["updated_at"] = _now()
    con = connect()
    con.execute(
        "INSERT OR REPLACE INTO release_state (id, champion_version, candidate_version,"
        " stage, candidate_share, shadow_on, updated_at) VALUES (1,?,?,?,?,?,?)",
        (state["champion_version"], state["candidate_version"], state["stage"],
         state["candidate_share"], state["shadow_on"], state["updated_at"]))
    con.commit()
    con.close()
    return state


def log_predictions(rows):
    """rows : liste de dicts avec les colonnes de predictions."""
    if os.getenv("DRY_RUN") == "1" or not rows:
        return 0
    con = connect()
    con.executemany(
        "INSERT OR IGNORE INTO predictions VALUES (:request_id, :wine_id, :time,"
        " :model_version, :variant, :features, :prediction, :latency_ms, :status_code)",
        rows)
    con.commit()
    con.close()
    return len(rows)


def read_predictions(variant=None, since=None):
    q, args = "SELECT * FROM predictions WHERE 1=1", []
    if variant:
        q += " AND variant = ?"
        args.append(variant)
    if since:
        q += " AND time >= ?"
        args.append(since)
    con = connect()
    rows = [dict(r) for r in con.execute(q + " ORDER BY time", args)]
    con.close()
    return rows


def abs_errors(variant, since=None):
    """Erreurs absolues |prediction - vraie qualite| pour les vins deja etiquetes."""
    q = ("SELECT ABS(p.prediction - l.quality) AS e FROM predictions p"
         " JOIN labels l ON p.wine_id = l.wine_id WHERE p.variant = ? AND p.prediction IS NOT NULL")
    args = [variant]
    if since:
        q += " AND p.time >= ?"
        args.append(since)
    con = connect()
    errs = [r["e"] for r in con.execute(q, args)]
    con.close()
    return errs