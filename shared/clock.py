"""
Shared clock (owner P3). Every agent takes the time from clock.now(), never
from datetime.now(), so a replayed day behaves the same for everyone.

Three modes:
- real:   real UTC time.
- replay: simulated time = start + (wall time - epoch) * speed. It only depends
          on the wall clock and on CLOCK_START / CLOCK_EPOCH / CLOCK_SPEED, so
          every agent process computes the same simulated time.
- frozen: time moves only through advance() / set_time(). For unit tests.

Configuration: configure(...) in code, or the environment variables
CLOCK_MODE, CLOCK_SPEED (default 720: 1 real second = 12 simulated minutes),
CLOCK_START (ISO-8601, simulated start) and CLOCK_EPOCH (unix seconds, real
moment the replay started). The harness sets them before starting the agents.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

DEFAULT_SPEED = 720.0
MODES = ("real", "replay", "frozen")

_state: dict = {}


def _parse(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def configure(mode: str | None = None, speed: float | None = None,
              start: datetime | str | None = None, epoch: float | None = None) -> None:
    """Sets the clock. Arguments left out are read from the environment."""
    mode = mode or os.environ.get("CLOCK_MODE", "real")
    if mode not in MODES:
        raise ValueError(f"clock mode must be one of {MODES}, got {mode!r}")
    start = start or os.environ.get("CLOCK_START")
    if isinstance(start, str):
        start = _parse(start)
    if start is None:
        start = datetime.now(timezone.utc)
    if epoch is None:
        epoch = float(os.environ.get("CLOCK_EPOCH", time.time()))
    speed = float(speed if speed is not None else os.environ.get("CLOCK_SPEED", DEFAULT_SPEED))
    _state.clear()
    _state.update(mode=mode, speed=speed, start=start, epoch=epoch, offset=timedelta(0))


def now() -> datetime:
    """Current time, always timezone-aware UTC."""
    if not _state:
        configure()
    mode = _state["mode"]
    if mode == "real":
        return datetime.now(timezone.utc) + _state["offset"]
    if mode == "frozen":
        return _state["start"] + _state["offset"]
    elapsed = (time.time() - _state["epoch"]) * _state["speed"]
    return _state["start"] + timedelta(seconds=elapsed) + _state["offset"]


def iso() -> str:
    """now() as an ISO-8601 string ending in Z, the format used in messages."""
    return now().isoformat().replace("+00:00", "Z")


def advance(delta: timedelta | float) -> datetime:
    """Moves the clock forward (seconds or timedelta). Used by the harness and tests."""
    if not _state:
        configure()
    if not isinstance(delta, timedelta):
        delta = timedelta(seconds=delta)
    _state["offset"] += delta
    return now()


def set_time(value: datetime | str) -> datetime:
    """Jumps to an exact time, keeping the current mode."""
    target = _parse(value) if isinstance(value, str) else value
    _state["offset"] += target - now()
    return now()


def mode() -> str:
    if not _state:
        configure()
    return _state["mode"]


def env_for_agents() -> dict:
    """Environment variables that make another process share this replay clock."""
    if not _state:
        configure()
    start = _state["start"] + _state["offset"]
    return {"CLOCK_MODE": _state["mode"], "CLOCK_SPEED": str(_state["speed"]),
            "CLOCK_START": start.isoformat(), "CLOCK_EPOCH": str(_state["epoch"])}
