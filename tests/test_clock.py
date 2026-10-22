"""Unit tests for shared/clock.py."""
import time
from datetime import datetime, timedelta, timezone

from shared import clock

START = datetime(2026, 3, 14, 6, 0, tzinfo=timezone.utc)


def test_frozen_clock_moves_only_when_told():
    clock.configure(mode="frozen", start=START)
    assert clock.now() == START
    clock.advance(timedelta(hours=1))
    assert clock.now() == START + timedelta(hours=1)
    clock.set_time("2026-03-15T00:00:00Z")
    assert clock.now() == datetime(2026, 3, 15, tzinfo=timezone.utc)
    assert clock.iso() == "2026-03-15T00:00:00Z"


def test_replay_clock_runs_720_times_faster():
    clock.configure(mode="replay", speed=720, start=START, epoch=time.time() - 1.0)
    elapsed = clock.now() - START
    assert timedelta(minutes=11) < elapsed < timedelta(minutes=14)   # ~12 simulated minutes


def test_replay_clock_is_the_same_in_another_process():
    clock.configure(mode="replay", speed=720, start=START, epoch=time.time() - 5.0)
    env = clock.env_for_agents()
    here = clock.now()
    clock.configure(mode=env["CLOCK_MODE"], speed=float(env["CLOCK_SPEED"]),
                    start=env["CLOCK_START"], epoch=float(env["CLOCK_EPOCH"]))
    assert abs(clock.now() - here) < timedelta(minutes=1)


def test_real_clock_is_utc():
    clock.configure(mode="real")
    assert clock.now().tzinfo is not None
    assert abs(clock.now() - datetime.now(timezone.utc)) < timedelta(seconds=5)
