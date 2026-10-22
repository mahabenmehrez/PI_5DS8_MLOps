import os

import pytest

from agents.deployer import state
from deployer_live import LiveHarness
from shared import memory


@pytest.fixture(autouse=True)
def restore_globals():
    old_now, old_db, old_mem = state._now, os.environ.get("BOARD_DB"), memory.DB_PATH
    yield
    state._now, memory.DB_PATH = old_now, old_mem
    if old_db is None:
        os.environ.pop("BOARD_DB", None)
    else:
        os.environ["BOARD_DB"] = old_db


def run(scenario, minutes=150):
    h = LiveHarness(scenario, 42)
    h.send_candidate()
    for _ in range(minutes):
        if h.agent.pending or h.release_state()["stage"] == "none":
            break
        h.tick()
    return h


def test_healthy_reaches_human_approval():
    h = run("Candidat sain")
    assert h.agent.pending is not None
    assert h.release_state()["stage"] == "50%"


def test_slow_candidate_rolls_back():
    h = run("Candidat lent (scenario G)")
    assert h.agent.history[-1]["to_stage"] == "rolled_back"
    assert h.release_state()["stage"] == "none"


def test_error_burst_rolls_back():
    h = run("Candidat en erreur (10 %)")
    assert h.agent.history[-1]["to_stage"] == "rolled_back"


def test_inaccurate_candidate_stays_in_shadow():
    h = run("Candidat moins precis", minutes=60)
    assert h.release_state()["stage"] == "shadow"


def test_human_reject_rolls_back_and_allows_new_candidate():
    h = run("Candidat sain")
    h.human_decision(False)
    assert h.release_state()["stage"] == "none"
    assert h.agent.history[-1]["to_stage"] == "rolled_back"
    assert not h.agent.holds                    # plus de hold bloquant
    h.send_candidate()
    assert h.release_state()["stage"] == "shadow"


def test_human_approve_promotes():
    h = run("Candidat sain")
    h.human_decision(True)
    assert h.release_state()["champion_version"] == "2"
    assert h.agent.history[-1]["to_stage"] == "100%"


def test_hold_then_release():
    h = LiveHarness("Candidat sain", 42)
    h.send_candidate()
    h.send_hold("unité suspecte")
    for _ in range(40):
        h.tick()
    assert h.release_state()["stage"] == "shadow"
    h.release_hold()
    for _ in range(30):
        h.tick()
    assert h.release_state()["stage"] != "shadow"


def test_memory_is_written():
    h = run("Candidat sain")
    assert not h.memory_rows().empty
