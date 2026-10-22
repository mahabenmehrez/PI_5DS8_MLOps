import pytest

from deployer_live import LiveHarness
from evaluation import experiment


def test_agent_limits_exposure_to_a_slow_candidate():
    r = experiment.run_agent("Candidat lent (scenario G)", 42)
    t = experiment.run_traditional("Candidat lent (scenario G)", 42)
    assert r["outcome"] == "rollback" and not r["promoted_bad"]
    assert r["bad_served"] < 200 and t["bad_served"] == 180 * 60
    assert r["rollback_verified"] is True              # incident.closed apres 2 minutes


def test_agent_blocks_an_inaccurate_candidate_before_any_user_sees_it():
    r = experiment.run_agent("Candidat moins precis", 42)
    assert r["outcome"] == "bloque" and r["bad_served"] == 0


def test_agent_promotes_a_healthy_candidate_only_with_a_human_card():
    r = experiment.run_agent("Candidat sain", 42)
    assert r["outcome"] == "promu" and r["human_cards"] == 1 and not r["promoted_bad"]
    assert r["minutes_to_promote"] > 30                # le cout de la prudence


def test_traditional_pipeline_promotes_a_bad_candidate_immediately():
    t = experiment.run_traditional("Candidat en erreur (10 %)", 42)
    assert t["promoted_bad"] and t["minutes_to_promote"] == 0 and t["human_cards"] == 0


def test_summary_has_one_row_per_scenario_and_system():
    runs = [experiment.run_traditional(s, 1) for s in experiment.SCENARIOS]
    runs += [experiment.run_agent("Candidat sain", 1)]
    rows = experiment.summarize(runs[:4] + runs[4:] + [experiment.run_agent(s, 1)
                                                      for s in list(experiment.SCENARIOS)[1:]])
    assert len(rows) == 8 and {r["systeme"] for r in rows} == {"traditionnel", "agent"}


def test_harness_posts_incident_closed_after_a_rollback():
    h = LiveHarness("Candidat lent (scenario G)", 42)
    h.send_candidate()
    for _ in range(120):
        if h.release_state()["stage"] == "none":
            break
        h.tick()
    assert h.agent.watching()
    h.settle()
    closed = [m for m in h.agent.outbox if m.kind == "incident.closed"]
    assert closed and closed[-1].payload["outcome"] == "recovered"
    assert not h.agent.watching()
