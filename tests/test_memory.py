import os
os.environ["BOARD_DB"] = "test_board_memory.db"
from shared import memory


def test_roundtrip():
    prob = {"error_class": "unit_change", "feature": "total_sulfur_dioxide",
            "text": "median 1/1000 of reference"}
    memory.remember("checker", "inc-1", prob, {"psi": 0.9}, "quarantine",
                    {"batch_id": "batch-0042"}, 1, "success", 120)
    memory.remember("checker", "inc-2", prob, {}, "repair", {}, 2, "failure")
    assert len(memory.similar(prob, "checker")) >= 2
    best = memory.best_fixes(prob, "checker")
    assert best[0][0] == "quarantine" and best[0][2] == 1.0