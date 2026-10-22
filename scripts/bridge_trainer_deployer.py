"""Pont: lit un request.deploy_candidate du board Trainer et le passe au vrai Deployer."""
import os
import sys
import time

sys.path.insert(0, os.getcwd())

from shared.messages import Message, MessageType
from shared.standins import SqliteBoard
from agents.deployer.agent import DeployerAgent
from agents.deployer import state

DB = os.environ.get("BOARD_DB", "runs/real/board.db")

def as_message(d: dict) -> Message:
    return Message(
        id=d.get("id", "msg-bridge"),
        type=MessageType(d["type"]),
        kind=d["kind"],
        sender=d.get("sender", "trainer"),
        to=d.get("to") or ["deployer"],
        topic=d.get("topic", "model:wine-quality"),
        level=d.get("level", 1),
        summary=d.get("summary", ""),
        payload=d.get("payload") or {},
        evidence=d.get("evidence") or {},
    )

def main():
    board = SqliteBoard(DB)
    agent = DeployerAgent()
    print(f"Listening on {DB} for request.deploy_candidate ...")
    seen = set()
    while True:
        for raw in board.read("deployer"):
            mid = raw.get("id")
            if mid in seen:
                continue
            seen.add(mid)
            print(f"<- {raw.get('kind')}: {raw.get('summary')}")
            if raw.get("kind") == "request.deploy_candidate":
                msg = as_message(raw)
                out = agent.step([msg])
                for m in out:
                    print(f"   deployer -> {m.kind}: {m.summary}")
                    # répondre sur le board (comme un vrai agent)
                    try:
                        board.answer(raw, sender="deployer", level=1,
                                     summary=m.summary,
                                     payload=getattr(m, "payload", {}) or {})
                    except Exception as e:
                        print("   board.answer:", e)
                st = state.get_release_state()
                print(f"   release_state: stage={st.get('stage')} candidate={st.get('candidate_version')}")
        time.sleep(1)

if __name__ == "__main__":
    main()