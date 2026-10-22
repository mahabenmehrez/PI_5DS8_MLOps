"""Run the Setup Agent: python -m agents.setup (also the docker-compose.agents.yml service).

    python -m agents.setup                         # first run: generate, dry-run, fix, commit on builder/<date>
    python -m agents.setup --rebuild "reason" --changed baseline/train.py   # what request.rebuild triggers
    python -m agents.setup --no-commit             # leave the files uncommitted on the builder branch
    python -m agents.setup --plan                  # show what it would create, change nothing

The result message (pipeline.setup_completed / setup_failed) is printed as JSON and appended to
agents/setup/logs/outbox.jsonl until shared/board.py exists; then it goes on the board.
"""
import argparse
import json
import sys


from agents.setup.agent import HERE, TOPIC, SetupAgent
from shared.messages import Message, MessageType

OUTBOX = HERE / "logs" / "outbox.jsonl"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agents.setup")
    ap.add_argument("--root", default=".", help="the team repo (default: here)")
    ap.add_argument("--no-commit", action="store_true", help="write the files on the builder branch, don't commit")
    ap.add_argument("--rebuild", metavar="REASON", help="act as on a request.rebuild with this reason")
    ap.add_argument("--changed", nargs="*", default=[], help="changed paths for --rebuild")
    ap.add_argument("--plan", action="store_true", help="only show the plan (team mode), change nothing")
    args = ap.parse_args(argv)

    if args.plan:
        from agents.setup.builder_agent.team.__main__ import main as team_main
        sys.argv = ["team", "--root", args.root]
        return team_main()

    agent = SetupAgent(root=args.root, commit=not args.no_commit, first_run=args.rebuild is None)
    if args.rebuild is not None:
        agent.handle(Message(type=MessageType.REQUEST, kind="request.rebuild", sender="human", to=["setup"],
                             topic=TOPIC, level=1, summary=f"rebuild: {args.rebuild}"[:160],
                             payload={"reason": args.rebuild, "changed_paths": args.changed}))
    msg = agent.step()
    if msg is None:
        print("nothing to do")
        return 0
    OUTBOX.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTBOX, "a", encoding="utf-8") as f:
        for m in agent.outbox:
            f.write(m.model_dump_json() + "\n")
    print(json.dumps(json.loads(msg.model_dump_json()), indent=2))
    return 0 if msg.kind == "pipeline.setup_completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
