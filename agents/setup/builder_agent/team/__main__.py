"""Team mode from the command line, run from the team repo's root.

    python -m agents.setup.builder_agent.team            # show the plan and what would be created (writes nothing)
    python -m agents.setup.builder_agent.team --write    # create the missing files (never overwrites)
"""
import argparse
import json

from . import TeamLayout, render_team, team_plan, write_team


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m agents.setup.builder_agent.team")
    ap.add_argument("--write", action="store_true", help="create the missing files (default: only show the plan)")
    ap.add_argument("--root", default=".", help="the team repo (default: here)")
    args = ap.parse_args()
    plan = team_plan(TeamLayout.from_contracts(root=args.root))
    files = render_team(plan)
    print(f"code: {plan.layout.code_dir}/   data: {plan.layout.data_path}   python {plan.python_version}")
    print(f"train: {plan.train_script}   serve: {plan.serve_target}   target: {plan.target}   "
          f"{len(plan.columns)} feature columns")
    for f in plan.findings:
        print(f"  finding [{f.kind}] {f.path}: {f.message}")
    if not args.write:
        from pathlib import Path
        for path in files:
            print(f"  {'exists  ' if (Path(args.root) / path).exists() else 'would create'} {path}")
        return 0
    report = write_team(plan, files, args.root)
    print(json.dumps({"created": report.created, "verified": report.verified}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
