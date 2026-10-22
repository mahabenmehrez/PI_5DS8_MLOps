"""95% intervals for the team_eval tables: cluster bootstrap over the faults (a fault and its 2-3 repos move
together, since one repo's result says much about the others), 10,000 resamples, seed 0.

    python -m agents.setup.eval.team_intervals team_eval_v1 team_eval_v2 team_eval_v3 team_eval_final

Writes eval/results/team_intervals.md. Paired differences use the same resamples for both configs.
"""
import json
import random
import sys
from pathlib import Path

from agents.setup.eval.team_eval import SETS, invalid_cases

HERE = Path(__file__).resolve().parent
N = 10_000


def interval(values: list[float]) -> str:
    values = sorted(values)
    return f"{100 * values[int(0.025 * N)]:.0f}–{100 * values[int(0.975 * N)]:.0f}%"


def analyse(name: str) -> list[str]:
    rows = [json.loads(line) for line in (HERE / "results" / f"{name}.jsonl").read_text().splitlines()]
    out = [f"## {name}", ""]
    for set_name, cases in SETS.items():
        bad = invalid_cases(rows, cases) if set_name in ("final", "cascade") else set()
        rs = [r for r in rows if (r["repo"], r["fault"]) in cases and (r["repo"], r["fault"]) not in bad]
        configs = [c for c in dict.fromkeys(r["config"] for r in rs)]
        if not rs or len({(r["repo"], r["fault"]) for r in rs}) < 2:
            continue
        by = {c: {(r["repo"], r["fault"]): r["success"] for r in rs if r["config"] == c} for c in configs}
        keys = sorted(set.intersection(*(set(v) for v in by.values())))
        faults = sorted({f for _, f in keys})
        if len(faults) < 3:                     # e.g. the slots set: one fault ("clean") on two repos
            out += [f"### {set_name} ({len(keys)} cases)", "", "Too few faults for an interval: "
                    + ", ".join(f"{c} {sum(by[c][k] for k in keys)}/{len(keys)}" for c in configs), ""]
            continue
        rng = random.Random(0)
        draws = []
        for _ in range(N):
            pick = [rng.choice(faults) for _ in faults]
            draws.append([k for f in pick for k in keys if k[1] == f])
        rate = lambda c, ks: sum(by[c][k] for k in ks) / len(ks)  # noqa: E731
        out += [f"### {set_name} ({len(keys)} cases, {len(faults)} faults)", "",
                "| Config | Success | 95% interval |", "|---|---|---|"]
        for c in configs:
            out.append(f"| {c} | {sum(by[c][k] for k in keys)}/{len(keys)} ({100 * rate(c, keys):.0f}%) "
                       f"| {interval([rate(c, d) for d in draws])} |")
        if "rules" in configs:
            out += ["", "| Difference vs rules | Point | 95% interval | Excludes 0 |", "|---|---|---|---|"]
            for c in configs:
                if c == "rules":
                    continue
                diffs = sorted(rate(c, d) - rate("rules", d) for d in draws)
                lo, hi = diffs[int(0.025 * N)], diffs[int(0.975 * N)]
                out.append(f"| {c} − rules | {100 * (rate(c, keys) - rate('rules', keys)):+.0f}% "
                           f"| {100 * lo:+.0f}% to {100 * hi:+.0f}% | {'yes' if lo > 0 or hi < 0 else 'no'} |")
        if "llm_sft" in configs and "llm_crag" in configs:
            diffs = sorted(rate("llm_sft", d) - rate("llm_crag", d) for d in draws)
            lo, hi = diffs[int(0.025 * N)], diffs[int(0.975 * N)]
            better = sum(by["llm_sft"][k] and not by["llm_crag"][k] for k in keys)
            worse = sum(by["llm_crag"][k] and not by["llm_sft"][k] for k in keys)
            out += ["", f"Fine-tuned − base (llm_sft − llm_crag): {100 * (rate('llm_sft', keys) - rate('llm_crag', keys)):+.0f}%, "
                        f"95% {100 * lo:+.0f}% to {100 * hi:+.0f}% ({'excludes 0' if lo > 0 or hi < 0 else 'includes 0'}); "
                        f"better on {better} cases, worse on {worse}"]
        if "llm_cascade" in configs and "llm_sft" in configs:
            diffs = sorted(rate("llm_cascade", d) - rate("llm_sft", d) for d in draws)
            lo, hi = diffs[int(0.025 * N)], diffs[int(0.975 * N)]
            better = sum(by["llm_cascade"][k] and not by["llm_sft"][k] for k in keys)
            worse = sum(by["llm_sft"][k] and not by["llm_cascade"][k] for k in keys)
            out += ["", f"Cascade − fine-tuned alone (llm_cascade − llm_sft): "
                        f"{100 * (rate('llm_cascade', keys) - rate('llm_sft', keys)):+.0f}%, 95% {100 * lo:+.0f}% to "
                        f"{100 * hi:+.0f}% ({'excludes 0' if lo > 0 or hi < 0 else 'includes 0'}); better on {better} cases, "
                        f"worse on {worse}"]
        out.append("")
    return out


def main(names: list[str]) -> int:
    out = ["# Team-mode evaluation: 95% intervals (cluster bootstrap over faults)", "",
           "Development sets (v1, heldout after v2, heldout_v3 after v3) were used to change the agent; only `final` "
           "is an unseen test. Few faults per set: wide intervals are the honest picture.", ""]
    for n in names:
        out += analyse(n)
    (HERE / "results" / "team_intervals.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["team_eval_v1", "team_eval_v2", "team_eval_v3", "team_eval_final"]))
