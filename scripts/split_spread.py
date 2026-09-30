"""How much do the headline numbers move when only the split changes?

Generates the default synthetic population (seed 42) once, then evaluates it
with several evaluation seeds. Each seed gives a different assignment of
stake pools to train, validation and test, and different model seeds. Prints
a Markdown table per task. Usage: python scripts/split_spread.py [n_seeds]
"""

from __future__ import annotations

import sys

from gil.config import default_config
from gil.data import LEGITIMATE
from gil.evaluate import EvalConfig, evaluate
from gil.simulate import generate


def main(n_seeds: int = 7) -> int:
    ds = generate(default_config(), seed=42)
    budget = EvalConfig().budget
    rows: dict[tuple[str, str], list[dict]] = {}
    for seed in range(1, n_seeds + 1):
        rep = evaluate(ds, EvalConfig(seed=seed, scenarios=())).report
        for task in ("player_task", "pair_task"):
            for model, r in rep[task]["models"].items():
                rows.setdefault((task, model), []).append(r)

    print(f"Default synthetic population (seed 42), evaluation seeds 1 to {n_seeds}. "
          f"Budget: at most {budget:.0%} of legitimate validation units flagged.")
    print()
    print("| Task | Model | PR-AUC min to max | Recall min to max | Test FPR min to max | Splits with test FPR over the budget |")
    print("|---|---|---|---|---|---|")
    for (task, model), rs in rows.items():
        ap = [r["pr_auc"] for r in rs]
        rec = [r["recall"] for r in rs]
        fpr = [r["fpr"] for r in rs]
        over = sum(f > budget for f in fpr)
        print(f"| {task.replace('_task', '')} | {model} | {min(ap):.3f} to {max(ap):.3f} | "
              f"{min(rec):.3f} to {max(rec):.3f} | {min(fpr):.2%} to {max(fpr):.2%} | {over} of {len(rs)} |")
    print()
    for task, unit, legit in (("player_task", "players", LEGITIMATE), ("pair_task", "pairs", ("friends", "other"))):
        print(f"Legitimate {unit} flagged on test, summed over all splits ({task.replace('_', ' ')}):")
        print()
        print("| Model | " + " | ".join(legit) + " |")
        print("|---|" + "---|" * len(legit))
        for (t, model), rs in rows.items():
            if t != task:
                continue
            cells = []
            for seg in legit:
                k = sum(r["segments"][seg]["flagged"] for r in rs)
                n = sum(r["segments"][seg]["n"] for r in rs)
                cells.append(f"{k}/{n} ({k / n:.1%})")
            print(f"| {model} | " + " | ".join(cells) + " |")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 7))
