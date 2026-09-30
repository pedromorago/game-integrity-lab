"""Runs every check against the real pipeline and against every seeded bug.

The real pipeline must pass everything; every seeded bug must fail at least
one check. Prints a Markdown table of which checks caught which bug, next to
the numbers each broken pipeline would have reported for gradient boosting on
the player task, so the effect of each bug on a report is visible too.
"""

from __future__ import annotations

import sys

from validation.checks import CHECKS, base_run
from validation.mutants import MUTANTS
from gil.pipeline import REAL


def failures(pipeline):
    caught = []
    for check in CHECKS:
        try:
            check.run(pipeline)
        except AssertionError:
            caught.append(check)
        except Exception as e:  # a crash counts as caught, but say so
            caught.append(check)
            print(f"  ({pipeline.name}: {check.name} crashed with {type(e).__name__}: {e})", file=sys.stderr)
    return caught


def reported(pipeline) -> str:
    """Three cells: GB PR-AUC, IF PR-AUC, and GB's precision/recall and headline."""
    try:
        runs = base_run(pipeline).player.runs
    except Exception as e:  # noqa: BLE001
        return f"(crashed: {type(e).__name__}) | | "
    gb, iforest = runs["gradient_boosting"].report, runs["isolation_forest"].report
    said = f"precision {gb['precision']:.2f}, recall {gb['recall']:.2f}"
    extra = {k: v for k, v in gb["headline"].items() if k not in ("precision", "recall")}
    if extra:
        said = ", ".join(f"{k} {v:.2f}" for k, v in extra.items()) + " as the headline"
    return f"{gb['pr_auc']:.3f} | {iforest['pr_auc']:.3f} | {said}"


def main() -> int:
    ok = True
    real = failures(REAL)
    if real:
        ok = False
        print("The real pipeline fails: " + ", ".join(c.name for c in real))
    print(f"{len(CHECKS)} checks; the real pipeline passes {len(CHECKS) - len(real)} of them.")
    print()
    print("| Seeded bug | Stage | GB PR-AUC | IF PR-AUC | GB reported | Caught by |")
    print("|---|---|---|---|---|---|")
    print(f"| none (real pipeline) | | {reported(REAL)} | {'nothing (passes every check)' if not real else 'FAILS'} |")
    for pipeline, stage in MUTANTS:
        caught = failures(pipeline)
        if not caught:
            ok = False
        names = ", ".join(f"`{c.name}` ({c.kind})" for c in caught) or "**nothing**"
        print(f"| {pipeline.name} | {stage} | {reported(pipeline)} | {names} |")
    print()
    print("GB and IF: gradient boosting and Isolation Forest on the player task, as each pipeline would report "
          "them on the check data (a quarter-size synthetic population, seed 11). \"GB reported\" is what the "
          "report's precision, recall and headline fields would say.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
