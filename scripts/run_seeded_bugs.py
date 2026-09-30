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
    try:
        rep = base_run(pipeline).player.runs["gradient_boosting"].report
    except Exception as e:  # noqa: BLE001
        return f"(crashed: {type(e).__name__})"
    head = ", ".join(f"{k} {v:.2f}" for k, v in rep["headline"].items())
    return f"{rep['pr_auc']:.3f} | {head}"


def main() -> int:
    ok = True
    real = failures(REAL)
    if real:
        ok = False
        print("The real pipeline fails: " + ", ".join(c.name for c in real))
    print(f"{len(CHECKS)} checks; the real pipeline passes {len(CHECKS) - len(real)} of them.")
    print()
    print("| Seeded bug | Stage | Reported PR-AUC | Reported headline | Caught by |")
    print("|---|---|---|---|---|")
    print(f"| none (real pipeline) | | {reported(REAL)} | {'**nothing should fail**' if not real else 'FAILS'} |")
    for pipeline, stage in MUTANTS:
        caught = failures(pipeline)
        if not caught:
            ok = False
        names = ", ".join(f"`{c.name}` ({c.kind})" for c in caught) or "**nothing**"
        print(f"| {pipeline.name} | {stage} | {reported(pipeline)} | {names} |")
    print()
    print("Reported PR-AUC and headline: gradient boosting on the player task, as each pipeline would report it "
          "on the check data (half-size population, seed 11).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
