"""Command line: `gil generate`, `gil evaluate`, `gil gate`.

Exit codes: 0 success (gate passed), 1 gate failed, 2 bad input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import report as report_mod
from .config import SCENARIOS, default_config, small_config
from .data import load, save
from .evaluate import EvalConfig, evaluate
from .features import FEATURE_GROUPS
from .gate import Tolerances, compare, to_dict, to_markdown
from .simulate import generate

CONFIGS = {"default": default_config, "small": small_config}


def _generate(args: argparse.Namespace) -> int:
    cfg = CONFIGS[args.config]()
    if args.scenario:
        cfg = SCENARIOS[args.scenario](cfg)
    description = f"synthetic, {args.config} config" + (f", scenario {args.scenario}" if args.scenario else "")
    ds = generate(cfg, seed=args.seed, description=description)
    out = Path(args.out)
    save(ds, out)
    meta = {"synthetic": True, "config": args.config, "scenario": args.scenario, "seed": args.seed,
            "fingerprint": ds.fingerprint()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out}/data.npz and players.csv: {len(ds.players.id)} synthetic players, "
          f"{len(ds.decisions.player)} decisions, fingerprint {ds.fingerprint()}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    data = Path(args.data)
    if not (data / "data.npz").exists():
        print(f"error: {data}/data.npz not found; run `gil generate --out {data}` first", file=sys.stderr)
        return 2
    meta_path = data / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"config": "default"}
    ds = load(data)
    cfg = EvalConfig(seed=args.seed, budget=args.budget, exclude_groups=tuple(args.exclude),
                     scenarios=() if args.no_scenarios else tuple(SCENARIOS))
    ev = evaluate(ds, cfg, sim_cfg=CONFIGS[meta.get("config", "default")](), with_pairs=not args.no_pairs)
    if args.no_scenarios:
        ev.report["robustness"] = {}
    js, md = report_mod.write(ev.report, Path(args.out))
    print(f"wrote {js} and {md}")
    for name, r in ev.report["player_task"]["models"].items():
        print(f"  player task, {name}: PR-AUC {r['pr_auc']:.3f}, precision {r['precision']:.3f}, "
              f"recall {r['recall']:.3f}, test FPR {r['fpr']:.2%}")
    return 0


def _gate(args: argparse.Namespace) -> int:
    try:
        baseline = json.loads(Path(args.baseline).read_text())
        candidate = json.loads(Path(args.candidate).read_text())
        tol = Tolerances.load(Path(args.config) if args.config else None)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    result = compare(baseline, candidate, tol)
    text = to_markdown(result, show_all=args.all)
    if args.markdown:
        Path(args.markdown).write_text(text)
    if args.json:
        Path(args.json).write_text(json.dumps(to_dict(result), indent=2) + "\n")
    if not args.quiet or not result.passed:
        print(text)
    else:
        print(f"Regression gate: PASS ({len(result.findings)} comparisons)")
    if result.errors:
        return 2
    return 0 if result.passed else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gil", description="Evaluation testbed for models that flag players. "
                                "All data is synthetic.")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate a synthetic population")
    g.add_argument("--out", default="data", help="output directory (default: data)")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--config", choices=sorted(CONFIGS), default="default")
    g.add_argument("--scenario", choices=sorted(SCENARIOS), default=None,
                   help="apply a robustness scenario to the generator")
    g.set_defaults(fn=_generate)

    e = sub.add_parser("evaluate", help="train, pick thresholds on validation, report on test")
    e.add_argument("--data", default="data", help="directory written by `gil generate`")
    e.add_argument("--out", default="out", help="where report.json and report.md go (default: out)")
    e.add_argument("--seed", type=int, default=7, help="seed for the split and the models")
    e.add_argument("--budget", type=float, default=0.01,
                   help="largest share of legitimate validation units that may be flagged (default: 0.01)")
    e.add_argument("--exclude", nargs="*", default=[], choices=sorted(FEATURE_GROUPS),
                   help="feature groups to leave out (to build a candidate model)")
    e.add_argument("--no-scenarios", action="store_true", help="skip the robustness scenarios")
    e.add_argument("--no-pairs", action="store_true", help="skip the pair (collusion) task")
    e.set_defaults(fn=_evaluate)

    c = sub.add_parser("gate", help="compare a candidate report with a baseline report")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--config", default=None, help="tolerances JSON (default: built-in, same as configs/gate.json)")
    c.add_argument("--markdown", default=None, help="also write the result as Markdown")
    c.add_argument("--json", default=None, help="also write the result as JSON")
    c.add_argument("--all", action="store_true", help="list every comparison, not only failures")
    c.add_argument("--quiet", action="store_true", help="print one line when the gate passes")
    c.set_defaults(fn=_gate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
