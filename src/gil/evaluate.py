"""The evaluation: train on one set of players, pick the operating threshold on
another, report on a third, then stress the frozen models with scenarios.

Two tasks share the machinery:
- player task: flag bots and RTA users among individual players;
- pair task: flag colluding pairs among pairs of players who often meet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import __version__
from .config import SCENARIOS, SimConfig, default_config
from .data import LEGITIMATE, OUT_OF_PLAYER_TASK, PLAYER_TASK_POSITIVE, Dataset
from .features import FeatureTable, pair_members, pair_segments
from .metrics import wilson_interval
from .models import PAIR_RULES, PLAYER_RULES, Scorer
from .pipeline import REAL, Pipeline
from .simulate import generate
from .splits import SPLITS, Split

PAIR_POSITIVE = ("colluding", "chip_dumping")


@dataclass(frozen=True)
class EvalConfig:
    seed: int = 7
    budget: float = 0.01                     # max share of legitimate units flagged
    sweep: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05)
    exclude_groups: tuple[str, ...] = ()     # feature groups left out (for candidate models)
    min_shared_sessions: int = 5
    scenarios: tuple[str, ...] = tuple(SCENARIOS)


@dataclass
class ModelRun:
    scorer: Scorer
    threshold: float
    scores: dict[str, np.ndarray]            # per split, aligned with TaskRun.ids
    report: dict


@dataclass
class TaskRun:
    name: str
    ids: dict[str, np.ndarray]               # per split
    y: dict[str, np.ndarray]
    segments: dict[str, np.ndarray]
    table: FeatureTable
    runs: dict[str, ModelRun] = field(default_factory=dict)


@dataclass
class Evaluation:
    report: dict
    player: TaskRun
    pair: TaskRun | None
    split: Split


def _segment_rates(segments: np.ndarray, flagged: np.ndarray) -> dict[str, dict]:
    out = {}
    for seg in sorted(set(segments.tolist())):
        m = segments == seg
        k, n = int(flagged[m].sum()), int(m.sum())
        lo, hi = wilson_interval(k, n)
        out[seg] = {"n": n, "flagged": k, "rate": k / n if n else 0.0, "ci95": [lo, hi]}
    return out


def _operating_report(pipeline: Pipeline, y: np.ndarray, scores: np.ndarray, segments: np.ndarray,
                      threshold: float) -> dict:
    flagged = scores >= threshold
    c = pipeline.confusion(y, flagged)
    precision, recall = pipeline.precision_recall(y, flagged)
    legit = c.fp + c.tn
    fpr = c.fp / legit if legit else 0.0
    return {
        "confusion": c.as_dict(),
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "fpr_ci95": list(wilson_interval(c.fp, legit)),
        "flagged_legit_per_10k": 10_000 * fpr,
        "headline": pipeline.headline(y, flagged),
        "segments": _segment_rates(segments, flagged),
    }


def _run_task(pipeline: Pipeline, name: str, table: FeatureTable, ids: dict[str, np.ndarray],
              y: dict[str, np.ndarray], segments: dict[str, np.ndarray], counts: dict[str, np.ndarray],
              rules: dict[str, int], cfg: EvalConfig) -> TaskRun:
    task = TaskRun(name, ids, y, segments, table)
    X = {s: table.rows(ids[s]) for s in SPLITS}
    thr_split, ev = pipeline.threshold_split, pipeline.eval_split
    for scorer in pipeline.models(cfg.seed, rules):
        scorer.fit(X["train"], y["train"], table.names)
        scores = {s: scorer.score(X[s]) for s in SPLITS}
        threshold = pipeline.select_threshold(scores[thr_split], y[thr_split], cfg.budget)
        ye, se, sege = pipeline.eval_units(ids[ev], y[ev], scores[ev], segments[ev], counts[ev])
        rep = {
            "threshold": threshold,
            "threshold_split_fpr": float((scores[thr_split][y[thr_split] == 0] >= threshold).mean()),
            "pr_auc": pipeline.average_precision(ye, se, pipeline.pr_curve),
            "base_rate": float(np.mean(ye)),
            "n_units": int(len(ye)),
            **_operating_report(pipeline, ye, se, sege, threshold),
            "sweep": [],
        }
        for b in cfg.sweep:
            t = pipeline.select_threshold(scores[thr_split], y[thr_split], b)
            op = _operating_report(pipeline, ye, se, sege, t)
            rep["sweep"].append({"budget": b, "threshold": t, "precision": op["precision"],
                                 "recall": op["recall"], "fpr": op["fpr"]})
        if hasattr(scorer, "cutoffs"):
            rep["rules"] = {n: {"cutoff": v, "suspicious_when": "below" if scorer.directions[n] < 0 else "above"}
                            for n, v in scorer.cutoffs.items()}
        task.runs[scorer.name] = ModelRun(scorer, threshold, scores, rep)
    return task


def _decision_counts(ds: Dataset, ids: np.ndarray) -> np.ndarray:
    counts = np.bincount(ds.index_of(ds.decisions.player), minlength=len(ds.players.id))
    return counts[ds.index_of(ids)]


def player_task(pipeline: Pipeline, ds: Dataset, split: Split, cfg: EvalConfig) -> TaskRun:
    table = pipeline.player_features(ds).drop(cfg.exclude_groups)
    ids, y, seg, counts = {}, {}, {}, {}
    for s in SPLITS:
        all_ids = split.get(s)
        sg = ds.segment_of(all_ids)
        keep = ~np.isin(sg, OUT_OF_PLAYER_TASK)
        ids[s], seg[s] = all_ids[keep], sg[keep]
        y[s] = np.isin(seg[s], PLAYER_TASK_POSITIVE).astype(int)
        counts[s] = _decision_counts(ds, ids[s])
    return _run_task(pipeline, "player", table, ids, y, seg, counts, PLAYER_RULES, cfg)


def pair_task(pipeline: Pipeline, ds: Dataset, split: Split, cfg: EvalConfig) -> TaskRun:
    table = pipeline.pair_features(ds, cfg.min_shared_sessions)
    lo, hi = pair_members(table.ids)
    ids, y, seg, counts = {}, {}, {}, {}
    between = table.column("decisions_between")
    for s in SPLITS:
        members = split.get(s)
        keep = np.isin(lo, members) & np.isin(hi, members)
        ids[s] = table.ids[keep]
        seg[s] = pair_segments(ds, ids[s])
        y[s] = np.isin(seg[s], PAIR_POSITIVE).astype(int)
        counts[s] = between[keep].astype(int)
    return _run_task(pipeline, "pair", table, ids, y, seg, counts, PAIR_RULES, cfg)


def _out_of_task(ds: Dataset, pipeline: Pipeline, split: Split, player: TaskRun, cfg: EvalConfig) -> dict:
    """How often the player task flags colluders, whom it was not built for."""
    ev_ids = split.get(pipeline.eval_split)
    col = ev_ids[np.isin(ds.segment_of(ev_ids), OUT_OF_PLAYER_TASK)]
    X = player.table.rows(col)
    out = {}
    for name, run in player.runs.items():
        k = int((run.scorer.score(X) >= run.threshold).sum()) if len(col) else 0
        out[name] = {"n": int(len(col)), "flagged": k}
    return out


def _scenario(pipeline: Pipeline, player: TaskRun, ds: Dataset, cfg: EvalConfig) -> dict:
    table = pipeline.player_features(ds).drop(cfg.exclude_groups)
    seg = ds.segment_of(table.ids)
    keep = ~np.isin(seg, OUT_OF_PLAYER_TASK)
    X, seg = table.X[keep], seg[keep]
    y = np.isin(seg, PLAYER_TASK_POSITIVE).astype(int)
    out = {}
    for name, run in player.runs.items():
        s = run.scorer.score(X)
        out[name] = {"pr_auc": pipeline.average_precision(y, s, pipeline.pr_curve),
                     **_operating_report(pipeline, y, s, seg, run.threshold)}
    return out


def evaluate(ds: Dataset, cfg: EvalConfig = EvalConfig(), pipeline: Pipeline = REAL,
             sim_cfg: SimConfig | None = None, with_pairs: bool = True) -> Evaluation:
    """Run both tasks and the robustness scenarios. `with_pairs=False` skips
    the pair task (the checks use it to save time)."""
    split = pipeline.split(ds, cfg.seed)
    player = player_task(pipeline, ds, split, cfg)
    pair = pair_task(pipeline, ds, split, cfg) if with_pairs else None

    scenarios = {}
    if cfg.scenarios:
        base = sim_cfg or default_config()
        runs = {"fresh_population": base, **{n: SCENARIOS[n](base) for n in cfg.scenarios}}
        for name, sc in runs.items():
            # the same new seed for every scenario, so each differs from the
            # fresh population only by its change to the generator
            scen_ds = generate(sc, seed=cfg.seed + 10_000, description=name)
            scenarios[name] = _scenario(pipeline, player, scen_ds, cfg)

    counts = {s: {seg: int(n) for seg, n in zip(*np.unique(ds.segment_of(split.get(s)), return_counts=True))}
              for s in SPLITS}
    report = {
        "tool": f"game-integrity-lab {__version__}",
        "synthetic": True,
        "data": {"fingerprint": ds.fingerprint(), "seed": ds.seed, "description": ds.description,
                 "players": int(len(ds.players.id)), "decisions": int(len(ds.decisions.player)),
                 "split_segment_counts": counts},
        "config": {"seed": cfg.seed, "budget": cfg.budget, "exclude_groups": list(cfg.exclude_groups),
                   "features": list(player.table.names), "threshold_split": pipeline.threshold_split,
                   "eval_split": pipeline.eval_split, "min_shared_sessions": cfg.min_shared_sessions},
        "player_task": {"positives": list(PLAYER_TASK_POSITIVE), "legitimate": list(LEGITIMATE),
                        "models": {n: r.report for n, r in player.runs.items()},
                        "out_of_task_colluders": _out_of_task(ds, pipeline, split, player, cfg)},
        "pair_task": {"positives": list(PAIR_POSITIVE), "candidate_pairs": int(len(pair.table.ids)),
                      "models": {n: r.report for n, r in pair.runs.items()}} if pair else None,
        "robustness": scenarios,
    }
    return Evaluation(report, player, pair, split)
