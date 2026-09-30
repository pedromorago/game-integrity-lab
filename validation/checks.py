"""Checks on the evaluation harness itself.

A model evaluation is software, and it can be wrong in ways that make every
model look better (or worse) than it is. Every check here takes a Pipeline
(gil.pipeline) and raises AssertionError with a readable message when it
fails. Each one names the kind of oracle it relies on:

- reference:    metrics against hand-computed small cases, or against sklearn
- invariant:    properties any correct report has (ranges, counts that add up)
- leakage:      information that must not flow from labels or test data into
                features, training or threshold choice
- sanity:       a model that cannot know anything must not look good
- metamorphic:  a change to the input that must leave the output unchanged
- determinism:  the same seed gives the same result
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from gil.config import small_config
from gil.data import OUT_OF_PLAYER_TASK, Dataset
from gil.evaluate import EvalConfig, Evaluation, evaluate
from gil.models import GradientBoostingScorer, IsolationForestScorer
from gil.pipeline import Pipeline
from gil.simulate import generate
from gil.splits import SPLITS

SEED = 7
CFG = EvalConfig(seed=SEED, scenarios=())


@dataclass(frozen=True)
class Check:
    name: str
    kind: str
    run: Callable[[Pipeline], None]


CHECKS: list[Check] = []


def check(kind: str):
    def register(fn: Callable[[Pipeline], None]) -> Callable[[Pipeline], None]:
        CHECKS.append(Check(fn.__name__, kind, fn))
        return fn

    return register


@lru_cache(maxsize=1)
def dataset() -> Dataset:
    return generate(small_config(), seed=11, description="check data")


@lru_cache(maxsize=None)
def base_run(pipeline: Pipeline) -> Evaluation:
    return evaluate(dataset(), CFG, pipeline)


def _tasks(ev: Evaluation):
    return [t for t in (ev.player, ev.pair) if t is not None]


# ------------------------------------------------------------------ reference


@check("reference")
def precision_recall_by_hand(p: Pipeline) -> None:
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    flagged = np.array([1, 1, 0, 1, 1, 0, 0, 0, 0, 0])
    c = p.confusion(y, flagged)
    assert (c.tp, c.fp, c.tn, c.fn) == (2, 2, 5, 1), f"confusion {c}"
    precision, recall = p.precision_recall(y, flagged)
    assert abs(precision - 0.5) < 1e-12 and abs(recall - 2 / 3) < 1e-12, (
        f"precision {precision}, recall {recall}; expected 0.5 and 0.667")
    assert p.precision_recall(y, np.zeros(10)) == (0.0, 0.0), "nothing flagged must give 0, 0"


@check("reference")
def pr_auc_by_hand(p: Pipeline) -> None:
    cases = [
        # thresholds 0.9, 0.8, 0.7, 0.1: (P, R) = (1, .5), (.5, .5), (2/3, 1), (.5, 1)
        (np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1]), 0.5 * 1 + 0.5 * 2 / 3),
        # a tie at the top counts as one threshold: (.5, .5), then (2/3, 1)
        (np.array([1, 0, 1]), np.array([0.5, 0.5, 0.2]), 0.5 * 0.5 + 0.5 * 2 / 3),
        (np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4]), 1.0),
    ]
    for y, s, expected in cases:
        got = p.average_precision(y, s, p.pr_curve)
        assert abs(got - expected) < 1e-12, f"PR-AUC of {s} with labels {y}: {got}, expected {expected}"


@check("reference")
def pr_curve_matches_sklearn(p: Pipeline) -> None:
    rng = np.random.default_rng(0)
    for i in range(40):
        n = int(rng.integers(5, 200))
        y = (rng.random(n) < rng.uniform(0.05, 0.5)).astype(int)
        y[0] = 1
        s = np.round(rng.random(n), int(rng.integers(1, 4)))  # rounding creates ties
        got = p.average_precision(y, s, p.pr_curve)
        want = average_precision_score(y, s)
        assert abs(got - want) < 1e-12, f"case {i}: PR-AUC {got}, sklearn {want}"
        prec, rec, thr = p.pr_curve(y, s)
        sk_p, sk_r, sk_t = precision_recall_curve(y, s)
        ours = {float(t): (a, b) for a, b, t in zip(prec, rec, thr)}
        for a, b, t in zip(sk_p[:-1], sk_r[:-1], sk_t):
            assert float(t) in ours, f"case {i}: threshold {t} missing from the curve"
            assert np.allclose(ours[float(t)], (a, b)), f"case {i}: at {t} got {ours[float(t)]}, sklearn {(a, b)}"


@check("reference")
def threshold_by_hand(p: Pipeline) -> None:
    legit = np.linspace(0.1, 1.0, 10)
    scores = np.r_[legit, np.full(10, 0.95)]
    y = np.r_[np.zeros(10), np.ones(10)]
    # 10 legitimate players and a 10% budget: exactly one of them may be flagged
    t = p.select_threshold(scores, y, 0.10)
    n_fp = int((legit >= t).sum())
    assert n_fp == 1, f"budget 10% of 10 legitimate players allows 1 flag, threshold {t} flags {n_fp}"
    assert (scores[10:] >= t).all(), "the positives above the cut must be flagged"
    t = p.select_threshold(scores, y, 0.05)
    assert int((legit >= t).sum()) == 0, f"budget 5% of 10 allows no flag, threshold {t} flags some"


# ------------------------------------------------------------------ invariant


def _unit_values(rep: dict) -> list[tuple[str, float]]:
    vals = [(k, rep[k]) for k in ("pr_auc", "precision", "recall", "fpr", "base_rate", "threshold_split_fpr")]
    vals += [(f"headline.{k}", v) for k, v in rep["headline"].items()]
    vals += [(f"sweep[{s['budget']}].{k}", s[k]) for s in rep["sweep"] for k in ("precision", "recall", "fpr")]
    vals += [(f"segments.{g}.rate", v["rate"]) for g, v in rep["segments"].items()]
    return vals


@check("invariant")
def metrics_in_range(p: Pipeline) -> None:
    ev = base_run(p)
    for task in _tasks(ev):
        for name, run in task.runs.items():
            for key, v in _unit_values(run.report):
                assert 0.0 <= v <= 1.0, f"{task.name}/{name}: {key} = {v} is outside [0, 1]"


@check("invariant")
def counts_add_up(p: Pipeline) -> None:
    ev = base_run(p)
    for task in _tasks(ev):
        ids = task.ids[p.eval_split]
        y = task.y[p.eval_split]
        for name, run in task.runs.items():
            rep = run.report
            c = rep["confusion"]
            total = c["tp"] + c["fp"] + c["tn"] + c["fn"]
            assert total == len(ids), (
                f"{task.name}/{name}: confusion counts {total} units, the evaluated set has {len(ids)}")
            assert c["tp"] + c["fn"] == int(y.sum()), f"{task.name}/{name}: tp + fn != positives"
            seg_n = sum(s["n"] for s in rep["segments"].values())
            seg_flagged = sum(s["flagged"] for s in rep["segments"].values())
            assert seg_n == total, f"{task.name}/{name}: segments cover {seg_n} units of {total}"
            assert seg_flagged == c["tp"] + c["fp"], f"{task.name}/{name}: segment flags != tp + fp"


@check("invariant")
def thresholds_keep_the_budget_on_validation(p: Pipeline) -> None:
    ev = base_run(p)
    for task in _tasks(ev):
        legit = task.y["val"] == 0
        for name, run in task.runs.items():
            s = run.scores["val"][legit]
            chosen = [(CFG.budget, run.threshold)] + [(w["budget"], w["threshold"]) for w in run.report["sweep"]]
            for budget, t in chosen:
                fpr = float((s >= t).mean())
                assert fpr <= budget + 1e-12, (
                    f"{task.name}/{name}: threshold for a {budget:.1%} budget flags {fpr:.2%} "
                    f"of legitimate validation units")


# ------------------------------------------------------------------ leakage


@check("leakage")
def no_player_in_two_splits(p: Pipeline) -> None:
    ds = dataset()
    split = p.split(ds, SEED)
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            both = np.intersect1d(split.get(a), split.get(b))
            assert len(both) == 0, f"{len(both)} players are in both {a} and {b}, e.g. {both[:3].tolist()}"
    covered = np.sort(np.concatenate([split.get(s) for s in SPLITS]))
    assert np.array_equal(covered, np.sort(ds.players.id)), "the splits do not cover every player exactly once"


@check("leakage")
def features_ignore_ground_truth(p: Pipeline) -> None:
    _features_ignore_ground_truth(p.player_features, p.pair_features)


@lru_cache(maxsize=None)
def _features_ignore_ground_truth(*fns) -> None:
    ds = dataset()
    scrambled = ds.with_truth_scrambled(seed=1)
    for fn in fns:
        a, b = fn(ds), fn(scrambled)
        assert a.names == b.names and np.array_equal(a.ids, b.ids), f"{fn.__name__}: rows or columns changed"
        diff = [n for i, n in enumerate(a.names) if not np.array_equal(a.X[:, i], b.X[:, i])]
        assert not diff, f"{fn.__name__}: features change when only ground truth changes: {diff}"


@check("leakage")
def threshold_ignores_test_labels(p: Pipeline) -> None:
    ds = dataset()
    test = p.split(ds, SEED).test
    idx = ds.index_of(test)
    idx = idx[~np.isin(ds.players.segment[idx], OUT_OF_PLAYER_TASK)]
    seg = ds.players.segment.copy()
    seg[idx] = np.random.default_rng(3).permutation(seg[idx])
    other = evaluate(replace(ds, players=replace(ds.players, segment=seg)), CFG, p, with_pairs=False)
    for name, run in base_run(p).player.runs.items():
        t2 = other.player.runs[name].threshold
        assert run.threshold == t2, (
            f"player/{name}: shuffling test labels moved the threshold from {run.threshold} to {t2}")


@check("leakage")
def reported_units_were_not_trained_on(p: Pipeline) -> None:
    ev = base_run(p)
    for task in _tasks(ev):
        seen = np.intersect1d(task.ids["train"], task.ids[p.eval_split])
        assert len(seen) == 0, f"{task.name}: {len(seen)} evaluated units were in the training set"


# ------------------------------------------------------------------ sanity


def _labels_permuted(ds: Dataset, seed: int) -> Dataset:
    """The same data with player-task labels shuffled across every player."""
    seg = ds.players.segment.copy()
    idx = np.flatnonzero(~np.isin(seg, OUT_OF_PLAYER_TASK))
    seg[idx] = np.random.default_rng(seed).permutation(seg[idx])
    return replace(ds, players=replace(ds.players, segment=seg))


@check("sanity")
def shuffled_labels_fall_to_base_rate(p: Pipeline) -> None:
    """Permute the labels over the whole dataset and run the whole pipeline.
    There is nothing left to learn, so a supervised model must score about
    the base rate on the evaluated players. It scores more only if it is
    evaluated on players it has memorised."""
    aps, bases = [], []
    for k in range(3):
        ev = evaluate(_labels_permuted(dataset(), k), CFG, p, with_pairs=False)
        rep = ev.player.runs["gradient_boosting"].report
        aps.append(rep["pr_auc"])
        bases.append(rep["base_rate"])
    gap = np.mean(aps) - np.mean(bases)
    assert gap < 0.12, (
        f"with permuted labels, PR-AUC is {np.mean(aps):.3f} against a base rate of {np.mean(bases):.3f}")


@check("sanity")
def flagging_nobody_scores_nothing(p: Pipeline) -> None:
    ev = base_run(p)
    y = ev.player.y[p.eval_split]
    base = float(np.mean(y))
    for key, v in p.headline(y, np.zeros(len(y), bool)).items():
        assert v <= base + 1e-12, (
            f"a model that flags no one gets headline {key} = {v:.3f} (base rate {base:.3f})")


@check("sanity")
def every_model_beats_chance(p: Pipeline) -> None:
    ev = base_run(p)
    for task in _tasks(ev):
        y = task.y["val"]
        base = float(y.mean())
        for name, run in task.runs.items():
            ap = p.average_precision(y, run.scores["val"], p.pr_curve)
            assert ap > base, f"{task.name}/{name}: validation PR-AUC {ap:.3f} is not above the base rate {base:.3f}"


# ------------------------------------------------------------------ metamorphic


@check("metamorphic")
def row_order_does_not_matter(p: Pipeline) -> None:
    ev = base_run(p)
    rng = np.random.default_rng(5)
    for task in _tasks(ev):
        y = task.y[p.eval_split]
        for name, run in task.runs.items():
            s = run.scores[p.eval_split]
            flagged = s >= run.threshold
            perm = rng.permutation(len(y))
            assert p.confusion(y, flagged) == p.confusion(y[perm], flagged[perm]), f"{task.name}/{name}: confusion"
            assert p.precision_recall(y, flagged) == p.precision_recall(y[perm], flagged[perm]), f"{task.name}/{name}: P/R"
            a, b = p.average_precision(y, s, p.pr_curve), p.average_precision(y[perm], s[perm], p.pr_curve)
            assert abs(a - b) < 1e-12, f"{task.name}/{name}: PR-AUC {a} becomes {b} when rows are reordered"


@check("metamorphic")
def duplicating_the_dataset_changes_nothing(p: Pipeline) -> None:
    ev = base_run(p)
    for task in _tasks(ev):
        y = task.y[p.eval_split]
        yv = task.y["val"]
        for name, run in task.runs.items():
            s = run.scores[p.eval_split]
            f = s >= run.threshold
            y2, s2, f2 = np.r_[y, y], np.r_[s, s], np.r_[f, f]
            pr1, pr2 = p.precision_recall(y, f), p.precision_recall(y2, f2)
            assert np.allclose(pr1, pr2), f"{task.name}/{name}: precision/recall {pr1} become {pr2} on a doubled set"
            a, b = p.average_precision(y, s, p.pr_curve), p.average_precision(y2, s2, p.pr_curve)
            assert abs(a - b) < 1e-12, f"{task.name}/{name}: PR-AUC {a} becomes {b} on a doubled set"
            sv = run.scores["val"]
            t1 = p.select_threshold(sv, yv, CFG.budget)
            t2 = p.select_threshold(np.r_[sv, sv], np.r_[yv, yv], CFG.budget)
            assert t1 == t2, f"{task.name}/{name}: threshold {t1} becomes {t2} on a doubled validation set"


@lru_cache(maxsize=None)
def _rescaling_differences(models) -> tuple[tuple[str, float], ...]:
    ev = base_run(Pipeline(models=models))
    task = ev.player
    X = task.table.rows(task.ids["train"])
    y = task.y["train"]
    out = []
    for scorer in models(SEED, {}):
        if isinstance(scorer, GradientBoostingScorer):
            f = lambda x: np.sign(x) * np.abs(x) ** 3 + x  # noqa: E731
        elif isinstance(scorer, IsolationForestScorer):
            f = lambda x: 3.0 * x + 7.0  # noqa: E731
        else:
            continue
        a = scorer.fit(X, y, task.table.names).score(X)
        b = scorer.fit(f(X), y, task.table.names).score(f(X))
        out.append((scorer.name, float(np.abs(a - b).max())))
    return tuple(out)


@check("metamorphic")
def monotone_rescaling_keeps_model_scores(p: Pipeline) -> None:
    """A tree split separates the training points by the order of one
    feature, so a strictly increasing transform of every feature must give
    gradient boosting exactly the same scores on the points it was trained
    on. (Unseen points can move: a split sits at the midpoint between two
    training values, and midpoints do not survive a nonlinear transform.)
    Isolation Forest draws split points uniformly between min and max, so it
    is only invariant to positive affine maps."""
    for name, diff in _rescaling_differences(p.models):
        assert diff < 1e-12, f"{name}: training scores move by up to {diff:.2e} under a monotone rescaling"


# ------------------------------------------------------------------ determinism


@check("determinism")
def same_seed_same_report(p: Pipeline) -> None:
    a = base_run(p).report["player_task"]
    b = evaluate(dataset(), CFG, p, with_pairs=False).report["player_task"]
    ja, jb = json.dumps(a, sort_keys=True, default=float), json.dumps(b, sort_keys=True, default=float)
    assert ja == jb, "two runs with the same seed give different player-task reports"


@check("determinism")
def generator_is_seeded(p: Pipeline) -> None:
    _generator_is_seeded()


@lru_cache(maxsize=1)
def _generator_is_seeded() -> None:
    cfg = small_config()
    a, b, c = generate(cfg, 5).fingerprint(), generate(cfg, 5).fingerprint(), generate(cfg, 6).fingerprint()
    assert a == b, "the same seed generated different data"
    assert a != c, "different seeds generated the same data"
