"""Deliberately broken evaluation pipelines, one bug each.

Every one of these is a mistake that is easy to make when evaluating a model
that flags players, and most of them make the model look better than it is.
`scripts/run_seeded_bugs.py` runs every check against every one of them.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from gil import metrics
from gil.data import POLICY_ASSISTED, Dataset
from gil.features import FeatureTable, player_features
from gil.models import GradientBoostingScorer, IsolationForestScorer, RuleBaseline
from gil.pipeline import REAL, Pipeline
from gil.splits import Split, split_by_pool


def _split_by_sessions(ds: Dataset, seed: int | None, fractions=(0.6, 0.2, 0.2)) -> Split:
    """Sessions are split at random and a player belongs to every split where
    they have a session, so most players end up in all three."""
    rng = np.random.default_rng(seed)
    part = rng.choice(3, size=len(ds.sessions.player), p=list(fractions))
    return Split(*(np.unique(ds.sessions.player[part == i]) for i in range(3)))


def _unseeded_split(ds: Dataset, seed: int | None, fractions=(0.6, 0.2, 0.2)) -> Split:
    return split_by_pool(ds, None, fractions)


def _swapped_precision_recall(y, flagged):
    p, r = metrics.precision_recall(y, flagged)
    return r, p


def _pr_curve_drops_last_point(y, scores):
    y = np.asarray(y).astype(float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    s, t = scores[order], y[order]
    last_of_tie = np.flatnonzero(np.diff(s) != 0)  # the final group is never closed
    tp = np.cumsum(t)[last_of_tie]
    precision = tp / (last_of_tie + 1)
    recall = tp / t.sum() if t.sum() else np.zeros_like(tp)
    return precision, recall, s[last_of_tie]


def _per_decision(ids, y, scores, segments, counts):
    """Every decision becomes a row carrying its player's label and score."""
    return np.repeat(y, counts), np.repeat(scores, counts), np.repeat(segments, counts)


def _accuracy_headline(y, flagged):
    y, f = np.asarray(y).astype(bool), np.asarray(flagged).astype(bool)
    return {"accuracy": float((y == f).mean())}


def _leaky_features(ds: Dataset) -> FeatureTable:
    """Adds the share of decisions marked as assisted, a column that exists
    only because the generator knows the truth (in real data: a column filled
    in after an investigation)."""
    ft = player_features(ds)
    g = ds.index_of(ds.decisions.player)
    n = len(ds.players.id)
    assisted = np.bincount(g, weights=(ds.decisions.policy == POLICY_ASSISTED).astype(float), minlength=n)
    share = assisted / np.maximum(np.bincount(g, minlength=n), 1)
    return FeatureTable(ft.ids, ft.names + ("assisted_share",), np.column_stack([ft.X, share]))


def _budget_over_all_players(scores, y, budget):
    """The number of allowed false positives is computed from every player,
    positives included, so more legitimate players get flagged."""
    scores = np.asarray(scores, dtype=float)
    legit = np.sort(scores[np.asarray(y) == 0])[::-1]
    k = math.floor(budget * len(scores) + 1e-9)
    if k >= len(legit):
        return -math.inf
    return float(np.nextafter(legit[k], math.inf))


def _models_with_raw_isolation_scores(seed, rules):
    return [RuleBaseline(rules), IsolationForestScorer(seed, negate=False), GradientBoostingScorer(seed)]


MUTANTS: list[tuple[Pipeline, str]] = [
    (replace(REAL, name="Players leak across splits (split by session)", split=_split_by_sessions), "split"),
    (replace(REAL, name="Split not seeded", split=_unseeded_split), "split"),
    (replace(REAL, name="Feature built from a ground-truth column", player_features=_leaky_features), "features"),
    (replace(REAL, name="Isolation Forest score sign not flipped", models=_models_with_raw_isolation_scores), "model"),
    (replace(REAL, name="Threshold tuned on the test set", threshold_split="test"), "threshold"),
    (replace(REAL, name="FP budget computed over all players", select_threshold=_budget_over_all_players), "threshold"),
    (replace(REAL, name="Evaluated on the training players", eval_split="train"), "evaluation"),
    (replace(REAL, name="Metrics over decisions, not players", eval_units=_per_decision), "evaluation"),
    (replace(REAL, name="Precision and recall swapped", precision_recall=_swapped_precision_recall), "metrics"),
    (replace(REAL, name="PR curve drops its last point", pr_curve=_pr_curve_drops_last_point), "metrics"),
    (replace(REAL, name="Accuracy as the headline metric", headline=_accuracy_headline), "report"),
]
