"""Metrics and threshold selection: hand-computed cases, sklearn, and properties."""

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import average_precision_score, precision_score, recall_score

from gil.metrics import average_precision, confusion, pr_curve, precision_recall, wilson_interval
from gil.thresholds import fpr_at, threshold_for_fpr_budget

labels_and_scores = st.integers(2, 60).flatmap(lambda n: st.tuples(
    st.lists(st.integers(0, 1), min_size=n, max_size=n),
    st.lists(st.integers(0, 8), min_size=n, max_size=n),  # few distinct values, so many ties
))


def test_confusion_by_hand():
    c = confusion(np.array([1, 0, 1, 0, 0]), np.array([1, 1, 0, 0, 0]))
    assert (c.tp, c.fp, c.tn, c.fn) == (1, 1, 2, 1)


def test_wilson_interval_known_value():
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0 and abs(hi - 0.0370) < 1e-3
    assert wilson_interval(0, 0) == (0.0, 1.0)


@settings(max_examples=200, deadline=None)
@given(labels_and_scores)
def test_average_precision_matches_sklearn(case):
    y, s = map(np.array, case)
    if y.sum() == 0:
        return
    assert math.isclose(average_precision(y, s), average_precision_score(y, s), abs_tol=1e-12)


@settings(max_examples=200, deadline=None)
@given(labels_and_scores, st.integers(0, 9))
def test_precision_recall_match_sklearn(case, t):
    y, s = map(np.array, case)
    flagged = (s >= t).astype(int)
    p, r = precision_recall(y, flagged)
    assert math.isclose(p, precision_score(y, flagged, zero_division=0))
    assert math.isclose(r, recall_score(y, flagged, zero_division=0))


@settings(max_examples=200, deadline=None)
@given(labels_and_scores)
def test_pr_curve_shape(case):
    y, s = map(np.array, case)
    precision, recall, thr = pr_curve(y, s)
    assert np.all(np.diff(thr) < 0), "one point per distinct score, highest first"
    assert np.all(np.diff(recall) >= 0)
    assert np.all((precision >= 0) & (precision <= 1))
    if y.sum():
        assert recall[-1] == 1.0


@settings(max_examples=300, deadline=None)
@given(labels_and_scores, st.floats(0.0, 0.5))
def test_threshold_keeps_budget_and_is_lowest(case, budget):
    y, s = map(np.array, case)
    s = s.astype(float)
    legit = s[y == 0]
    t = threshold_for_fpr_budget(s, y, budget)
    if len(legit) == 0:
        return
    assert fpr_at(s, y, t) <= budget + 1e-12
    # any lower threshold that flags one more legitimate score breaks the budget
    below = legit[legit < t]
    if len(below):
        assert fpr_at(s, y, below.max()) > budget


@settings(max_examples=100, deadline=None)
@given(labels_and_scores, st.floats(0.0, 0.5))
def test_threshold_ignores_positive_scores(case, budget):
    y, s = map(np.array, case)
    s = s.astype(float)
    moved = np.where(y == 1, s + 100.0, s)
    assert threshold_for_fpr_budget(s, y, budget) == threshold_for_fpr_budget(moved, y, budget)


@settings(max_examples=100, deadline=None)
@given(labels_and_scores)
def test_monotone_transform_keeps_average_precision(case):
    y, s = map(np.array, case)
    if y.sum() == 0:
        return
    s = s.astype(float)
    assert math.isclose(average_precision(y, s), average_precision(y, np.exp(s) * 3 - 1), abs_tol=1e-12)
