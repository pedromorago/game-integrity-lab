"""Choosing the operating threshold under a false-positive budget."""

from __future__ import annotations

import math

import numpy as np


def threshold_for_fpr_budget(scores: np.ndarray, y: np.ndarray, budget: float) -> float:
    """The lowest threshold that flags at most `budget` of the legitimate units.

    Flagging is `score >= threshold`. With n legitimate units, at most
    floor(budget * n) of them may be flagged, so the threshold sits just above
    the (k+1)-th highest legitimate score. With k = 0 no legitimate unit is
    flagged. Returns -inf when the budget allows every legitimate unit to be
    flagged. Ties at the cut are all left unflagged, so the budget is never
    exceeded on the data the threshold was chosen on.
    """
    legit = np.sort(np.asarray(scores, dtype=float)[np.asarray(y) == 0])[::-1]
    k = math.floor(budget * len(legit) + 1e-9)
    if k >= len(legit):
        return -math.inf
    return float(np.nextafter(legit[k], math.inf))


def fpr_at(scores: np.ndarray, y: np.ndarray, threshold: float) -> float:
    legit = np.asarray(scores)[np.asarray(y) == 0]
    return float((legit >= threshold).mean()) if len(legit) else 0.0
