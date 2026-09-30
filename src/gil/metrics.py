"""Evaluation metrics, written out so that each one can be checked by hand.

Conventions: `y` is 1 for a positive (a player or pair the task should flag),
`scores` are higher for more suspicious, and a unit is flagged when its score
is at or above the threshold. Precision with nothing flagged is 0.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def confusion(y: np.ndarray, flagged: np.ndarray) -> Confusion:
    y = np.asarray(y).astype(bool)
    f = np.asarray(flagged).astype(bool)
    return Confusion(int((y & f).sum()), int((~y & f).sum()), int((~y & ~f).sum()), int((y & ~f).sum()))


def precision_recall(y: np.ndarray, flagged: np.ndarray) -> tuple[float, float]:
    c = confusion(y, flagged)
    precision = c.tp / (c.tp + c.fp) if c.tp + c.fp else 0.0
    recall = c.tp / (c.tp + c.fn) if c.tp + c.fn else 0.0
    return precision, recall


def false_positive_rate(y: np.ndarray, flagged: np.ndarray) -> float:
    c = confusion(y, flagged)
    return c.fp / (c.fp + c.tn) if c.fp + c.tn else 0.0


def pr_curve(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision and recall at every distinct score used as a threshold,
    from the highest threshold to the lowest (recall non-decreasing)."""
    y = np.asarray(y).astype(float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    s, t = scores[order], y[order]
    last_of_tie = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(t)[last_of_tie]
    flagged = last_of_tie + 1
    positives = t.sum()
    precision = tp / flagged
    recall = tp / positives if positives else np.zeros_like(tp)
    return precision, recall, s[last_of_tie]


def average_precision(y: np.ndarray, scores: np.ndarray, curve=pr_curve) -> float:
    """PR-AUC as the step-wise sum of precision times the gain in recall,
    the same definition as sklearn's average_precision_score."""
    precision, recall, _ = curve(y, scores)
    gains = np.diff(np.r_[0.0, recall])
    return float(np.sum(gains * precision))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def headline(y: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    """The numbers a report leads with at the operating threshold."""
    p, r = precision_recall(y, flagged)
    return {"precision": p, "recall": r}
