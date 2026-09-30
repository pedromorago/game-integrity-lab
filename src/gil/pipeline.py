"""The evaluation pipeline as a set of replaceable parts.

`validation/mutants.py` builds broken pipelines by replacing one part at a
time, and `validation/checks.py` runs against any pipeline. The real one is
`REAL`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import features, metrics, models, splits, thresholds


def per_unit(ids: np.ndarray, y: np.ndarray, scores: np.ndarray, segments: np.ndarray,
             counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The unit of evaluation is the player (or the pair): one row each,
    however many decisions it played."""
    return y, scores, segments


@dataclass(frozen=True)
class Pipeline:
    name: str = "real pipeline"
    split: Callable = splits.split_by_pool
    player_features: Callable = features.player_features
    pair_features: Callable = features.pair_features
    models: Callable = models.default_models
    confusion: Callable = metrics.confusion
    precision_recall: Callable = metrics.precision_recall
    pr_curve: Callable = metrics.pr_curve
    average_precision: Callable = metrics.average_precision
    select_threshold: Callable = thresholds.threshold_for_fpr_budget
    headline: Callable = metrics.headline
    eval_units: Callable = per_unit
    threshold_split: str = "val"   # where the operating threshold is chosen
    eval_split: str = "test"       # where the reported numbers come from


REAL = Pipeline()
