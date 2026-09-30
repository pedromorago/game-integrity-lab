"""Splitting players into train, validation and test.

The unit of splitting is the stake pool. Opponents come from the same pool,
so every pair of players who met lands in one split, and no player is in two.
The split reads only the observable pool column, never a label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Dataset

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def get(self, name: str) -> np.ndarray:
        return getattr(self, name)


def split_by_pool(ds: Dataset, seed: int | None, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> Split:
    players = ds.players
    rng = np.random.default_rng(seed)
    pools = rng.permutation(np.unique(players.pool))
    sizes = np.array([(players.pool == p).sum() for p in pools])
    cum = np.cumsum(sizes) / sizes.sum()
    # a pool goes to the split its cumulative midpoint falls in
    mid = cum - sizes / sizes.sum() / 2
    bounds = np.cumsum(fractions)
    part = np.searchsorted(bounds, mid, side="right")
    groups = [np.sort(players.id[np.isin(players.pool, pools[part == i])]) for i in range(3)]
    return Split(*groups)
