"""Three scorers with one interface: higher score means more suspicious.

- RuleBaseline: transparent rules, each a threshold on one feature set at a
  quantile of the legitimate training players. The score is how many fire.
- IsolationForestScorer: unsupervised; fitted on training players without labels.
- GradientBoostingScorer: supervised, sklearn's GradientBoostingClassifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest


class Scorer(Protocol):
    name: str

    def fit(self, X: np.ndarray, y: np.ndarray, names: tuple[str, ...]) -> "Scorer": ...

    def score(self, X: np.ndarray) -> np.ndarray: ...


# Suspicious direction per feature: -1 means low values are suspicious.
PLAYER_RULES: dict[str, int] = {
    "time_cv": -1,               # machine-regular timing
    "time_difficulty_corr": -1,  # time does not grow with difficulty
    "loss_gap": -1,              # no worse on hard spots than on easy ones
    "session_hours_max": +1,     # very long sessions
    "hour_spread": +1,           # plays at all hours
    "ev_loss_mean": -1,          # near-reference play overall
}
PAIR_RULES: dict[str, int] = {
    "together_share": +1,        # sit together often
    "min_raise_ratio": -1,       # one of them rarely raises the other
    "loss_vs_partner_gap": +1,   # loses more against the partner than against others
}


@dataclass
class RuleBaseline:
    directions: dict[str, int]
    quantile: float = 0.02
    name: str = "rules"
    cutoffs: dict[str, float] = field(default_factory=dict)
    _cols: dict[str, int] = field(default_factory=dict)

    def fit(self, X: np.ndarray, y: np.ndarray, names: tuple[str, ...]) -> "RuleBaseline":
        legit = X[np.asarray(y) == 0]
        self._cols = {n: names.index(n) for n in self.directions if n in names}
        self.cutoffs = {}
        for n, col in self._cols.items():
            q = self.quantile if self.directions[n] < 0 else 1 - self.quantile
            self.cutoffs[n] = float(np.quantile(legit[:, col], q))
        return self

    def fired(self, X: np.ndarray) -> dict[str, np.ndarray]:
        out = {}
        for n, col in self._cols.items():
            v = X[:, col]
            out[n] = v < self.cutoffs[n] if self.directions[n] < 0 else v > self.cutoffs[n]
        return out

    def score(self, X: np.ndarray) -> np.ndarray:
        fired = self.fired(X)
        return np.sum(list(fired.values()), axis=0).astype(float) if fired else np.zeros(len(X))


@dataclass
class IsolationForestScorer:
    seed: int
    n_estimators: int = 100
    negate: bool = True  # sklearn's score_samples is higher for normal points
    name: str = "isolation_forest"
    model: IsolationForest | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, names: tuple[str, ...]) -> "IsolationForestScorer":
        self.model = IsolationForest(n_estimators=self.n_estimators, random_state=self.seed).fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        s = self.model.score_samples(X)
        return -s if self.negate else s


@dataclass
class GradientBoostingScorer:
    seed: int
    name: str = "gradient_boosting"
    model: GradientBoostingClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, names: tuple[str, ...]) -> "GradientBoostingScorer":
        self.model = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.1, random_state=self.seed,
        ).fit(X, y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def default_models(seed: int, rules: dict[str, int]) -> list[Scorer]:
    return [RuleBaseline(rules), IsolationForestScorer(seed), GradientBoostingScorer(seed)]
