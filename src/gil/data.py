"""The synthetic dataset: players, sessions and decisions, as numpy arrays.

Columns are either observable (what an operator's logs would hold) or ground
truth (what only the generator knows). Features may read observable columns
only; `validation/checks.py` has a check that scrambles every ground-truth
column and requires the features not to change.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np

# Player segments. The player task flags bots and RTA users; colluders are the
# target of the pair task. Everyone else is legitimate.
LEGITIMATE = ("recreational", "regular", "multi_tabler")
PLAYER_TASK_POSITIVE = ("bot", "rta")
OUT_OF_PLAYER_TASK = ("colluder",)
SEGMENTS = LEGITIMATE + PLAYER_TASK_POSITIVE + OUT_OF_PLAYER_TASK

# Decision policy codes (ground truth): where the action came from.
POLICY_OWN, POLICY_MISTAKE, POLICY_ASSISTED, POLICY_SOFTPLAY, POLICY_DUMP = range(5)
RAISE = 2

TRUTH_COLUMNS = {
    "players": ("segment", "pair_id", "pair_kind", "chip_dumper"),
    "decisions": ("policy",),
}


@dataclass(frozen=True)
class Players:
    id: np.ndarray            # observable: account id
    pool: np.ndarray          # observable: stake pool
    segment: np.ndarray       # truth
    pair_id: np.ndarray       # truth: -1 unless part of a colluding or friend pair
    pair_kind: np.ndarray     # truth: "", "colluding" or "friends"
    chip_dumper: np.ndarray   # truth


@dataclass(frozen=True)
class Sessions:
    player: np.ndarray        # account id
    start_hour: np.ndarray    # hour of day the session started, [0, 24)
    hours: np.ndarray         # session length


@dataclass(frozen=True)
class Decisions:
    player: np.ndarray        # account id
    session: np.ndarray       # row index into Sessions
    villain: np.ndarray       # account id of the main opponent in the decision
    difficulty: np.ndarray    # [0, 1], how hard the spot is for a human
    best_action: np.ndarray   # the reference's highest-EV action
    action: np.ndarray        # the action taken
    ev_loss: np.ndarray       # EV of the best action minus EV of the action taken, big blinds
    time_s: np.ndarray        # decision time, seconds
    hours_in: np.ndarray      # hours into the session when the decision was made
    policy: np.ndarray        # truth: POLICY_* code


@dataclass(frozen=True)
class Dataset:
    players: Players
    sessions: Sessions
    decisions: Decisions
    seed: int
    description: str = ""

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for table in (self.players, self.sessions, self.decisions):
            for f in fields(table):
                arr = np.ascontiguousarray(getattr(table, f.name))
                h.update(f.name.encode())
                h.update(arr.astype(str).tobytes() if arr.dtype.kind in "UO" else arr.tobytes())
        return h.hexdigest()[:16]

    def with_truth_scrambled(self, seed: int) -> "Dataset":
        """The same observable data with every ground-truth column randomised."""
        rng = np.random.default_rng(seed)
        p = self.players
        players = replace(
            p,
            segment=rng.permutation(p.segment),
            pair_id=rng.permutation(p.pair_id),
            pair_kind=rng.permutation(p.pair_kind),
            chip_dumper=rng.permutation(p.chip_dumper),
        )
        decisions = replace(self.decisions, policy=rng.integers(0, 5, len(self.decisions.policy)))
        return replace(self, players=players, decisions=decisions)

    def segment_of(self, ids: np.ndarray) -> np.ndarray:
        return self.players.segment[self.index_of(ids)]

    def index_of(self, ids: np.ndarray) -> np.ndarray:
        order = np.argsort(self.players.id)
        pos = np.searchsorted(self.players.id, ids, sorter=order)
        return order[pos]


def _prefixed(name: str, table) -> dict[str, np.ndarray]:
    return {f"{name}.{f.name}": getattr(table, f.name) for f in fields(table)}


def save(ds: Dataset, directory: Path) -> None:
    """Write data.npz (everything) and players.csv (readable, includes truth)."""
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {**_prefixed("players", ds.players), **_prefixed("sessions", ds.sessions), **_prefixed("decisions", ds.decisions)}
    np.savez_compressed(directory / "data.npz", seed=np.array(ds.seed), description=np.array(ds.description), **arrays)
    with open(directory / "players.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        names = [f.name for f in fields(Players)]
        w.writerow(names)
        for row in zip(*(getattr(ds.players, n) for n in names)):
            w.writerow([x.item() if hasattr(x, "item") else x for x in row])


def load(directory: Path) -> Dataset:
    z = np.load(directory / "data.npz", allow_pickle=False)

    def table(cls, name):
        return cls(**{f.name: z[f"{name}.{f.name}"] for f in fields(cls)})

    return Dataset(
        players=table(Players, "players"),
        sessions=table(Sessions, "sessions"),
        decisions=table(Decisions, "decisions"),
        seed=int(z["seed"]),
        description=str(z["description"]),
    )
