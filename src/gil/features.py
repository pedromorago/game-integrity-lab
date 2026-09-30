"""Behavioural features per player and per pair of players.

Features read observable columns only (see gil.data). Every feature is a
plain aggregate that a reviewer can recompute by hand from the decision log.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import RAISE, Dataset

EASY_MAX = 1 / 3
HARD_MIN = 2 / 3

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "accuracy": ("ev_loss_mean", "ev_loss_easy", "ev_loss_hard", "loss_gap", "agree_rate", "fatigue_gap"),
    "timing": ("log_time_mean", "time_cv", "time_difficulty_corr", "time_hard_minus_easy"),
    "schedule": ("session_hours_mean", "session_hours_max", "hour_spread", "decisions_per_hour"),
}
PLAYER_FEATURES = tuple(n for group in FEATURE_GROUPS.values() for n in group)

PAIR_FEATURES = (
    "shared_sessions", "together_share", "decisions_between",
    "min_raise_ratio", "mean_raise_ratio", "loss_vs_partner_gap",
)


@dataclass(frozen=True)
class FeatureTable:
    ids: np.ndarray          # player ids, or pair keys for the pair table
    names: tuple[str, ...]
    X: np.ndarray            # one row per id, one column per name

    def drop(self, groups: tuple[str, ...]) -> "FeatureTable":
        dropped = {n for g in groups for n in FEATURE_GROUPS[g]}
        keep = [i for i, n in enumerate(self.names) if n not in dropped]
        return FeatureTable(self.ids, tuple(self.names[i] for i in keep), self.X[:, keep])

    def rows(self, ids: np.ndarray) -> np.ndarray:
        order = np.argsort(self.ids)
        return self.X[order[np.searchsorted(self.ids, ids, sorter=order)]]

    def column(self, name: str) -> np.ndarray:
        return self.X[:, self.names.index(name)]


def _group_mean(values: np.ndarray, groups: np.ndarray, n: int, mask: np.ndarray | None = None,
                fallback: np.ndarray | None = None) -> np.ndarray:
    if mask is not None:
        values, groups = values[mask], groups[mask]
    counts = np.bincount(groups, minlength=n)
    sums = np.bincount(groups, weights=values, minlength=n)
    out = np.divide(sums, counts, out=np.zeros(n), where=counts > 0)
    if fallback is not None:
        out = np.where(counts > 0, out, fallback)
    return out


def player_features(ds: Dataset) -> FeatureTable:
    d, s = ds.decisions, ds.sessions
    ids = ds.players.id
    n = len(ids)
    g = ds.index_of(d.player)
    easy = d.difficulty < EASY_MAX
    hard = d.difficulty >= HARD_MIN

    loss = _group_mean(d.ev_loss, g, n)
    loss_easy = _group_mean(d.ev_loss, g, n, easy, fallback=loss)
    loss_hard = _group_mean(d.ev_loss, g, n, hard, fallback=loss)
    agree = _group_mean((d.action == d.best_action).astype(float), g, n)
    early = _group_mean(d.ev_loss, g, n, d.hours_in < 1.0, fallback=loss)
    late = _group_mean(d.ev_loss, g, n, d.hours_in >= 2.0, fallback=early)

    lt = np.log(d.time_s)
    lt_mean = _group_mean(lt, g, n)
    t_mean = _group_mean(d.time_s, g, n)
    t_sq = _group_mean(d.time_s**2, g, n)
    t_cv = np.sqrt(np.maximum(t_sq - t_mean**2, 0)) / t_mean
    diff_mean = _group_mean(d.difficulty, g, n)
    cov = _group_mean(lt * d.difficulty, g, n) - lt_mean * diff_mean
    var_lt = np.maximum(_group_mean(lt**2, g, n) - lt_mean**2, 1e-12)
    var_d = np.maximum(_group_mean(d.difficulty**2, g, n) - diff_mean**2, 1e-12)
    corr = cov / np.sqrt(var_lt * var_d)
    lt_easy = _group_mean(lt, g, n, easy, fallback=lt_mean)
    lt_hard = _group_mean(lt, g, n, hard, fallback=lt_mean)

    sg = ds.index_of(s.player)
    sess_count = np.bincount(sg, minlength=n)
    hours_total = np.bincount(sg, weights=s.hours, minlength=n)
    sess_mean = hours_total / np.maximum(sess_count, 1)
    sess_max = np.zeros(n)
    np.maximum.at(sess_max, sg, s.hours)
    angle = 2 * np.pi * s.start_hour / 24
    c = _group_mean(np.cos(angle), sg, n)
    si = _group_mean(np.sin(angle), sg, n)
    spread = 1 - np.sqrt(c**2 + si**2)
    dph = np.bincount(g, minlength=n) / np.maximum(hours_total, 1e-9)

    columns = {
        "ev_loss_mean": loss, "ev_loss_easy": loss_easy, "ev_loss_hard": loss_hard,
        "loss_gap": loss_hard - loss_easy, "agree_rate": agree, "fatigue_gap": late - early,
        "log_time_mean": lt_mean, "time_cv": t_cv, "time_difficulty_corr": corr,
        "time_hard_minus_easy": lt_hard - lt_easy,
        "session_hours_mean": sess_mean, "session_hours_max": sess_max, "hour_spread": spread,
        "decisions_per_hour": dph,
    }
    return FeatureTable(ids.copy(), PLAYER_FEATURES, np.column_stack([columns[k] for k in PLAYER_FEATURES]))


# ---------------------------------------------------------------------- pairs

PAIR_KEY = 10**7


def pair_key(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return lo.astype(np.int64) * PAIR_KEY + hi


def pair_members(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return keys // PAIR_KEY, keys % PAIR_KEY


def pair_features(ds: Dataset, min_shared_sessions: int = 5) -> FeatureTable:
    """Candidate pairs are those who met in at least `min_shared_sessions`
    sessions (either player's), as seen from the villain column."""
    d = ds.decisions
    n_players = len(ds.players.id)
    a_idx = ds.index_of(d.player)
    directed = d.player.astype(np.int64) * PAIR_KEY + d.villain
    dkeys, dinv, dn = np.unique(directed, return_inverse=True, return_counts=True)
    raises = np.bincount(dinv, weights=(d.action == RAISE).astype(float))
    losses = np.bincount(dinv, weights=d.ev_loss)
    sess_keys = np.unique(np.stack([directed, d.session]), axis=1)[0]
    dsess = np.searchsorted(dkeys, sess_keys)
    dsessions = np.bincount(dsess, minlength=len(dkeys))

    tot_n = np.bincount(a_idx, minlength=n_players).astype(float)
    tot_raise = np.bincount(a_idx, weights=(d.action == RAISE).astype(float), minlength=n_players)
    tot_loss = np.bincount(a_idx, weights=d.ev_loss, minlength=n_players)
    n_sessions = np.bincount(ds.index_of(ds.sessions.player), minlength=n_players)

    src, dst = dkeys // PAIR_KEY, dkeys % PAIR_KEY
    src_i = ds.index_of(src)
    others_n = tot_n[src_i] - dn
    raise_ratio = ((raises + 1) / (dn + 3)) / ((tot_raise[src_i] - raises + 1) / (others_n + 3))
    loss_gap = losses / dn - (tot_loss[src_i] - losses) / np.maximum(others_n, 1)

    ukeys, uinv = np.unique(pair_key(src, dst), return_inverse=True)
    m = len(ukeys)
    shared = np.bincount(uinv, weights=dsessions, minlength=m)
    between = np.bincount(uinv, weights=dn, minlength=m)
    # per direction: fill missing directions with neutral values
    rr_min = np.full(m, np.inf)
    np.minimum.at(rr_min, uinv, raise_ratio)
    has_both = np.bincount(uinv, minlength=m) == 2
    rr_min = np.where(has_both, rr_min, np.minimum(rr_min, 1.0))
    rr_mean = np.bincount(uinv, weights=raise_ratio, minlength=m) / np.bincount(uinv, minlength=m)
    lg_max = np.full(m, -np.inf)
    np.maximum.at(lg_max, uinv, loss_gap)
    lg_max = np.where(has_both, lg_max, np.maximum(lg_max, 0.0))

    lo, hi = pair_members(ukeys)
    together = shared / (n_sessions[ds.index_of(lo)] + n_sessions[ds.index_of(hi)])
    columns = {
        "shared_sessions": shared, "together_share": together, "decisions_between": between,
        "min_raise_ratio": rr_min, "mean_raise_ratio": rr_mean, "loss_vs_partner_gap": lg_max,
    }
    keep = shared >= min_shared_sessions
    X = np.column_stack([columns[k] for k in PAIR_FEATURES])[keep]
    return FeatureTable(ukeys[keep], PAIR_FEATURES, X)


def pair_segments(ds: Dataset, keys: np.ndarray) -> np.ndarray:
    """Ground truth for pairs: colluding, chip_dumping, friends or other."""
    lo, hi = pair_members(keys)
    p = ds.players
    ilo, ihi = ds.index_of(lo), ds.index_of(hi)
    same = (p.pair_id[ilo] == p.pair_id[ihi]) & (p.pair_id[ilo] >= 0)
    kind = np.where(same, p.pair_kind[ilo], "other")
    dumping = same & (p.chip_dumper[ilo] | p.chip_dumper[ihi])
    kind = np.where((kind == "colluding") & dumping, "chip_dumping", kind)
    return kind.astype("<U12")
