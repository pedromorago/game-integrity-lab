"""A seeded, decision-level simulation of players at poker tables.

This is not a poker engine. Each recorded decision is an abstract spot with a
difficulty, three actions with EVs, and a reference ("solver-like") action
distribution. Players differ in how often they deviate from the reference,
how long they take, how long they play and at what hours. All assumptions are
listed in docs/data-model.md; all parameters live in gil.config.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import Behaviour, SimConfig
from .data import (
    POLICY_ASSISTED, POLICY_DUMP, POLICY_MISTAKE, POLICY_OWN, POLICY_SOFTPLAY, RAISE,
    Dataset, Decisions, Players, Sessions,
)


@dataclass
class _Player:
    segment: str
    behaviour: Behaviour
    pair_id: int = -1
    pair_kind: str = ""
    chip_dumper: bool = False
    id: int = 0
    pool: int = 0
    partner: int = -1  # index into the roster


def _roster(cfg: SimConfig) -> list[_Player]:
    col = cfg.collusion
    roster: list[_Player] = []
    pair = 0
    dumping_pairs = round(col.colluding_pairs * col.chip_dump_share)
    for k in range(col.colluding_pairs):
        a = _Player("colluder", cfg.behaviours["colluder"], pair, "colluding", chip_dumper=k < dumping_pairs)
        b = _Player("colluder", cfg.behaviours["colluder"], pair, "colluding")
        roster += [a, b]
        pair += 1
    for segment, n in cfg.counts.items():
        roster += [_Player(segment, cfg.behaviours[segment]) for _ in range(n)]
    regulars = [p for p in roster if p.segment == "regular"]
    for k in range(min(col.friend_pairs, len(regulars) // 2)):
        for p in regulars[2 * k: 2 * k + 2]:
            p.pair_id, p.pair_kind = pair, "friends"
        pair += 1
    return roster


def _assign_ids_and_pools(roster: list[_Player], cfg: SimConfig, rng: np.random.Generator) -> None:
    ids = rng.permutation(len(roster)) + 1000  # ids carry no information about the segment
    for p, i in zip(roster, ids):
        p.id = int(i)
    by_pair: dict[int, list[int]] = {}
    for idx, p in enumerate(roster):
        if p.pair_id >= 0:
            by_pair.setdefault(p.pair_id, []).append(idx)
    for a, b in by_pair.values():
        roster[a].partner, roster[b].partner = b, a
    units = [members for members in by_pair.values()]
    units += [[i] for i, p in enumerate(roster) if p.pair_id < 0]
    order = rng.permutation(len(units))
    n_pools = max(1, round(len(roster) / cfg.pool_size))
    filled = 0
    for u in order:
        pool = min(n_pools - 1, filled // cfg.pool_size)
        for i in units[u]:
            roster[i].pool = pool
        filled += len(units[u])


def _individual(b: Behaviour, rng: np.random.Generator) -> Behaviour:
    """This player's own version of the population's behaviour."""
    f = np.exp(rng.normal(0, b.player_spread, 3))
    return replace(
        b,
        base_error=b.base_error * f[0], difficulty_error=b.difficulty_error * f[0],
        fatigue_per_hour=b.fatigue_per_hour * f[0],
        time_median_s=b.time_median_s * f[1], time_difficulty=b.time_difficulty * f[2],
    )


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = x / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def generate(cfg: SimConfig, seed: int, description: str = "") -> Dataset:
    rng = np.random.default_rng(seed)
    roster = _roster(cfg)
    _assign_ids_and_pools(roster, cfg, rng)
    pool_of = np.array([p.pool for p in roster])
    pools = {k: np.flatnonzero(pool_of == k) for k in np.unique(pool_of)}
    sp = cfg.spots
    col = cfg.collusion

    s_player, s_start, s_hours = [], [], []
    cols: dict[str, list[np.ndarray]] = {k: [] for k in (
        "player", "session", "villain", "difficulty", "best_action", "action", "ev_loss", "time_s", "hours_in", "policy")}

    for idx, p in enumerate(roster):
        b = _individual(p.behaviour, rng)
        n_sess = int(rng.integers(b.sessions[0], b.sessions[1] + 1))
        hours = rng.uniform(*b.session_hours, n_sess)
        if b.start_hour_sd is None:
            starts = rng.uniform(0, 24, n_sess)
        else:
            starts = (rng.uniform(0, 24) + rng.normal(0, b.start_hour_sd, n_sess)) % 24
        n_dec = np.maximum(1, rng.poisson(b.decisions_per_hour * hours))
        first_session = len(s_player)
        s_player += [p.id] * n_sess
        s_start += list(starts)
        s_hours += list(hours)

        # opponents per session, from the same pool; a partner joins with some probability
        members = pools[p.pool][pools[p.pool] != idx]
        k = min(sp.opponents_per_session, len(members))
        opp = np.stack([rng.choice(members, k, replace=False) for _ in range(n_sess)])
        if p.partner >= 0:
            prob = col.together_prob if p.pair_kind == "colluding" else col.friend_together_prob
            joins = rng.random(n_sess) < prob
            has = (opp == p.partner).any(axis=1)
            opp[joins & ~has, 0] = p.partner

        n = int(n_dec.sum())
        sess = np.repeat(np.arange(n_sess), n_dec)
        hours_in = rng.uniform(0, 1, n) * hours[sess]
        difficulty = rng.uniform(0, 1, n)
        best = rng.integers(0, sp.n_actions, n)
        gaps = rng.exponential(sp.ev_gap_scale, (n, sp.n_actions))
        gaps[np.arange(n), best] = 0.0
        ev = -gaps
        err = np.clip(b.base_error + b.difficulty_error * difficulty + b.fatigue_per_hour * hours_in, 0, 0.95)
        free = rng.random(n) < err
        assisted = (difficulty >= b.assist_min_difficulty) & (rng.random(n) < b.assist_rate)
        free &= ~assisted
        probs = np.where(free[:, None], _softmax(ev, sp.mistake_temperature), _softmax(ev, sp.reference_temperature))
        action = (rng.random(n)[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
        action = np.minimum(action, sp.n_actions - 1)
        policy = np.where(assisted, POLICY_ASSISTED, np.where(free, POLICY_MISTAKE, POLICY_OWN))

        villain_idx = opp[sess, rng.integers(0, k, n)]
        villain = np.array([roster[v].id for v in villain_idx]) if n else np.zeros(0, int)
        if p.pair_kind == "colluding":
            vs_partner = villain_idx == p.partner
            soften = vs_partner & (action == RAISE) & (rng.random(n) > col.softplay_keep_raise)
            action = np.where(soften, 1, action)
            policy = np.where(soften, POLICY_SOFTPLAY, policy)
            if p.chip_dumper:
                dump = vs_partner & (rng.random(n) < col.dump_prob)
                action = np.where(dump, np.argmin(ev, axis=1), action)
                policy = np.where(dump, POLICY_DUMP, policy)
        ev_loss = -ev[np.arange(n), action]

        log_t = (np.log(b.time_median_s) + b.time_difficulty * difficulty + b.time_fatigue * hours_in
                 + rng.normal(0, b.time_sigma, n))
        delay = np.maximum(0, rng.normal(b.assist_delay_s[0], b.assist_delay_s[1], n)) * assisted
        time_s = np.exp(log_t) + delay

        cols["player"].append(np.full(n, p.id))
        cols["session"].append(sess + first_session)
        cols["villain"].append(villain)
        cols["difficulty"].append(difficulty)
        cols["best_action"].append(best)
        cols["action"].append(action)
        cols["ev_loss"].append(ev_loss)
        cols["time_s"].append(time_s)
        cols["hours_in"].append(hours_in)
        cols["policy"].append(policy)

    players = Players(
        id=np.array([p.id for p in roster]),
        pool=np.array([p.pool for p in roster]),
        segment=np.array([p.segment for p in roster]),
        pair_id=np.array([p.pair_id for p in roster]),
        pair_kind=np.array([p.pair_kind for p in roster]),
        chip_dumper=np.array([p.chip_dumper for p in roster]),
    )
    sessions = Sessions(player=np.array(s_player), start_hour=np.array(s_start), hours=np.array(s_hours))
    decisions = Decisions(**{k: np.concatenate(v) for k, v in cols.items()})
    return Dataset(players, sessions, decisions, seed=seed, description=description)
