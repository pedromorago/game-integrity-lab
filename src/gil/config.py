"""Every parameter of the synthetic data generator, in one place.

Nothing here is estimated from real players. The numbers are chosen so that
each population has the qualitative signature described in docs/data-model.md,
and so that the populations overlap enough for the evaluation to be
interesting. Changing them changes every result in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Behaviour:
    """How one kind of player makes decisions, spends time and plays sessions."""

    # Decision quality. On each decision the player either follows the
    # reference distribution or makes a free choice (a softmax over EVs with a
    # high temperature, so mistakes still lean towards less costly actions).
    # P(free choice) = base_error + difficulty_error * difficulty
    #                  + fatigue_per_hour * hours_into_session, clipped to [0, 0.95].
    base_error: float
    difficulty_error: float
    fatigue_per_hour: float
    # Decision time in seconds: lognormal with median
    # time_median_s * exp(time_difficulty * difficulty + time_fatigue * hours_into_session).
    time_median_s: float
    time_difficulty: float
    time_sigma: float
    time_fatigue: float = 0.0
    # Sessions: a count drawn uniformly from the range, each lasting a length
    # in hours drawn uniformly from the range.
    sessions: tuple[int, int] = (6, 14)
    session_hours: tuple[float, float] = (1.0, 5.0)
    # How many decisions are recorded per hour of play. This is a sample of
    # decision points, not every hand.
    decisions_per_hour: float = 12.0
    # Session start hour: normal around a personal preferred hour with this
    # standard deviation (hours). None means uniform over the day.
    start_hour_sd: float | None = 2.0
    # Real-time assistance: on spots at least this difficult, with this
    # probability, the action is copied from the reference and the decision
    # takes extra time (normal, in seconds, floored at 0).
    assist_rate: float = 0.0
    assist_min_difficulty: float = 0.6
    assist_delay_s: tuple[float, float] = (0.0, 0.0)
    # Players within a population differ: each player's error rates, time
    # median and time-difficulty slope are multiplied by exp(N(0, player_spread)).
    player_spread: float = 0.35


RECREATIONAL = Behaviour(
    base_error=0.35, difficulty_error=0.30, fatigue_per_hour=0.03,
    time_median_s=4.0, time_difficulty=0.4, time_sigma=0.70, time_fatigue=0.0,
    sessions=(3, 10), session_hours=(0.5, 3.0), decisions_per_hour=10.0, start_hour_sd=2.5,
)

REGULAR = Behaviour(
    base_error=0.04, difficulty_error=0.30, fatigue_per_hour=0.015,
    time_median_s=2.5, time_difficulty=1.3, time_sigma=0.45, time_fatigue=0.03,
    sessions=(6, 14), session_hours=(1.5, 7.0), decisions_per_hour=12.0, start_hour_sd=2.0,
)

MULTI_TABLER = Behaviour(
    base_error=0.08, difficulty_error=0.25, fatigue_per_hour=0.03,
    time_median_s=1.4, time_difficulty=0.6, time_sigma=0.40, time_fatigue=0.05,
    sessions=(6, 14), session_hours=(2.0, 7.0), decisions_per_hour=30.0, start_hour_sd=1.5,
)

BOT = Behaviour(
    base_error=0.01, difficulty_error=0.0, fatigue_per_hour=0.0,
    time_median_s=2.0, time_difficulty=0.0, time_sigma=0.08, time_fatigue=0.0,
    sessions=(6, 12), session_hours=(8.0, 20.0), decisions_per_hour=12.0, start_hour_sd=None,
    player_spread=0.2,
)

RTA_USER = replace(
    REGULAR, base_error=0.08, assist_rate=0.5, assist_min_difficulty=0.6, assist_delay_s=(3.0, 1.5),
)


@dataclass(frozen=True)
class CollusionConfig:
    """Pairs of players who sit together, and what they do differently."""

    colluding_pairs: int = 50
    friend_pairs: int = 80          # legitimate pairs who also sit together often
    together_prob: float = 0.7      # P(partner is at the table in a session)
    friend_together_prob: float = 0.6
    # Soft play: a raise against the partner is kept with this probability,
    # otherwise it becomes a call.
    softplay_keep_raise: float = 0.5
    # Chip dumping: this share of colluding pairs also dumps chips; the dumper
    # picks the worst action against the partner with dump_prob.
    chip_dump_share: float = 0.3
    dump_prob: float = 0.3


@dataclass(frozen=True)
class SpotConfig:
    """The decision points. See docs/data-model.md."""

    n_actions: int = 3              # fold, call, raise (index 2 is the aggressive action)
    ev_gap_scale: float = 1.0       # EV gaps below the best action are exponential, in big blinds
    reference_temperature: float = 0.15
    mistake_temperature: float = 1.0
    opponents_per_session: int = 5


@dataclass(frozen=True)
class SimConfig:
    counts: dict[str, int] = field(default_factory=lambda: {
        "recreational": 1400, "regular": 900, "multi_tabler": 300, "bot": 160, "rta": 160,
    })
    behaviours: dict[str, Behaviour] = field(default_factory=lambda: {
        "recreational": RECREATIONAL, "regular": REGULAR, "multi_tabler": MULTI_TABLER,
        "bot": BOT, "rta": RTA_USER, "colluder": REGULAR,
    })
    collusion: CollusionConfig = field(default_factory=CollusionConfig)
    spots: SpotConfig = field(default_factory=SpotConfig)
    pool_size: int = 60             # players are split into stake pools; opponents come from the same pool


def default_config() -> SimConfig:
    return SimConfig()


def small_config() -> SimConfig:
    """A quarter of the default population, used by the checks and the tests."""
    return SimConfig(
        counts={"recreational": 350, "regular": 225, "multi_tabler": 75, "bot": 40, "rta": 40},
        collusion=replace(CollusionConfig(), colluding_pairs=12, friend_pairs=20),
        pool_size=40,
    )


# ------------------------------------------------------------ robustness scenarios
# Each scenario changes the generator, draws a new population with a different
# seed, and scores it with the models and thresholds trained on the base data.


def adaptive_bots(cfg: SimConfig) -> SimConfig:
    """Bots that add timing noise that grows with difficulty, make deliberate
    mistakes, and keep human-looking session lengths and hours."""
    bot = replace(
        cfg.behaviours["bot"], base_error=0.10, difficulty_error=0.15, time_difficulty=0.9,
        time_sigma=0.45, session_hours=(2.0, 6.0), start_hour_sd=2.0,
    )
    return replace(cfg, behaviours={**cfg.behaviours, "bot": bot})


def careful_rta(cfg: SimConfig) -> SimConfig:
    """RTA users who consult the tool on a quarter of the hard spots (half as
    often as in the base data), with half the extra delay."""
    rta = replace(cfg.behaviours["rta"], assist_rate=0.25, assist_delay_s=(1.5, 1.0))
    return replace(cfg, behaviours={**cfg.behaviours, "rta": rta})


def legitimate_drift(cfg: SimConfig) -> SimConfig:
    """The legitimate population changes: regulars study with solvers and make
    half the mistakes, sessions get longer, and twice as many players multi-table."""
    reg = replace(cfg.behaviours["regular"], base_error=0.02, difficulty_error=0.15, session_hours=(3.0, 9.0))
    counts = dict(cfg.counts)
    counts["multi_tabler"] *= 2
    counts["recreational"] -= cfg.counts["multi_tabler"]
    return replace(cfg, counts=counts, behaviours={**cfg.behaviours, "regular": reg, "colluder": reg})


SCENARIOS = {
    "adaptive_bots": adaptive_bots,
    "careful_rta": careful_rta,
    "legitimate_drift": legitimate_drift,
}
