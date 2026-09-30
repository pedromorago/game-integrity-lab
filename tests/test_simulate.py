"""The generator and the features: each population has the signature the
docs describe, the data is seeded, and saving and loading keeps it intact."""

from functools import lru_cache

import numpy as np

from gil.config import small_config
from gil.data import load, save
from gil.features import pair_features, pair_segments, player_features
from gil.simulate import generate


@lru_cache(maxsize=1)
def data():
    ds = generate(small_config(), seed=3)
    return ds, player_features(ds)


def _median(name: str, segment: str) -> float:
    ds, ft = data()
    return float(np.median(ft.rows(ds.players.id[ds.players.segment == segment])[:, ft.names.index(name)]))


def test_population_counts():
    ds, _ = data()
    cfg = small_config()
    for seg, n in cfg.counts.items():
        assert (ds.players.segment == seg).sum() == n
    assert (ds.players.segment == "colluder").sum() == 2 * cfg.collusion.colluding_pairs


def test_bots_time_like_machines_and_play_long_sessions_at_all_hours():
    for human in ("recreational", "regular", "multi_tabler"):
        assert _median("time_cv", "bot") < _median("time_cv", human)
        assert _median("session_hours_max", "bot") > _median("session_hours_max", human)
        assert _median("hour_spread", "bot") > _median("hour_spread", human)
    assert abs(_median("time_difficulty_corr", "bot")) < 0.1


def test_regulars_slow_down_on_hard_spots():
    assert _median("time_difficulty_corr", "regular") > 0.3


def test_rta_users_are_accurate_and_slow_on_hard_spots():
    assert _median("ev_loss_hard", "rta") < _median("ev_loss_hard", "regular")
    assert _median("loss_gap", "rta") < _median("loss_gap", "regular")
    assert _median("time_hard_minus_easy", "rta") > _median("time_hard_minus_easy", "regular")


def test_colluding_pairs_sit_together_and_raise_each_other_less():
    ds, _ = data()
    pt = pair_features(ds)
    seg = pair_segments(ds, pt.ids)
    col = np.isin(seg, ("colluding", "chip_dumping"))
    other = seg == "other"
    assert col.sum() >= 0.8 * small_config().collusion.colluding_pairs
    assert np.median(pt.column("together_share")[col]) > np.median(pt.column("together_share")[other])
    assert np.median(pt.column("min_raise_ratio")[col]) < np.median(pt.column("min_raise_ratio")[other])


def test_same_seed_same_data_and_ids_do_not_encode_segment():
    a, b = generate(small_config(), 9), generate(small_config(), 9)
    assert a.fingerprint() == b.fingerprint()
    bots = np.sort(a.players.id[a.players.segment == "bot"])
    # ids are a random permutation: bots are not a contiguous block
    assert bots[-1] - bots[0] > 2 * len(bots)


def test_save_and_load_round_trip(tmp_path):
    ds, _ = data()
    save(ds, tmp_path)
    back = load(tmp_path)
    assert back.fingerprint() == ds.fingerprint()
    assert (tmp_path / "players.csv").read_text().splitlines()[0].startswith("id,pool,segment")
