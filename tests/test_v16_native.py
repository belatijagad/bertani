from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("bertani._rust", reason="the maturin extension has not been built")

from bertani import SelfPlayEnv, VecEnv
from bertani.v16 import V16OpponentPolicy


def test_native_v16_matches_python_baseline_against_pass() -> None:
    baseline = Path(__file__).parents[1] / "baselines" / "v16_rc5" / "main.py"
    environment = VecEnv(2, auto_reset=False)
    batch = environment.reset(np.asarray((11, 11), dtype=np.uint64))
    policy = V16OpponentPolicy.from_path(
        baseline, max_orders=environment.max_orders
    )
    unit_actions, market_actions, market_lengths = environment.clear_actions()
    games = np.arange(2)
    seats = np.asarray((0, 1))

    for _ in range(719):
        actions = policy.act(environment, batch, seats=seats)
        assert actions.unit_actions.dtype == np.int16
        assert actions.market_actions.dtype == np.int16
        assert actions.market_lengths.dtype == np.int16
        unit_actions.fill(0)
        market_actions.fill(0)
        market_lengths.fill(0)
        unit_actions[games, seats] = actions.unit_actions[games, seats]
        market_actions[games, seats] = actions.market_actions[games, seats]
        market_lengths[games, seats] = actions.market_lengths[games, seats]
        batch = environment.step(
            unit_actions, market_actions, market_lengths
        )

    np.testing.assert_array_equal(
        batch.rewards,
        ((190_184.0, 3_000.0), (3_000.0, 146_354.0)),
    )


def test_v16_opponent_supports_auto_reset_self_play() -> None:
    baseline = Path(__file__).parents[1] / "baselines" / "v16_rc5" / "main.py"
    environment = VecEnv(4, auto_reset=True, episode_steps=3)
    opponent = V16OpponentPolicy.from_path(
        baseline,
        episode_steps=3,
        max_orders=environment.max_orders,
    )
    self_play = SelfPlayEnv(environment, opponent)
    batch = self_play.reset(np.asarray((21, 22, 23, 24), dtype=np.uint64))
    learner = np.zeros((4, environment.max_units, 3), dtype=np.int64)

    for _ in range(3):
        batch = self_play.step(learner)

    assert (batch.episode_ids == 1).all()
    assert self_play.last_step_profile.opponent_seconds > 0.0
    assert opponent.cache_stats.hits == opponent.cache_stats.misses == 0
