from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("bertani._rust", reason="the maturin extension has not been built")

from bertani import VecEnv
from bertani.v16_native import NativeV16Policy, load_v16_actions


def test_native_v16_matches_python_baseline_against_pass() -> None:
    baseline = Path(__file__).parents[1] / "baselines" / "v16_rc5" / "main.py"
    environment = VecEnv(2, auto_reset=False)
    batch = environment.reset(np.asarray((11, 11), dtype=np.uint64))
    policy = NativeV16Policy(load_v16_actions(baseline))
    unit_actions, market_actions, market_lengths = environment.clear_actions()
    games = np.arange(2)
    seats = np.asarray((0, 1))

    for _ in range(719):
        actions = policy.act(batch)
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
