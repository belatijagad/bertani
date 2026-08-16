from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bertani._rust", reason="the maturin extension has not been built")

from bertani import Item, VecEnv
from bertani_rules.agent import build_policy


def test_intent_scales_labor_animals_and_selects_cash_crop() -> None:
    env = VecEnv(1, weed_spawn_chance=0.0)
    batch = env.reset()
    policy = build_policy(use_opening=False)

    # Make carrot clearly superior to every other cash crop.
    batch.observation_views.global_features[..., 0] = 72 / 719
    ratios = batch.observation_views.global_features[..., 5:22:2]
    ratios[..., Item.CARROT] = 2.0
    ratios[..., Item.TOMATO] = 0.1
    ratios[..., Item.STRAWBERRY] = 0.1
    ratios[..., Item.MELON] = 0.1
    intent = policy.plan(batch)

    np.testing.assert_array_equal(intent.target_hands, 8)
    np.testing.assert_array_equal(intent.target_animal_counts[..., 1], 8)
    np.testing.assert_array_equal(intent.target_animal_counts[..., 2], 4)
    np.testing.assert_array_equal(intent.target_crop_counts[..., Item.WHEAT], 15)
    np.testing.assert_array_equal(intent.target_crop_counts[..., Item.CARROT], 9)


def test_agent_expands_and_keeps_twelve_animals_alive() -> None:
    env = VecEnv(1, seed=100, auto_reset=False, weed_spawn_chance=0.0)
    batch = env.reset()
    policy = build_policy()

    for _ in range(719):
        actions = policy.act(batch, max_orders=env.max_orders)
        batch = env.step(
            actions.unit_actions,
            actions.market_actions,
            actions.market_lengths,
        )

    farm = env.state_snapshot(0)["farms"][0]
    animals = [
        tile["animal"]
        for row in farm["tiles"]
        for tile in row
        if isinstance(tile, dict) and "animal" in tile
    ]
    assert farm["unlocked_quadrants"] == [0, 1]
    assert animals.count(int(Item.COW) - int(Item.GOOSE)) == 8
    assert animals.count(int(Item.SHEEP) - int(Item.GOOSE)) == 4
    assert batch.rewards[0, 0] > 20_000
