from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bertani._rust", reason="the maturin extension has not been built")

from bertani_rules.agent import OPENING_BOOK, build_policy

from bertani import (
    Item,
    MarketOp,
    RuleConfig,
    RulePhase,
    UnitOp,
    VecEnv,
    VectorRulePolicy,
)


def test_rule_features_and_opening_intent_are_batched() -> None:
    env = VecEnv(4, seed=17, weed_spawn_chance=0.0)
    batch = env.reset()
    policy = build_policy(use_opening=False)

    features = policy.extract_features(batch)
    intent = policy.plan(batch)

    assert features.step.shape == (4, 2)
    assert features.crop_counts.shape == (4, 2, 5)
    assert features.animal_counts.shape == (4, 2, 3)
    assert features.market_price_ratios.shape == (4, 2, 9)
    np.testing.assert_array_equal(features.step, 0)
    np.testing.assert_array_equal(features.money, 3_000)
    np.testing.assert_array_equal(features.crop_counts, 0)
    np.testing.assert_array_equal(intent.phase, RulePhase.OPENING)
    np.testing.assert_array_equal(intent.target_hands, 5)
    np.testing.assert_array_equal(intent.target_crop_counts[..., 0], 7)
    np.testing.assert_array_equal(intent.target_crop_counts[..., 4], 12)


def test_masked_executor_waters_a_plant_on_the_current_tile() -> None:
    env = VecEnv(2, seed=31, max_market_orders=1, weed_spawn_chance=0.0)
    env.reset()
    market = np.zeros((2, 2, 1, 3), dtype=np.int64)
    lengths = np.zeros((2, 2), dtype=np.int64)
    market[:, 0, 0] = (MarketOp.BUY_SEED, Item.CARROT, 1)
    lengths[:, 0] = 1
    env.step(market_actions=market, market_lengths=lengths)

    units = np.zeros((2, 2, env.max_units, 3), dtype=np.int64)
    units[:, 0, 0] = (UnitOp.PLANT, Item.CARROT, 0)
    batch = env.step(unit_actions=units)

    policy = build_policy(use_opening=False)
    actions = policy.act(batch, max_orders=env.max_orders)

    np.testing.assert_array_equal(actions.unit_actions[:, 0, 0, 0], UnitOp.WATER)
    np.testing.assert_array_equal(
        actions.unit_actions[:, 1, 0, 0], UnitOp.BUILD_PASTURE
    )
    chosen = actions.unit_actions[..., 0]
    selected_masks = np.take_along_axis(
        batch.mask_views.unit_ops, chosen[..., None], axis=-1
    )[..., 0]
    assert selected_masks[batch.active_units].all()


def test_action_buffers_are_reused_and_liquidation_serializes_sales() -> None:
    env = VecEnv(1, seed=9, max_market_orders=3, weed_spawn_chance=0.0)
    env.reset()
    market = np.zeros((1, 2, 3, 3), dtype=np.int64)
    lengths = np.zeros((1, 2), dtype=np.int64)
    market[0, 0, 0] = (MarketOp.BUY_PRODUCT, Item.WHEAT, 3)
    lengths[0, 0] = 1
    batch = env.step(market_actions=market, market_lengths=lengths)

    policy = build_policy(use_opening=False, liquidation_days=30)
    first = policy.act(batch, max_orders=env.max_orders)
    second = policy.act(batch, max_orders=env.max_orders)

    assert first is second
    assert first.unit_actions.dtype == np.int64
    assert first.market_actions.shape == (1, 2, 3, 3)
    assert first.market_lengths[0, 0] == 3
    np.testing.assert_array_equal(
        first.market_actions[0, 0, 0], (MarketOp.SELL, Item.WHEAT, 3)
    )
    np.testing.assert_array_equal(
        first.market_actions[0, 0, 1:, 0], MarketOp.HIRE
    )
    assert first.market_lengths[0, 1] == 3
    np.testing.assert_array_equal(
        first.market_actions[0, 1, :, 0], MarketOp.HIRE
    )


def test_post_opening_economy_funds_feed_and_daily_labor() -> None:
    env = VecEnv(1, seed=11, weed_spawn_chance=0.0)
    policy = build_policy()
    batch = env.reset()
    for _ in OPENING_BOOK:
        actions = policy.act(batch, max_orders=env.max_orders)
        batch = env.step(
            actions.unit_actions,
            actions.market_actions,
            actions.market_lengths,
        )

    actions = policy.act(batch, max_orders=env.max_orders)
    market = actions.market_actions[0, 0, : actions.market_lengths[0, 0]]

    assert tuple(market[0]) == (MarketOp.SELL, Item.FERTILIZER, 4)
    assert tuple(market[1]) == (MarketOp.BUY_PRODUCT, Item.WHEAT, 2)
    assert sum(market[:, 0] == MarketOp.HIRE) == 5
    assert actions.unit_actions[0, 0, 0, 0] == UnitOp.PICKUP
    assert actions.unit_actions[0, 0, 0, 1] == Item.WHEAT


def test_dynamic_rules_complete_a_profitable_season() -> None:
    env = VecEnv(1, seed=100, auto_reset=False, weed_spawn_chance=0.0)
    policy = build_policy()
    batch = env.reset()

    for _ in range(719):
        actions = policy.act(batch, max_orders=env.max_orders)
        batch = env.step(
            actions.unit_actions,
            actions.market_actions,
            actions.market_lengths,
        )

    assert batch.dones.all()
    assert (batch.rewards > 20_000).all()
    for player in range(2):
        farm = env.state_snapshot(0)["farms"][player]
        animals = sum(
            int(isinstance(tile, dict) and "animal" in tile)
            for row in farm["tiles"]
            for tile in row
        )
        assert animals >= 12
