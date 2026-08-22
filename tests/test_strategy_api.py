from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bertani._rust", reason="the maturin extension has not been built")

from bertani import Item, RuleFeatures, RulePhase, VecEnv
from bertani_rules.strategy import RulePlan, build_python_policy


def test_python_strategy_fills_reused_native_policy_targets() -> None:
    calls = 0

    def strategy(features: RuleFeatures, plan: RulePlan) -> None:
        nonlocal calls
        calls += 1
        plan.target_hands[...] = np.where(features.day < 3, 3, 6)
        plan.wheat_reserve[...] = 5
        plan.crop(Item.WHEAT)[...] = 10
        plan.animal(Item.COW)[...] = 2

    environment = VecEnv(2, seed=11, weed_spawn_chance=0.0)
    batch = environment.reset()
    policy = build_python_policy(strategy, liquidation_days=1)

    policy.act(batch, max_orders=environment.max_orders)
    planner = policy.intent_planner
    first = planner.last_plan
    assert first is not None
    np.testing.assert_array_equal(first.target_hands, 3)
    np.testing.assert_array_equal(first.crop(Item.WHEAT), 10)
    np.testing.assert_array_equal(first.animal(Item.COW), 2)
    np.testing.assert_array_equal(first.phase, RulePhase.MIDGAME)

    policy.act(batch, max_orders=environment.max_orders)
    assert planner.last_plan is first
    assert calls == 2


def test_rule_plan_resets_values_not_written_on_the_next_turn() -> None:
    write_targets = True

    def strategy(features: RuleFeatures, plan: RulePlan) -> None:
        del features
        if write_targets:
            plan.target_hands.fill(7)
            plan.crop(Item.MELON).fill(9)

    environment = VecEnv(1, seed=12, weed_spawn_chance=0.0)
    batch = environment.reset()
    policy = build_python_policy(strategy)

    policy.act(batch, max_orders=environment.max_orders)
    planner = policy.intent_planner
    assert planner.last_plan is not None
    np.testing.assert_array_equal(planner.last_plan.target_hands, 7)

    write_targets = False
    policy.act(batch, max_orders=environment.max_orders)
    np.testing.assert_array_equal(planner.last_plan.target_hands, 0)
    np.testing.assert_array_equal(planner.last_plan.crop(Item.MELON), 0)


def test_rule_plan_rejects_wrong_item_families() -> None:
    plan = RulePlan.allocate((1, 2))
    with pytest.raises(ValueError, match="not a crop"):
        plan.crop(Item.COW)
    with pytest.raises(ValueError, match="not an animal"):
        plan.animal(Item.WHEAT)
