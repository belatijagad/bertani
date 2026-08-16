from __future__ import annotations

import numpy as np

from bertani import (
    Item,
    MarketOp,
    MarketPlanBatch,
    RuleConfig,
    RulePhase,
    VecEnv,
    VectorRulePolicy,
)
from bertani_rules.v1 import V1IntentPlanner


def test_market_plan_preserves_order_and_resource_reservations() -> None:
    plan = MarketPlanBatch.allocate(2, 2, max_orders=3)
    selected = np.array([[True, False], [False, True]], dtype=np.bool_)

    plan.reserve_cash(selected, 400)
    plan.reserve_item(selected, Item.WHEAT, 4)
    plan.append(selected, MarketOp.SELL, item=Item.FERTILIZER, count=4)
    plan.append(selected, MarketOp.BUY_ANIMAL, item=Item.COW, count=1)
    plan.append(selected, MarketOp.HIRE)

    np.testing.assert_array_equal(plan.lengths, [[3, 0], [0, 3]])
    np.testing.assert_array_equal(
        plan.actions[0, 0],
        [
            (MarketOp.SELL, Item.FERTILIZER, 4),
            (MarketOp.BUY_ANIMAL, Item.COW, 1),
            (MarketOp.HIRE, 0, 0),
        ],
    )
    assert plan.reserved_cash[0, 0] == 400
    assert plan.reserved_items[1, 1, Item.WHEAT] == 4
    assert not plan.overflow.any()


def test_market_plan_reports_order_capacity_overflow() -> None:
    plan = MarketPlanBatch.allocate(1, 2, max_orders=1)
    selected = np.array([[True, False]], dtype=np.bool_)

    plan.append(selected, MarketOp.HIRE)
    plan.append(selected, MarketOp.BUY_LAND)

    assert plan.lengths[0, 0] == 1
    assert plan.overflow[0, 0]
    np.testing.assert_array_equal(
        plan.actions[0, 0, 0], (MarketOp.HIRE, 0, 0)
    )


def test_custom_market_rule_consumes_intent_without_raw_tensor_access() -> None:
    class OpeningHireRule:
        def propose(self, batch: object, intent: object, plan: MarketPlanBatch) -> None:
            del batch
            opening = intent.phase == RulePhase.OPENING
            plan.reserve_cash(opening, 100)
            plan.append(opening, MarketOp.HIRE)

    env = VecEnv(1, max_market_orders=2, weed_spawn_chance=0.0)
    batch = env.reset()
    config = RuleConfig()
    policy = VectorRulePolicy(
        config,
        intent_planner=V1IntentPlanner(config),
        task_rules=(),
        market_rules=(OpeningHireRule(),),
    )

    actions = policy.act(batch, max_orders=env.max_orders)

    np.testing.assert_array_equal(actions.market_lengths, 1)
    np.testing.assert_array_equal(actions.market_actions[..., 0, 0], MarketOp.HIRE)
    assert policy.last_market_plan is not None
    np.testing.assert_array_equal(policy.last_market_plan.reserved_cash, 100)
