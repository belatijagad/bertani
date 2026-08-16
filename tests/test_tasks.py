from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bertani._rust", reason="the maturin extension has not been built")

from bertani_rules.v1 import OPENING_BOOK, build_policy

from bertani import (
    MarketOp,
    TaskBatch,
    TaskKind,
    TaskScheduler,
    UnitOp,
    VecEnv,
    VectorRulePolicy,
)


def test_tile_task_proposals_arbitrate_by_priority() -> None:
    tasks = TaskBatch.allocate(2, 2, board_size=5)
    mask = np.zeros((2, 2, 5, 5), dtype=np.bool_)
    mask[:, :, 3, 4] = True

    tasks.propose_tiles(TaskKind.CARE, mask, 70.0)
    tasks.propose_tiles(TaskKind.FEED, mask, 110.0, required_item=0)
    tasks.propose_tiles(TaskKind.WATER, mask, 90.0)

    slot = 3 * 5 + 4
    assert tasks.active[..., slot].all()
    np.testing.assert_array_equal(tasks.kind[..., slot], TaskKind.FEED)
    np.testing.assert_array_equal(tasks.priority[..., slot], 110.0)
    np.testing.assert_array_equal(tasks.required_item[..., slot], 0)


def test_scheduler_reserves_exclusive_tasks_for_distinct_units() -> None:
    env = VecEnv(1, max_market_orders=2, weed_spawn_chance=0.0)
    env.reset()
    market = np.zeros((1, 2, 2, 3), dtype=np.int64)
    market[0, 0, :, 0] = MarketOp.HIRE
    lengths = np.array([[2, 0]], dtype=np.int64)
    batch = env.step(market_actions=market, market_lengths=lengths)

    tasks = TaskBatch.allocate(1, 2, env.board_size)
    first = np.zeros((1, 2, env.board_size, env.board_size), dtype=np.bool_)
    second = first.copy()
    first[0, 0, 0, 0] = True
    second[0, 0, 4, 0] = True
    tasks.propose_tiles(TaskKind.WATER, first, 100.0)
    tasks.propose_tiles(TaskKind.HARVEST, second, 90.0)

    assignments = TaskScheduler(env.board_size).assign(batch, tasks)
    assigned = assignments.task_index[0, 0]
    assigned = assigned[assigned >= 0]
    assert len(assigned) == 2
    assert len(set(assigned.tolist())) == 2


def test_custom_rule_extends_policy_without_emitting_raw_actions() -> None:
    class BuildPastureRule:
        def propose(
            self, batch: object, intent: object, tasks: TaskBatch
        ) -> None:
            del batch, intent
            mask = np.zeros(
                (1, 2, tasks.board_size, tasks.board_size), dtype=np.bool_
            )
            mask[:, :, 4, 4] = True
            tasks.propose_tiles(TaskKind.BUILD_PASTURE, mask, 500.0)

    env = VecEnv(1, weed_spawn_chance=0.0)
    batch = env.reset()
    policy = VectorRulePolicy(
        task_rules=(BuildPastureRule(),),
    )

    actions = policy.act(batch, max_orders=env.max_orders)
    np.testing.assert_array_equal(
        actions.unit_actions[0, :, 0, 0], UnitOp.BUILD_PASTURE
    )


def test_post_opening_feed_workflow_fetches_routes_and_feeds() -> None:
    env = VecEnv(1, seed=100, weed_spawn_chance=0.0)
    policy = build_policy()
    batch = env.reset()
    for _ in OPENING_BOOK:
        actions = policy.act(batch, max_orders=env.max_orders)
        batch = env.step(
            actions.unit_actions,
            actions.market_actions,
            actions.market_lengths,
        )

    expected = [UnitOp.PICKUP, UnitOp.WEST, UnitOp.NORTH, UnitOp.FEED]
    observed: list[UnitOp] = []
    for _ in expected:
        actions = policy.act(batch, max_orders=env.max_orders)
        observed.append(UnitOp(actions.unit_actions[0, 0, 0, 0]))
        batch = env.step(
            actions.unit_actions,
            actions.market_actions,
            actions.market_lengths,
        )

    assert observed == expected
    assert policy.last_tasks is not None
    assert policy.last_assignments is not None
