from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bertani._rust", reason="the maturin extension has not been built")

from bertani_rules.agent import OPENING_BOOK, build_policy

from bertani import (
    MarketOp,
    TaskBatch,
    TaskKind,
    TaskScheduler,
    UnitOp,
    VecEnv,
    VectorRulePolicy,
    WorkforcePlan,
    WorkRole,
    WorkZone,
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


def test_scheduler_chooses_nearest_task_within_an_urgency_tier() -> None:
    env = VecEnv(1, weed_spawn_chance=0.0)
    batch = env.reset()
    tasks = TaskBatch.allocate(1, 2, env.board_size)
    mask = np.zeros((1, 2, env.board_size, env.board_size), dtype=np.bool_)
    mask[0, 0, 0, 0] = True
    mask[0, 0, 4, 4] = True
    tasks.propose_tiles(TaskKind.WATER, mask, 100.0)

    assignments = TaskScheduler(env.board_size).assign(batch, tasks)

    assert assignments.task_index[0, 0, 0] == 4 * env.board_size + 4


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


def test_post_opening_feed_workflow_uses_parallel_carriers() -> None:
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

    # The first dynamic turn hires the daily workforce. On the following turn,
    # several units should fetch wheat concurrently instead of making the main
    # farmer carry the entire feeding backlog.
    actions = policy.act(batch, max_orders=env.max_orders)
    batch = env.step(
        actions.unit_actions,
        actions.market_actions,
        actions.market_lengths,
    )
    actions = policy.act(batch, max_orders=env.max_orders)
    active_ops = actions.unit_actions[0, 0, :9, 0]
    assert np.count_nonzero(active_ops == UnitOp.PICKUP) >= 1

    observed_feed = False
    for _ in range(5):
        active_ops = actions.unit_actions[0, 0, :9, 0]
        observed_feed |= bool(np.any(active_ops == UnitOp.FEED))
        batch = env.step(
            actions.unit_actions,
            actions.market_actions,
            actions.market_lengths,
        )
        actions = policy.act(batch, max_orders=env.max_orders)

    assert observed_feed
    assert policy.last_tasks is not None
    assert policy.last_assignments is not None


def test_scheduler_finishes_nearly_as_urgent_unclaimed_work_underfoot() -> None:
    env = VecEnv(1, seed=10, weed_spawn_chance=0.0)
    batch = env.reset()
    tasks = TaskBatch.allocate(1, 2, env.board_size)
    scheduler = TaskScheduler(env.board_size)

    local = np.zeros((1, 2, env.board_size, env.board_size), dtype=np.bool_)
    remote = np.zeros_like(local)
    local[0, 0, 4, 4] = True
    remote[0, 0, 0, 0] = True
    tasks.propose_tiles(TaskKind.CARE, local, 85.0)
    tasks.propose_tiles(TaskKind.WATER, remote, 90.0)

    assignments = scheduler.assign(batch, tasks)
    assert assignments.task_index[0, 0, 0] == 4 * env.board_size + 4

    # More than five points of urgency still overrides work underfoot.
    tasks.clear()
    tasks.propose_tiles(TaskKind.CARE, local, 84.0)
    tasks.propose_tiles(TaskKind.WATER, remote, 90.0)
    assignments = scheduler.assign(batch, tasks)
    assert assignments.task_index[0, 0, 0] == 0


def test_scheduler_prefers_nearby_work_within_an_urgency_band() -> None:
    env = VecEnv(1, seed=10, weed_spawn_chance=0.0)
    batch = env.reset()
    tasks = TaskBatch.allocate(1, 2, env.board_size)
    scheduler = TaskScheduler(env.board_size)

    nearby = np.zeros((1, 2, env.board_size, env.board_size), dtype=np.bool_)
    remote = np.zeros_like(nearby)
    nearby[0, 0, 3, 4] = True
    remote[0, 0, 0, 0] = True
    tasks.propose_tiles(TaskKind.CARE, nearby, 100.0)
    tasks.propose_tiles(TaskKind.WATER, remote, 105.0)

    assignments = scheduler.assign(batch, tasks)
    assert assignments.task_index[0, 0, 0] == 3 * env.board_size + 4

    # A genuinely more urgent band still wins regardless of travel distance.
    tasks.clear()
    tasks.propose_tiles(TaskKind.CARE, nearby, 100.0)
    tasks.propose_tiles(TaskKind.FEED, remote, 120.0)
    assignments = scheduler.assign(batch, tasks)
    assert assignments.task_index[0, 0, 0] == 0


def test_scheduler_supports_soft_workforce_roles() -> None:
    env = VecEnv(1, seed=10, weed_spawn_chance=0.0)
    batch = env.reset()
    tasks = TaskBatch.allocate(1, 2, env.board_size)

    field = np.zeros((1, 2, env.board_size, env.board_size), dtype=np.bool_)
    logistics = np.zeros_like(field)
    field[0, 0, 3, 4] = True
    logistics[0, 0, 4, 3] = True
    tasks.propose_tiles(
        TaskKind.WATER,
        field,
        100.0,
        work_role=WorkRole.FIELD,
    )
    tasks.propose_tiles(
        TaskKind.CLEAR_WEED,
        logistics,
        100.0,
        work_role=WorkRole.LOGISTICS,
    )
    role = np.full(batch.active_units.shape, WorkRole.ANY, dtype=np.int16)
    zone = np.full(batch.active_units.shape, WorkZone.ANY, dtype=np.int16)
    role[0, 0, 0] = WorkRole.LOGISTICS

    assignments = TaskScheduler(env.board_size).assign(
        batch,
        tasks,
        WorkforcePlan(role=role, zone=zone, role_bonus=4.0),
    )
    assert assignments.task_index[0, 0, 0] == 4 * env.board_size + 3
