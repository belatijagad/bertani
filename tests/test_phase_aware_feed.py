from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bertani._rust", reason="the maturin extension has not been built")

from bertani import Item, TaskBatch, TaskKind, VecEnv
from bertani.market import MarketPlanBatch
from bertani.tasks import propose_native_maintenance_tasks
from bertani.vec_env import MarketOp


def _set_step(batch, step: int, episode_steps: int = 720) -> None:
    batch.observation_views.global_features[0, 0, 0] = step / float(episode_steps - 1)


def _set_unfed_animal_at_shed_access(batch, x: int = 4, y: int = 4) -> None:
    tile = batch.observation_views.tiles[0, 0, 0, y, x]
    tile.fill(0.0)
    tile[6] = 1.0
    tile[15] = 0.0
    tile[17] = 0.0


def _buy_wheat_plan(env: VecEnv) -> MarketPlanBatch:
    plan = MarketPlanBatch.allocate(1, 2, env.max_orders)
    seat = np.array([[True, False]], dtype=np.bool_)
    plan.append(seat, MarketOp.BUY_PRODUCT, item=Item.WHEAT, count=1)
    return plan


def _maintenance(env: VecEnv, batch, plan: MarketPlanBatch) -> TaskBatch:
    tasks = TaskBatch.allocate(1, 2, env.board_size)
    propose_native_maintenance_tasks(
        batch,
        tasks,
        market_plan=plan,
        seat_mask=np.array([[True, False]], dtype=np.bool_),
        turns_per_day=24,
        shed_capacity=100,
        episode_steps=720,
    )
    return tasks


def test_pending_wheat_buy_stages_in_future_fetch_slot() -> None:
    env = VecEnv(1, seed=17, weed_spawn_chance=0.0)
    batch = env.reset()
    _set_unfed_animal_at_shed_access(batch)
    _set_step(batch, 21)

    tasks = _maintenance(env, batch, _buy_wheat_plan(env))

    slot = tasks.tile_slots
    assert tasks.active[0, 0, slot]
    assert tasks.kind[0, 0, slot] == int(TaskKind.STAGE)
    assert tasks.item[0, 0, slot] == int(Item.WHEAT)
    assert tasks.quantity[0, 0, slot] == 1
    assert tasks.deadline[0, 0, slot] == 21


def test_pending_wheat_buy_is_not_staged_after_latest_feasible_hour() -> None:
    env = VecEnv(1, seed=18, weed_spawn_chance=0.0)
    batch = env.reset()
    _set_unfed_animal_at_shed_access(batch)
    _set_step(batch, 22)

    tasks = _maintenance(env, batch, _buy_wheat_plan(env))

    for extra_slot in (0, 3):
        slot = tasks.tile_slots + extra_slot
        assert not tasks.active[0, 0, slot]


def test_confirmed_wheat_releases_real_fetch_task() -> None:
    env = VecEnv(1, seed=19, weed_spawn_chance=0.0)
    batch = env.reset()

    plan = _buy_wheat_plan(env)
    unit_actions, market_actions, market_lengths = env.clear_actions()
    market_actions[...] = plan.actions
    market_lengths[...] = plan.lengths
    batch = env.step(unit_actions, market_actions, market_lengths)

    _set_unfed_animal_at_shed_access(batch)
    assert batch.observation_views.private[0, 0, int(Item.WHEAT)] > 0.0

    empty_plan = MarketPlanBatch.allocate(1, 2, env.max_orders)
    tasks = _maintenance(env, batch, empty_plan)

    slot = tasks.tile_slots
    assert tasks.active[0, 0, slot]
    assert tasks.kind[0, 0, slot] == int(TaskKind.FETCH_ITEM)
    assert tasks.item[0, 0, slot] == int(Item.WHEAT)
    assert tasks.quantity[0, 0, slot] == 1
