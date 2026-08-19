from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bertani._rust", reason="the maturin extension has not been built")

from bertani.market import MarketPlanBatch
from bertani.rule_based import RulePhase, StrategicIntent
from bertani.tasks import TaskBatch, TaskKind, propose_native_production_tasks
from bertani.vec_env import Item, MarketOp, VecEnv


def _intent(*, cows: int = 0, sheep: int = 0) -> StrategicIntent:
    shape = (1, 2)
    target_animals = np.zeros((*shape, 3), dtype=np.int64)
    target_animals[0, 0, 1] = cows
    target_animals[0, 0, 2] = sheep
    return StrategicIntent(
        phase=np.full(shape, RulePhase.MIDGAME, dtype=np.int8),
        target_hands=np.zeros(shape, dtype=np.int64),
        cash_reserve=np.zeros(shape, dtype=np.float64),
        wheat_reserve=np.zeros(shape, dtype=np.int64),
        target_crop_counts=np.zeros((*shape, 5), dtype=np.int64),
        target_animal_counts=target_animals,
        liquidate=np.zeros(shape, dtype=np.bool_),
    )


def _set_step(batch, step: int, episode_steps: int = 720) -> None:
    batch.observation_views.global_features[0, 0, 0] = step / float(episode_steps - 1)


def _make_empty_pasture(batch, x: int = 4, y: int = 4) -> None:
    tile = batch.observation_views.tiles[0, 0, 0, y, x]
    tile.fill(0.0)
    tile[5] = 1.0


def _tasks(env, batch, intent, plan) -> TaskBatch:
    tasks = TaskBatch.allocate(1, 2, env.board_size)
    propose_native_production_tasks(
        batch,
        intent,
        tasks,
        market_plan=plan,
        seat_mask=np.array([[True, False]], dtype=np.bool_),
        turns_per_day=24,
        shed_capacity=100,
        episode_steps=720,
    )
    return tasks


def test_planned_cow_buy_stages_animal_fetch_when_pasture_exists() -> None:
    env = VecEnv(1, seed=31, weed_spawn_chance=0.0)
    batch = env.reset()
    _make_empty_pasture(batch)
    _set_step(batch, 10)

    plan = MarketPlanBatch.allocate(1, 2, env.max_orders)
    seat = np.array([[True, False]], dtype=np.bool_)
    plan.append(seat, MarketOp.BUY_ANIMAL, item=Item.COW, count=1)

    tasks = _tasks(env, batch, _intent(cows=1), plan)
    slot = tasks.tile_slots + 2

    assert tasks.active[0, 0, slot]
    assert tasks.kind[0, 0, slot] == int(TaskKind.STAGE)
    assert tasks.item[0, 0, slot] == int(Item.COW)
    assert tasks.quantity[0, 0, slot] == 1


def test_planned_cow_buy_does_not_stage_without_placement_capacity() -> None:
    env = VecEnv(1, seed=32, weed_spawn_chance=0.0)
    batch = env.reset()
    _set_step(batch, 10)

    plan = MarketPlanBatch.allocate(1, 2, env.max_orders)
    seat = np.array([[True, False]], dtype=np.bool_)
    plan.append(seat, MarketOp.BUY_ANIMAL, item=Item.COW, count=1)

    tasks = _tasks(env, batch, _intent(cows=1), plan)
    slot = tasks.tile_slots + 2

    assert not tasks.active[0, 0, slot]


def test_existing_cow_fetch_is_capped_by_empty_pasture_capacity() -> None:
    env = VecEnv(1, seed=33, weed_spawn_chance=0.0)
    batch = env.reset()
    _make_empty_pasture(batch)
    batch.observation_views.private[0, 0, int(Item.COW)] = 4.0 / 100.0

    plan = MarketPlanBatch.allocate(1, 2, env.max_orders)
    tasks = _tasks(env, batch, _intent(cows=4), plan)
    slot = tasks.tile_slots + 2

    assert tasks.active[0, 0, slot]
    assert tasks.kind[0, 0, slot] == int(TaskKind.FETCH_ITEM)
    assert tasks.item[0, 0, slot] == int(Item.COW)
    assert tasks.quantity[0, 0, slot] == 1


def test_pending_cow_stage_respects_place_deadline() -> None:
    env = VecEnv(1, seed=34, weed_spawn_chance=0.0)
    batch = env.reset()
    _make_empty_pasture(batch, 3, 3)
    _set_step(batch, 20)

    plan = MarketPlanBatch.allocate(1, 2, env.max_orders)
    seat = np.array([[True, False]], dtype=np.bool_)
    plan.append(seat, MarketOp.BUY_ANIMAL, item=Item.COW, count=1)

    tasks = _tasks(env, batch, _intent(cows=1), plan)
    slot = tasks.tile_slots + 2

    # latest stage = 23 - distance(2) - PICKUP/PLACE(2) = 19
    assert not tasks.active[0, 0, slot]
