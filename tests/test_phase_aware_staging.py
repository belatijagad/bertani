from __future__ import annotations

import numpy as np

from bertani.market import MarketPlanBatch
from bertani.rule_based import RulePhase, StrategicIntent
from bertani.tasks import TaskBatch, TaskKind, propose_native_farm_tasks
from bertani.vec_env import Item, MarketOp, VecEnv


def _intent() -> StrategicIntent:
    shape = (1, 2)
    target_crop_counts = np.zeros((*shape, 5), dtype=np.int64)
    target_crop_counts[0, 0, Item.WHEAT] = 1
    return StrategicIntent(
        phase=np.full(shape, RulePhase.MIDGAME, dtype=np.int8),
        target_hands=np.zeros(shape, dtype=np.int64),
        cash_reserve=np.zeros(shape, dtype=np.float64),
        wheat_reserve=np.zeros(shape, dtype=np.int64),
        target_crop_counts=target_crop_counts,
        target_animal_counts=np.zeros((*shape, 3), dtype=np.int64),
        liquidate=np.zeros(shape, dtype=np.bool_),
    )


def test_planned_seed_buy_stages_then_releases_plant_next_turn() -> None:
    env = VecEnv(1, weed_spawn_chance=0.0)
    batch = env.reset(np.array([123], dtype=np.uint64))
    intent = _intent()
    seat_mask = np.array([[True, False]], dtype=np.bool_)

    plan = MarketPlanBatch.allocate(1, 2, env.max_orders)
    plan.append(seat_mask, MarketOp.BUY_SEED, item=Item.WHEAT, count=1)

    tasks = TaskBatch.allocate(1, 2, env.board_size)
    propose_native_farm_tasks(
        batch, intent, tasks, market_plan=plan, seat_mask=seat_mask,
        turns_per_day=24, shed_capacity=100, episode_steps=720,
    )

    active = tasks.active[0, 0]
    kinds = tasks.kind[0, 0]
    items = tasks.item[0, 0]
    staged = active & (kinds == int(TaskKind.STAGE)) & (items == int(Item.WHEAT))
    planted = active & (kinds == int(TaskKind.PLANT))
    assert np.count_nonzero(staged) == 1
    assert np.count_nonzero(planted) == 0

    unit_actions, market_actions, market_lengths = env.clear_actions()
    market_actions[...] = plan.actions
    market_lengths[...] = plan.lengths
    batch = env.step(unit_actions, market_actions, market_lengths)

    plan.clear()
    tasks.clear()
    propose_native_farm_tasks(
        batch, intent, tasks, market_plan=plan, seat_mask=seat_mask,
        turns_per_day=24, shed_capacity=100, episode_steps=720,
    )

    active = tasks.active[0, 0]
    kinds = tasks.kind[0, 0]
    staged = active & (kinds == int(TaskKind.STAGE))
    planted = active & (kinds == int(TaskKind.PLANT))
    assert np.count_nonzero(staged) == 0
    assert np.count_nonzero(planted) == 1
