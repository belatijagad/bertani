"""Turn-by-turn parity checks for the V12 native farm pipeline.

Checks two independent changes:
1. fused native farm-task generation == maintenance then production;
2. native assign+execute == native assign followed by the V11 Python executor.

The Python executor below is test-only and intentionally mirrors the V11 code.
"""

from __future__ import annotations

import argparse

import numpy as np

from bertani.market import MarketPlanBatch
from bertani.tasks import TaskBatch, TaskKind, TaskScheduler, WorkRole
from bertani.vec_env import UnitOp, VecEnv
from bertani_rules.agent import (
    EconomyMarketRule,
    FarmTaskRule,
    MaintenanceTaskRule,
    ProductionTaskRule,
    build_policy,
)


OPERATIONS = {
    TaskKind.WATER: UnitOp.WATER,
    TaskKind.FEED: UnitOp.FEED,
    TaskKind.CARE: UnitOp.CARE,
    TaskKind.HARVEST: UnitOp.HARVEST,
    TaskKind.COLLECT_FERTILIZER: UnitOp.COLLECT_FERTILIZER,
    TaskKind.CLEAR_WEED: UnitOp.DIG,
    TaskKind.PLANT: UnitOp.PLANT,
    TaskKind.FERTILIZE: UnitOp.FERTILIZE,
    TaskKind.BUILD_COOP: UnitOp.BUILD_COOP,
    TaskKind.BUILD_PASTURE: UnitOp.BUILD_PASTURE,
    TaskKind.PLACE_ANIMAL: UnitOp.PLACE,
    TaskKind.FETCH_ITEM: UnitOp.PICKUP,
    TaskKind.DEPOSIT_INVENTORY: UnitOp.DROP,
}
ARGUMENT_OPERATIONS = {UnitOp.PICKUP, UnitOp.PLACE, UnitOp.PLANT}


def active_unit_limit(active_units: np.ndarray) -> int:
    active_slots = np.flatnonzero(np.any(active_units, axis=(0, 1)))
    return int(active_slots[-1]) + 1 if active_slots.size else 0


def python_execute(batch, tasks, assignments, unit_actions, board_size: int) -> None:
    unit_actions.fill(0)
    active_limit = active_unit_limit(batch.active_units)
    if active_limit == 0:
        return

    units = batch.observation_views.units[:, :, 0, :active_limit]
    scale = max(1, board_size - 1)
    unit_x = np.rint(units[..., 2] * scale).astype(np.int16)
    unit_y = np.rint(units[..., 3] * scale).astype(np.int16)
    task_index = assignments.task_index[..., :active_limit]
    assigned = task_index >= 0
    safe_task = np.maximum(task_index, 0)

    def task_field(values):
        return np.take_along_axis(values, safe_task, axis=2)

    target_x = task_field(tasks.target_x).copy()
    target_y = task_field(tasks.target_y).copy()
    kind = task_field(tasks.kind)
    item = task_field(tasks.item)
    count = task_field(tasks.quantity)

    deposit = assigned & (kind == TaskKind.DEPOSIT_INVENTORY)
    half = board_size // 2
    low_center, high_center = max(0, half - 1), half
    at_shed = deposit & np.isin(unit_x, (low_center, high_center)) & np.isin(
        unit_y, (low_center, high_center)
    )
    target_x[deposit] = np.where(
        unit_x[deposit] <= low_center, low_center, high_center
    )
    target_y[deposit] = np.where(
        unit_y[deposit] <= low_center, low_center, high_center
    )

    moving = assigned & ~at_shed & ((unit_x != target_x) | (unit_y != target_y))
    movement = np.where(
        unit_x < target_x,
        UnitOp.EAST,
        np.where(
            unit_x > target_x,
            UnitOp.WEST,
            np.where(unit_y < target_y, UnitOp.SOUTH, UnitOp.NORTH),
        ),
    ).astype(np.int64)

    operation_lookup = np.full(max(TaskKind) + 1, UnitOp.PASS, dtype=np.int64)
    for task_kind, operation in OPERATIONS.items():
        operation_lookup[task_kind] = operation
    operation = operation_lookup[kind]
    operation = np.where(moving, movement, operation)
    operation = np.where(at_shed, UnitOp.DROP, operation)

    grid = np.indices(assigned.shape)
    legal = assigned & batch.mask_views.unit_ops[
        grid[0], grid[1], grid[2], operation
    ]
    interaction = assigned & ~moving & ~at_shed
    needs_argument = interaction & np.isin(operation, tuple(ARGUMENT_OPERATIONS))
    safe_item = np.maximum(item, 0)
    argument_legal = batch.mask_views.unit_args[
        grid[0], grid[1], grid[2], operation, safe_item
    ]
    legal &= ~needs_argument | ((item >= 0) & argument_legal)

    active_actions = unit_actions[..., :active_limit, :]
    active_actions[..., 0][legal] = operation[legal]
    write_arguments = legal & interaction
    active_actions[..., 1][write_arguments] = safe_item[write_arguments]
    active_actions[..., 2][write_arguments] = count[write_arguments]


def compare_tasks(turn: int, left: TaskBatch, right: TaskBatch) -> None:
    fields = (
        "active",
        "kind",
        "target_x",
        "target_y",
        "item",
        "quantity",
        "priority",
        "deadline",
        "estimated_value",
        "required_item",
        "required_count",
        "exclusive",
        "work_role",
    )
    for field in fields:
        a = getattr(left, field)
        b = getattr(right, field)
        if not np.array_equal(a, b, equal_nan=True):
            mismatch = np.argwhere(a != b)
            index = tuple(mismatch[0]) if mismatch.size else ()
            raise AssertionError(
                f"turn {turn} task field {field} mismatch at {index}: "
                f"sequential={a[index] if index else 'n/a'} "
                f"fused={b[index] if index else 'n/a'}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--seed-source", type=int, default=2026)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed_source)
    seeds = rng.integers(0, 2**63, size=args.num_seeds, dtype=np.uint64)
    env = VecEnv(args.num_seeds, seed=0, auto_reset=False)
    batch = env.reset(seeds)
    config = build_policy().config
    driver = build_policy(config)

    sequential = TaskBatch.allocate(args.num_seeds, 2, env.board_size)
    fused = TaskBatch.allocate(args.num_seeds, 2, env.board_size)
    masked = TaskBatch.allocate(args.num_seeds, 2, env.board_size)
    seat_mask = np.zeros((args.num_seeds, 2), dtype=np.bool_)
    seat_mask[np.arange(args.num_seeds), np.arange(args.num_seeds) % 2] = True
    maintenance = MaintenanceTaskRule(
        turns_per_day=config.turns_per_day,
        shed_capacity=config.shed_capacity,
        episode_steps=config.episode_steps,
    )
    production = ProductionTaskRule(
        turns_per_day=config.turns_per_day,
        shed_capacity=config.shed_capacity,
        episode_steps=config.episode_steps,
    )
    farm = FarmTaskRule(
        turns_per_day=config.turns_per_day,
        shed_capacity=config.shed_capacity,
        episode_steps=config.episode_steps,
    )
    market_rule = EconomyMarketRule(
        starting_money=config.starting_money,
        shed_capacity=config.shed_capacity,
        episode_steps=config.episode_steps,
        turns_per_day=config.turns_per_day,
    )
    market_full = MarketPlanBatch.allocate(args.num_seeds, 2, env.max_orders)
    market_masked = MarketPlanBatch.allocate(args.num_seeds, 2, env.max_orders)

    scheduler_split = TaskScheduler(
        env.board_size,
        shed_capacity=config.shed_capacity,
        episode_steps=config.episode_steps,
        turns_per_day=config.turns_per_day,
    )
    scheduler_fused = TaskScheduler(
        env.board_size,
        shed_capacity=config.shed_capacity,
        episode_steps=config.episode_steps,
        turns_per_day=config.turns_per_day,
    )
    actions_python = np.zeros(
        (args.num_seeds, 2, env.max_units, 3), dtype=np.int64
    )
    actions_native = np.zeros_like(actions_python)

    for turn in range(config.episode_steps - 1):
        features = driver.extract_features(batch)
        planner_from_features = getattr(driver.intent_planner, "from_features", None)
        intent = (
            planner_from_features(batch, features)
            if planner_from_features is not None
            else driver.plan(batch)
        )

        sequential.clear()
        maintenance.propose(batch, intent, sequential)
        production.propose(batch, intent, sequential)
        fused.clear()
        farm.propose(batch, intent, fused)
        compare_tasks(turn, sequential, fused)

        masked.clear()
        farm.propose_masked(batch, intent, masked, seat_mask)
        for field in (
            "active",
            "kind",
            "target_x",
            "target_y",
            "item",
            "quantity",
            "priority",
            "deadline",
            "estimated_value",
            "required_item",
            "required_count",
            "exclusive",
            "work_role",
        ):
            np.testing.assert_array_equal(
                getattr(masked, field)[seat_mask],
                getattr(sequential, field)[seat_mask],
                err_msg=f"turn {turn} masked task field {field}",
            )
        if np.any(masked.active[~seat_mask]):
            raise AssertionError(f"turn {turn} uncontrolled seat received tasks")

        market_full.clear()
        market_masked.clear()
        market_rule.propose(batch, intent, market_full)
        market_rule.propose_masked(batch, intent, market_masked, seat_mask)
        np.testing.assert_array_equal(
            market_masked.actions[seat_mask],
            market_full.actions[seat_mask],
            err_msg=f"turn {turn} masked market actions",
        )
        np.testing.assert_array_equal(
            market_masked.lengths[seat_mask],
            market_full.lengths[seat_mask],
            err_msg=f"turn {turn} masked market lengths",
        )
        if np.any(market_masked.lengths[~seat_mask] != 0):
            raise AssertionError(f"turn {turn} uncontrolled seat received market orders")

        workforce = (
            driver.workforce_planner(batch, intent, sequential)
            if driver.workforce_planner is not None
            else None
        )
        assignment_split = scheduler_split.assign(batch, sequential, workforce)
        python_execute(
            batch,
            sequential,
            assignment_split,
            actions_python,
            env.board_size,
        )
        assignment_fused = scheduler_fused.assign_and_execute(
            batch,
            fused,
            actions_native,
            workforce,
        )

        if not np.array_equal(
            assignment_split.task_index, assignment_fused.task_index
        ):
            idx = tuple(
                np.argwhere(
                    assignment_split.task_index != assignment_fused.task_index
                )[0]
            )
            raise AssertionError(
                f"turn {turn} assignment mismatch at {idx}: "
                f"split={assignment_split.task_index[idx]} "
                f"fused={assignment_fused.task_index[idx]}"
            )
        np.testing.assert_array_equal(
            assignment_split.score,
            assignment_fused.score,
            err_msg=f"turn {turn} assignment score mismatch",
        )
        if not np.array_equal(actions_python, actions_native):
            idx = tuple(np.argwhere(actions_python != actions_native)[0])
            raise AssertionError(
                f"turn {turn} unit action mismatch at {idx}: "
                f"python={actions_python[idx]} native={actions_native[idx]}"
            )

        driver_actions = driver.act(batch, max_orders=env.max_orders)
        batch = env.step(
            driver_actions.unit_actions,
            driver_actions.market_actions,
            driver_actions.market_lengths,
        )

    assert scheduler_split.full_solves == scheduler_fused.full_solves
    assert scheduler_split.cache_hits == scheduler_fused.cache_hits
    assert scheduler_split.cache_miss_reasons == scheduler_fused.cache_miss_reasons
    assert scheduler_split.force_replan_reasons == scheduler_fused.force_replan_reasons
    print(
        "V12 parity passed: "
        f"{args.num_seeds} environments x {config.episode_steps - 1} turns"
    )


if __name__ == "__main__":
    main()
