"""Compare the native maintenance kernel with the pre-migration Python rule.

Run after rebuilding the extension:
    uv run python scripts/check_maintenance_parity.py --num-seeds 8
"""

from __future__ import annotations

import argparse

import numpy as np

from bertani.tasks import TaskBatch, TaskKind, WorkRole, propose_native_maintenance_tasks
from bertani.vec_env import Item, VecEnv
from bertani_rules.agent import build_policy


def propose_python_reference(
    batch,
    tasks: TaskBatch,
    *,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    episode_steps: int = 720,
) -> None:
    episode_days = max(1, (episode_steps + turns_per_day - 1) // turns_per_day)
    last_step = max(1, episode_steps - 1)
    tiles = batch.observation_views.tiles[:, :, 0]
    plants = tiles[..., 3] > 0.5
    animals = tiles[..., 6:9].sum(axis=-1) > 0.5
    watered_or_fed = tiles[..., 15] > 0.5
    cared = tiles[..., 16] > 0.5
    consecutive_missed = tiles[..., 17] * 2.0
    harvestable = tiles[..., 23] > 0.5
    crop_age = np.rint(tiles[..., 14] * episode_days).astype(np.int16)
    step = np.rint(
        batch.observation_views.global_features[..., 0] * last_step
    ).astype(np.int64)
    day = step // turns_per_day
    crop_channels = tiles[..., 9:14]

    one_time_ready = (
        (crop_channels[..., Item.WHEAT] > 0.5) & (crop_age >= 4)
    ) | ((crop_channels[..., Item.CARROT] > 0.5) & (crop_age >= 3)) | (
        (crop_channels[..., Item.MELON] > 0.5) & (crop_age >= 10)
    )
    ongoing = crop_channels[..., Item.TOMATO : Item.MELON].sum(axis=-1) > 0.5
    animal_product = animals & harvestable
    harvest_now = harvestable & (one_time_ready | ongoing | animal_product)
    fertilizer_available = tiles[..., 20] > 0.5

    tasks.propose_tiles(
        TaskKind.CARE,
        animals & ~cared,
        100.0 + 5.0 * consecutive_missed,
        deadline=turns_per_day - 1,
        work_role=WorkRole.LIVESTOCK,
    )
    tasks.propose_tiles(
        TaskKind.COLLECT_FERTILIZER,
        animals & fertilizer_available,
        95.0,
        estimated_value=100.0,
        work_role=WorkRole.LIVESTOCK,
    )

    wheat_bonus = (
        (crop_channels[..., Item.WHEAT] > 0.5)
        & (crop_age >= 2)
        & (crop_age <= 4)
    )
    carrot_bonus = (
        (crop_channels[..., Item.CARROT] > 0.5)
        & (crop_age >= 2)
        & (crop_age <= 3)
    )
    melon_bonus = (
        (crop_channels[..., Item.MELON] > 0.5)
        & (crop_age >= 6)
        & (crop_age <= 10)
    )
    tomato_production = (
        (crop_channels[..., Item.TOMATO] > 0.5)
        & (crop_age >= 8)
        & (crop_age <= 11)
    )
    strawberry_production = (
        (crop_channels[..., Item.STRAWBERRY] > 0.5)
        & (crop_age >= 10)
        & (crop_age <= 16)
        & (((crop_age - 10) % 2) == 0)
    )
    yield_water = (
        wheat_bonus
        | carrot_bonus
        | melon_bonus
        | tomato_production
        | strawberry_production
    )
    needs_water = plants & ~watered_or_fed & (
        (consecutive_missed >= 1.0) | yield_water
    )
    tasks.propose_tiles(
        TaskKind.WATER,
        needs_water,
        105.0 + 20.0 * consecutive_missed,
        deadline=turns_per_day - 1,
        work_role=WorkRole.FIELD,
    )

    fertilizer_days = tiles[..., 19] * 3.0
    ongoing_needing_fertilizer = ongoing & (fertilizer_days < 1.0)
    tasks.propose_tiles(
        TaskKind.FERTILIZE,
        ongoing_needing_fertilizer,
        90.0,
        required_item=Item.FERTILIZER,
        required_count=1,
        work_role=WorkRole.FIELD,
    )
    tasks.propose_tiles(
        TaskKind.HARVEST,
        harvest_now & plants,
        110.0,
        work_role=WorkRole.FIELD,
    )
    tasks.propose_tiles(
        TaskKind.HARVEST,
        harvest_now & animals,
        np.where((day == 6)[..., None, None], 145.0, 110.0),
        work_role=WorkRole.LIVESTOCK,
    )

    feed_priority = 120.0 + 30.0 * consecutive_missed
    needs_feed = animals & ~watered_or_fed
    tasks.propose_tiles(
        TaskKind.FEED,
        needs_feed,
        feed_priority,
        deadline=turns_per_day - 1,
        required_item=Item.WHEAT,
        required_count=1,
        work_role=WorkRole.LIVESTOCK,
    )

    views = batch.observation_views
    units = views.units[:, :, 0]

    carried_wheat = np.rint(
        units[..., 5 + int(Item.WHEAT)] * shed_capacity
    ).astype(np.int64)
    carried_wheat *= batch.active_units
    available_wheat = np.rint(
        views.private[..., int(Item.WHEAT)] * shed_capacity
    ).astype(np.int64)
    feed_count = needs_feed.sum(axis=(2, 3), dtype=np.int64)
    missing_wheat = np.maximum(0, feed_count - carried_wheat.sum(axis=-1))
    total_wheat_fetch = np.minimum(missing_wheat, available_wheat)
    access = max(0, tasks.board_size // 2 - 1)
    maximum_feed_priority = feed_priority.max(axis=(2, 3)) + 1.0
    quotient, remainder = np.divmod(total_wheat_fetch, 2)
    for index, slot in enumerate((0, 3)):
        quantity = quotient + (remainder > index)
        tasks.set_global(
            slot,
            quantity > 0,
            TaskKind.FETCH_ITEM,
            access,
            access,
            maximum_feed_priority,
            item=Item.WHEAT,
            quantity=quantity,
            deadline=turns_per_day - 1,
            work_role=WorkRole.LOGISTICS,
        )

    carried_fertilizer = np.rint(
        units[..., 5 + int(Item.FERTILIZER)] * shed_capacity
    ).astype(np.int64)
    carried_fertilizer *= batch.active_units
    available_fertilizer = np.rint(
        views.private[..., int(Item.FERTILIZER)] * shed_capacity
    ).astype(np.int64)
    missing_fertilizer = np.maximum(
        0,
        ongoing_needing_fertilizer.sum(axis=(2, 3), dtype=np.int64)
        - carried_fertilizer.sum(axis=-1),
    )
    total_fertilizer_fetch = np.minimum(
        missing_fertilizer, available_fertilizer
    )
    tasks.set_global(
        6,
        total_fertilizer_fetch > 0,
        TaskKind.FETCH_ITEM,
        access,
        access,
        86.0,
        item=Item.FERTILIZER,
        quantity=total_fertilizer_fetch,
        work_role=WorkRole.LOGISTICS,
    )


def compare_tasks(reference: TaskBatch, native: TaskBatch, turn: int) -> None:
    exact_fields = (
        "active",
        "kind",
        "target_x",
        "target_y",
        "item",
        "quantity",
        "deadline",
        "required_item",
        "required_count",
        "exclusive",
        "work_role",
    )
    float_fields = ("priority", "estimated_value")
    for name in exact_fields:
        left = getattr(reference, name)
        right = getattr(native, name)
        if not np.array_equal(left, right):
            mismatch = np.argwhere(left != right)[0]
            idx = tuple(int(value) for value in mismatch)
            raise AssertionError(
                f"turn={turn} field={name} idx={idx}: "
                f"python={left[idx]!r}, native={right[idx]!r}"
            )
    for name in float_fields:
        left = getattr(reference, name)
        right = getattr(native, name)
        if not np.allclose(left, right, rtol=0.0, atol=1e-6, equal_nan=True):
            mismatch = np.argwhere(~np.isclose(left, right, rtol=0.0, atol=1e-6, equal_nan=True))[0]
            idx = tuple(int(value) for value in mismatch)
            raise AssertionError(
                f"turn={turn} field={name} idx={idx}: "
                f"python={left[idx]!r}, native={right[idx]!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--seed-source", type=int, default=2026)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed_source)
    seeds = rng.integers(0, np.iinfo(np.uint64).max, size=args.num_seeds, dtype=np.uint64)
    env = VecEnv(args.num_seeds, seed=args.seed_source, auto_reset=False, weed_spawn_chance=0.0)
    batch = env.reset(seeds)
    policy = build_policy()
    reference = TaskBatch.allocate(args.num_seeds, 2, env.board_size)
    native = TaskBatch.allocate(args.num_seeds, 2, env.board_size)

    for turn in range(719):
        reference.clear()
        native.clear()
        propose_python_reference(batch, reference)
        propose_native_maintenance_tasks(
            batch,
            native,
            turns_per_day=24,
            shed_capacity=100,
            episode_steps=720,
        )
        compare_tasks(reference, native, turn)
        actions = policy.act(batch, max_orders=env.max_orders)
        batch = env.step(
            actions.unit_actions,
            actions.market_actions,
            actions.market_lengths,
        )

    print(
        f"maintenance parity passed: {args.num_seeds} environments x 719 turns"
    )


if __name__ == "__main__":
    main()
