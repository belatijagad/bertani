from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bertani._rust", reason="the maturin extension has not been built")

from bertani import Item, MarketOp, VecEnv
from bertani.tasks import TaskKind
from bertani_rules.agent import build_policy


def test_intent_uses_stable_cohort_and_yarn_store_branch() -> None:
    env = VecEnv(1, weed_spawn_chance=0.0)
    batch = env.reset()
    policy = build_policy(use_opening=False)

    batch.observation_views.global_features[..., 0] = (14 * 24) / 719
    ratios = batch.observation_views.global_features[..., 5:22:2]
    ratios[..., Item.CARROT] = 2.0
    ratios[..., Item.TOMATO] = 0.1
    ratios[..., Item.STRAWBERRY] = 0.1
    ratios[..., Item.MELON] = 0.1
    intent = policy.plan(batch)

    np.testing.assert_array_equal(intent.target_hands, 12)
    np.testing.assert_array_equal(intent.target_animal_counts[..., 1], 10)
    np.testing.assert_array_equal(intent.target_animal_counts[..., 2], 4)
    np.testing.assert_array_equal(intent.target_crop_counts[..., Item.WHEAT], 19)
    np.testing.assert_array_equal(intent.target_crop_counts[..., Item.STRAWBERRY], 42)
    np.testing.assert_array_equal(intent.target_crop_counts[..., Item.CARROT], 0)

    batch.observation_views.global_features[..., 22 + 7] = 2 / 8
    intent = policy.plan(batch)
    np.testing.assert_array_equal(intent.target_crop_counts[..., Item.WHEAT], 26)
    np.testing.assert_array_equal(intent.target_animal_counts[..., 1], 6)
    np.testing.assert_array_equal(intent.target_animal_counts[..., 2], 12)


def test_intent_does_not_plant_a_crop_that_cannot_mature() -> None:
    env = VecEnv(1, weed_spawn_chance=0.0)
    batch = env.reset()
    policy = build_policy(use_opening=False)
    batch.observation_views.global_features[..., 0] = (28 * 24) / 719

    intent = policy.plan(batch)

    np.testing.assert_array_equal(intent.target_crop_counts, 0)


def test_two_yarn_stores_trigger_fourth_land_purchase() -> None:
    env = VecEnv(1, weed_spawn_chance=0.0)
    batch = env.reset()
    policy = build_policy(use_opening=False)
    views = batch.observation_views
    views.global_features[..., 0] = (12 * 24) / 719
    views.global_features[..., 22 + 7] = 2 / 8
    views.farms[..., 0] = 2.0
    views.farms[..., 4:8] = (1.0, 1.0, 1.0, 0.0)

    actions = policy.act(batch, max_orders=env.max_orders)
    market = actions.market_actions[..., 0, 0]

    np.testing.assert_array_equal(market, MarketOp.BUY_LAND)


def test_intent_builds_strawberry_cohort_and_reinvests_surplus_cash() -> None:
    env = VecEnv(1, weed_spawn_chance=0.0)
    batch = env.reset()
    policy = build_policy(use_opening=False)

    batch.observation_views.global_features[..., 0] = (8 * 24) / 719
    intent = policy.plan(batch)
    np.testing.assert_array_equal(
        intent.target_crop_counts[..., Item.WHEAT], 7
    )
    np.testing.assert_array_equal(
        intent.target_crop_counts[..., Item.STRAWBERRY], 19
    )
    np.testing.assert_array_equal(
        intent.target_crop_counts[..., Item.MELON], 12
    )

    batch.observation_views.global_features[..., 0] = (14 * 24) / 719
    batch.observation_views.farms[..., 0] = 2.0
    intent = policy.plan(batch)
    np.testing.assert_array_equal(intent.target_hands, 12)


def test_agent_expands_and_keeps_twelve_animals_alive() -> None:
    env = VecEnv(1, seed=100, auto_reset=False, weed_spawn_chance=0.0)
    batch = env.reset()
    policy = build_policy()
    maximum_third_field_utilization = 0
    second_quadrant_unlock_step = None
    day_eight_strawberries = 0
    day_eight_new_strawberry_positions = []
    day_eight_assignment_kinds = []
    day_seven_animals = (0, 0)

    for turn in range(719):
        actions = policy.act(batch, max_orders=env.max_orders)
        if turn == 8 * 24 + 1:
            assert policy.last_assignments is not None
            assert policy.last_tasks is not None
            assigned = policy.last_assignments.task_index[0, 0]
            day_eight_assignment_kinds = [
                TaskKind(int(policy.last_tasks.kind[0, 0, task]))
                for task in assigned
                if task >= 0
            ]
        batch = env.step(
            actions.unit_actions,
            actions.market_actions,
            actions.market_lengths,
        )
        snapshot = env.state_snapshot(0)["farms"][0]
        if (
            second_quadrant_unlock_step is None
            and len(snapshot["unlocked_quadrants"]) >= 2
        ):
            second_quadrant_unlock_step = turn + 1
        if turn + 1 == 9 * 24:
            day_eight_strawberries = sum(
                tile.get("kind") == "PLANT"
                and tile.get("crop") == int(Item.STRAWBERRY)
                for row in snapshot["tiles"]
                for tile in row
            )
            day_eight_new_strawberry_positions = [
                (x, y)
                for y, row in enumerate(snapshot["tiles"])
                for x, tile in enumerate(row)
                if tile.get("kind") == "PLANT"
                and tile.get("crop") == int(Item.STRAWBERRY)
                and tile.get("planted_day") == 7
            ]
        if turn + 1 == 7 * 24:
            day_seven_animals = (
                sum(
                    tile.get("animal") == int(Item.COW) - int(Item.GOOSE)
                    for row in snapshot["tiles"]
                    for tile in row
                ),
                sum(
                    tile.get("animal") == int(Item.SHEEP) - int(Item.GOOSE)
                    for row in snapshot["tiles"]
                    for tile in row
                ),
            )
        if (turn + 1) % 24 == 0:
            if len(snapshot["unlocked_quadrants"]) >= 3:
                occupied = sum(
                    tile.get("kind") not in {"EMPTY", "LOCKED", "WEED"}
                    for row in snapshot["tiles"]
                    for tile in row
                )
                maximum_third_field_utilization = max(
                    maximum_third_field_utilization, occupied
                )

    farm = env.state_snapshot(0)["farms"][0]
    animal_tiles = [
        (x, y, tile["animal"])
        for y, row in enumerate(farm["tiles"])
        for x, tile in enumerate(row)
        if isinstance(tile, dict) and "animal" in tile
    ]
    animals = [animal for _, _, animal in animal_tiles]
    assert farm["unlocked_quadrants"][:3] == [0, 1, 2]
    center_access = ((4, 4), (5, 4), (4, 5), (5, 5))
    assert second_quadrant_unlock_step is not None
    assert second_quadrant_unlock_step <= 7 * 24 - 1
    assert day_seven_animals == (4, 2)
    assert day_eight_strawberries >= 10
    assert len(day_eight_new_strawberry_positions) >= 4
    assert all(
        min(abs(x - cx) + abs(y - cy) for cx, cy in center_access) <= 3
        for x, y in day_eight_new_strawberry_positions
    )
    assert day_eight_assignment_kinds.count(TaskKind.PLANT) >= 3
    assert day_eight_assignment_kinds.count(TaskKind.FETCH_ITEM) <= 3
    if len(farm["unlocked_quadrants"]) == 4:
        assert animals.count(int(Item.COW) - int(Item.GOOSE)) >= 6
        assert animals.count(int(Item.SHEEP) - int(Item.GOOSE)) >= 8
    else:
        assert animals.count(int(Item.COW) - int(Item.GOOSE)) >= 8
        assert animals.count(int(Item.SHEEP) - int(Item.GOOSE)) == 4
    assert all(
        min(abs(x - cx) + abs(y - cy) for cx, cy in center_access) <= 3
        for x, y, _ in animal_tiles
    )
    occupied_quadrants = {
        (x >= 5, y >= 5) for x, y, _ in animal_tiles
    }
    assert len(occupied_quadrants) >= 3
    assert maximum_third_field_utilization >= 60
    assert batch.rewards[0, 0] > 20_000
