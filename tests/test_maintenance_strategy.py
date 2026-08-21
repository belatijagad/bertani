from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bertani._rust", reason="the maturin extension has not been built")

from bertani import Item, TaskBatch, TaskKind, VecEnv
from bertani.tasks import propose_native_maintenance_tasks


def _maintenance_tasks_for_tile(tile: np.ndarray) -> TaskBatch:
    env = VecEnv(1, seed=10, weed_spawn_chance=0.0)
    batch = env.reset()
    target = batch.observation_views.tiles[0, 0, 0, 0, 0]
    target.fill(0.0)
    target[: tile.size] = tile
    tasks = TaskBatch.allocate(1, 2, env.board_size)
    propose_native_maintenance_tasks(
        batch,
        tasks,
        turns_per_day=24,
        shed_capacity=100,
        episode_steps=720,
    )
    return tasks


def _strawberry(age: int, *, watered: bool, fertilized_days: float = 0.0, harvestable: bool = False) -> np.ndarray:
    tile = np.zeros(24, dtype=np.float32)
    tile[3] = 1.0
    tile[9 + int(Item.STRAWBERRY)] = 1.0
    tile[14] = age / 30.0
    tile[15] = float(watered)
    tile[19] = fertilized_days / 3.0
    tile[23] = float(harvestable)
    return tile


def _tomato(age: int, *, watered: bool, fertilized_days: float = 0.0) -> np.ndarray:
    tile = np.zeros(24, dtype=np.float32)
    tile[3] = 1.0
    tile[9 + int(Item.TOMATO)] = 1.0
    tile[14] = age / 30.0
    tile[15] = float(watered)
    tile[19] = fertilized_days / 3.0
    return tile


def test_strawberry_fertilizer_is_synchronized_to_yield_days() -> None:
    # No fertilizer is spent during the long pre-yield growth period.
    tasks = _maintenance_tasks_for_tile(_strawberry(9, watered=True))
    assert tasks.kind[0, 0, 0] != TaskKind.FERTILIZE

    # If old yield is still waiting, harvesting it first is legitimate.
    # The scheduler can then WATER and FERTILIZE later in the same day.
    tasks = _maintenance_tasks_for_tile(
        _strawberry(10, watered=False, harvestable=True)
    )
    assert tasks.kind[0, 0, 0] == TaskKind.HARVEST

    # With no old yield waiting, WATER is the first production-day task.
    tasks = _maintenance_tasks_for_tile(
        _strawberry(10, watered=False, harvestable=False)
    )
    assert tasks.kind[0, 0, 0] == TaskKind.WATER

    # If old yield is still waiting, HARVEST owns the tile first. After
    # harvesting, the same tile can still be fertilized later that day.
    tasks = _maintenance_tasks_for_tile(
        _strawberry(10, watered=True, harvestable=True)
    )
    assert tasks.kind[0, 0, 0] == TaskKind.HARVEST

    # Once old yield has been cleared, fertilization becomes the highest-value
    # remaining production-day task.
    tasks = _maintenance_tasks_for_tile(
        _strawberry(10, watered=True, harvestable=False)
    )
    assert tasks.kind[0, 0, 0] == TaskKind.FERTILIZE
    assert tasks.priority[0, 0, 0] == pytest.approx(115.0)

    # Strawberry does not produce at age 11, so no fertilizer is spent there.
    tasks = _maintenance_tasks_for_tile(_strawberry(11, watered=True))
    assert tasks.kind[0, 0, 0] != TaskKind.FERTILIZE

    # Existing fertilizer coverage suppresses a redundant application.
    tasks = _maintenance_tasks_for_tile(
        _strawberry(12, watered=True, fertilized_days=1.0)
    )
    assert tasks.kind[0, 0, 0] != TaskKind.FERTILIZE


def test_exhausted_strawberry_is_not_refertilized() -> None:
    # At age 16 with no yield left, the production rule considers Strawberry
    # exhausted and proposes CLEAR_WEED. Maintenance must not override it with
    # a fresh fertilizer request.
    tasks = _maintenance_tasks_for_tile(
        _strawberry(16, watered=True, harvestable=False)
    )
    assert tasks.kind[0, 0, 0] != TaskKind.FERTILIZE


def test_final_strawberry_yield_is_harvested_before_retirement() -> None:
    tasks = _maintenance_tasks_for_tile(
        _strawberry(16, watered=True, harvestable=True)
    )
    assert tasks.kind[0, 0, 0] == TaskKind.HARVEST


def test_tomato_fertilizer_tracks_actual_production_window() -> None:
    tasks = _maintenance_tasks_for_tile(_tomato(7, watered=True))
    assert tasks.kind[0, 0, 0] != TaskKind.FERTILIZE

    tasks = _maintenance_tasks_for_tile(_tomato(8, watered=True))
    assert tasks.kind[0, 0, 0] == TaskKind.FERTILIZE

    tasks = _maintenance_tasks_for_tile(
        _tomato(9, watered=True, fertilized_days=2.0)
    )
    assert tasks.kind[0, 0, 0] != TaskKind.FERTILIZE


def test_fed_animal_care_precedes_routine_harvest() -> None:
    tile = np.zeros(24, dtype=np.float32)
    tile[6] = 1.0  # any encoded animal channel
    tile[15] = 1.0  # already fed today
    tile[16] = 0.0  # not cared today
    tile[23] = 1.0  # product waiting to be harvested

    tasks = _maintenance_tasks_for_tile(tile)
    assert tasks.kind[0, 0, 0] == TaskKind.CARE
    assert tasks.priority[0, 0, 0] == pytest.approx(118.0)

    # Survival still dominates: an unfed animal must be fed before care.
    tile[15] = 0.0
    tasks = _maintenance_tasks_for_tile(tile)
    assert tasks.kind[0, 0, 0] == TaskKind.FEED
