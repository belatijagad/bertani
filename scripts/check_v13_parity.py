"""Turn-by-turn parity checks for V13 native feature/workforce planning."""

from __future__ import annotations

import argparse
import numpy as np
from numpy.typing import NDArray

from bertani.rule_based import RuleFeatures, extract_rule_features
from bertani.tasks import TaskBatch, TaskKind, WorkforcePlan, WorkRole, WorkZone
from bertani.vec_env import Batch, Item, VecEnv
from bertani_rules.agent import (
    FarmTaskRule,
    IntentPlanner,
    TerritorialWorkforcePlanner,
    build_policy,
)


def python_extract_rule_features(batch: Batch, config) -> RuleFeatures:
    views = batch.observation_views
    global_features = views.global_features
    own_farms = views.farms[:, :, 0]
    own_tiles = views.tiles[:, :, 0]
    opponent_tiles = views.tiles[:, :, 1]
    last_step = max(1, config.episode_steps - 1)
    step = np.rint(global_features[..., 0] * last_step).astype(np.int64)
    day = step // config.turns_per_day
    hour = step % config.turns_per_day
    money = own_farms[..., 0].astype(np.float64) * config.starting_money
    crop_counts = np.rint(own_tiles[..., 9:14].sum(axis=(2, 3))).astype(np.int64)
    animal_counts = np.rint(own_tiles[..., 6:9].sum(axis=(2, 3))).astype(np.int64)
    shed = np.rint(views.private[..., :12] * config.shed_capacity).astype(np.int64)
    seeds = np.rint(views.private[..., 12:17] * 10).astype(np.int64)
    shops = np.rint(global_features[..., 22:30] * 8).astype(np.int64)
    opponent_crops = np.rint(opponent_tiles[..., 9:14].sum(axis=(2, 3))).astype(np.int64)
    return RuleFeatures(
        step=step,
        day=day,
        hour=hour,
        money=money,
        crop_counts=crop_counts,
        animal_counts=animal_counts,
        shed=shed,
        seeds=seeds,
        shop_counts=shops,
        opponent_crop_counts=opponent_crops,
        market_price_ratios=global_features[..., 5:22:2].copy(),
    )

class PythonTerritorialWorkforcePlanner:
    """Allocate crews to quadrants in proportion to their live task backlog."""

    def __init__(
        self,
        shed_capacity: int = 100,
        turns_per_day: int = 24,
        episode_steps: int = 720,
        role_bonus: float = 0.0,
        zone_bonus: float = 0.0,
    ) -> None:
        self.shed_capacity = shed_capacity
        self.turns_per_day = turns_per_day
        self.last_step = max(1, episode_steps - 1)
        self.role_bonus = role_bonus
        self.zone_bonus = zone_bonus
        self._zone_shape: tuple[int, int, int] | None = None
        self._daily_zone: NDArray[np.int16] | None = None
        self._last_day: NDArray[np.int64] | None = None

    def __call__(
        self,
        batch: Batch,
        intent: StrategicIntent,
        tasks: TaskBatch,
    ) -> WorkforcePlan:
        del intent
        active = batch.active_units
        shape = active.shape
        active_slots = np.flatnonzero(np.any(active, axis=(0, 1)))
        active_limit = int(active_slots[-1]) + 1 if active_slots.size else 0
        active_prefix = active[..., :active_limit]

        role = np.full(shape, WorkRole.ANY, dtype=np.int16)
        zone = np.full(shape, WorkZone.ANY, dtype=np.int16)
        role_prefix = role[..., :active_limit]
        zone_prefix = zone[..., :active_limit]
        unit_index = np.arange(active_limit, dtype=np.int16)[None, None, :]

        tiles = batch.observation_views.tiles[:, :, 0]
        animals = np.rint(tiles[..., 6:9].sum(axis=(2, 3))).astype(np.int16)
        animal_total = animals.sum(axis=-1)
        livestock_workers = np.where(
            animal_total > 0,
            np.clip((animal_total + 3) // 4, 1, 3),
            0,
        ).astype(np.int16)

        logistics = active_prefix & (unit_index == 0)
        livestock = active_prefix & (unit_index >= 1) & (
            unit_index <= livestock_workers[..., None]
        )
        field = active_prefix & ~logistics & ~livestock
        role_prefix[logistics] = WorkRole.LOGISTICS
        role_prefix[livestock] = WorkRole.LIVESTOCK
        role_prefix[field] = WorkRole.FIELD

        farms = batch.observation_views.farms[:, :, 0]
        unlocked = np.rint(farms[..., 4:8]).astype(np.bool_)

        # Estimate one-turn workload rather than splitting workers evenly.
        # Survival and harvest jobs receive extra mass because delaying them can
        # destroy assets or block the following planting job.  Weeds and empty
        # tiles therefore attract workers naturally when their backlog grows.
        kind = tasks.kind
        weight = np.ones_like(tasks.priority, dtype=np.float32)
        weight += np.isin(
            kind,
            (TaskKind.WATER, TaskKind.FEED, TaskKind.CARE),
        ).astype(np.float32)
        weight += 0.5 * np.isin(
            kind,
            (TaskKind.HARVEST, TaskKind.CLEAR_WEED, TaskKind.PLANT),
        ).astype(np.float32)
        task_slots = np.arange(tasks.capacity)[None, None, :]
        tile_task = task_slots < tasks.tile_slots
        task_zone = (
            (tasks.target_y >= tasks.board_size // 2).astype(np.int16) * 2
            + (tasks.target_x >= tasks.board_size // 2).astype(np.int16)
        )
        demand = np.zeros((*shape[:2], 4), dtype=np.float32)
        for quadrant in range(4):
            in_quadrant = (
                tasks.active
                & tile_task
                & (task_zone == quadrant)
                & (tasks.work_role != WorkRole.LOGISTICS)
            )
            demand[..., quadrant] = (weight * in_quadrant).sum(axis=-1)

        demand = np.where(unlocked, demand, 0.0)

        territory_workers = active_prefix & ~logistics
        total_demand = demand.sum(axis=-1)
        fallback = unlocked.astype(np.float32)
        effective_demand = np.where(
            (total_demand > 0)[..., None], demand, fallback
        )
        step = np.rint(
            batch.observation_views.global_features[..., 0] * self.last_step
        ).astype(np.int64)
        day = step // self.turns_per_day
        hour = step % self.turns_per_day
        if self._zone_shape != shape:
            self._daily_zone = np.full(shape, WorkZone.ANY, dtype=np.int16)
            self._last_day = np.full(shape[:2], -1, dtype=np.int64)
            self._zone_shape = shape
        assert self._daily_zone is not None
        assert self._last_day is not None
        new_day = day != self._last_day
        self._daily_zone[new_day] = WorkZone.ANY
        self._daily_zone[~active] = WorkZone.ANY
        self._last_day[...] = day

        # Assign each newly hired worker to the quadrant with the highest
        # remaining demand per assigned worker. Keep that territory for the
        # rest of the day so a changing task list cannot make routes thrash.
        #
        # Only the live unit prefix participates here. The native observation
        # buffer reserves 231 slots, but the policy normally uses around a
        # dozen workers; iterating every padded slot dominated this planner.
        daily_zone_prefix = self._daily_zone[..., :active_limit]
        assigned_count = np.stack(
            [
                ((daily_zone_prefix == quadrant) & territory_workers).sum(
                    axis=-1
                )
                for quadrant in range(4)
            ],
            axis=-1,
        ).astype(np.float32)
        for worker_raw in active_slots:
            worker = int(worker_raw)
            if worker == 0:
                continue
            needs_zone = territory_workers[..., worker] & (
                daily_zone_prefix[..., worker] == WorkZone.ANY
            )
            if not np.any(needs_zone):
                continue
            pressure = effective_demand / (assigned_count + 1.0)
            chosen = np.argmax(pressure, axis=-1).astype(np.int16)
            daily_zone_prefix[..., worker] = np.where(
                needs_zone, chosen, daily_zone_prefix[..., worker]
            )
            for quadrant in range(4):
                assigned_count[..., quadrant] += needs_zone & (
                    chosen == quadrant
                )
        zone_prefix[territory_workers] = daily_zone_prefix[territory_workers]

        # Inventory state overrides the default daily role. A worker already
        # carrying an animal should finish pasture placement; any other loaded
        # worker should complete the short shed route before returning afield.
        units = batch.observation_views.units[:, :, 0, :active_limit]
        inventories = np.rint(
            units[..., 5:17] * self.shed_capacity
        ).astype(np.int64)
        carrying_animal = inventories[..., Item.COW : Item.SHEEP + 1].sum(
            axis=-1
        ) > 0
        carrying_anything = inventories.sum(axis=-1) > 0
        role_prefix[active_prefix & carrying_anything] = WorkRole.LOGISTICS
        role_prefix[active_prefix & carrying_animal] = WorkRole.LIVESTOCK

        plant_backlog = (
            tasks.active & (tasks.kind == TaskKind.PLANT)
        ).sum(axis=-1)
        active_count = active_prefix.sum(axis=-1)
        reserved_by_kind = np.zeros(
            (*shape[:2], max(TaskKind) + 1), dtype=np.int16
        )
        reserve_planting = (
            (day >= 14)
            & (hour < self.turns_per_day - 4)
            & (plant_backlog >= 8)
            & (active_count >= 8)
        )
        reserved_by_kind[..., TaskKind.PLANT] = reserve_planting.astype(
            np.int16
        )

        return WorkforcePlan(
            role=role,
            zone=zone,
            role_bonus=self.role_bonus,
            zone_bonus=self.zone_bonus,
            reserved_by_kind=reserved_by_kind,
        )



def compare_features(turn: int, left: RuleFeatures, right: RuleFeatures) -> None:
    for field in (
        "step", "day", "hour", "money", "crop_counts", "animal_counts",
        "shed", "seeds", "shop_counts", "opponent_crop_counts",
        "market_price_ratios",
    ):
        a = getattr(left, field)
        b = getattr(right, field)
        if not np.array_equal(a, b):
            mismatch = np.argwhere(a != b)
            index = tuple(mismatch[0]) if mismatch.size else ()
            raise AssertionError(
                f"turn {turn} feature {field} mismatch at {index}: "
                f"python={a[index] if index else 'n/a'} native={b[index] if index else 'n/a'}"
            )


def compare_workforce(turn: int, left: WorkforcePlan, right: WorkforcePlan) -> None:
    for field in ("role", "zone", "reserved_by_kind"):
        a = getattr(left, field)
        b = getattr(right, field)
        if a is None or b is None:
            if a is not b:
                raise AssertionError(f"turn {turn} workforce {field} None mismatch")
            continue
        if not np.array_equal(a, b):
            mismatch = np.argwhere(a != b)
            index = tuple(mismatch[0]) if mismatch.size else ()
            raise AssertionError(
                f"turn {turn} workforce {field} mismatch at {index}: "
                f"python={a[index] if index else 'n/a'} native={b[index] if index else 'n/a'}"
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
    driver = build_policy()
    config = driver.config
    intent_planner = IntentPlanner(config)
    farm_rule = FarmTaskRule(
        shed_capacity=config.shed_capacity,
        turns_per_day=config.turns_per_day,
        episode_steps=config.episode_steps,
    )
    tasks = TaskBatch.allocate(args.num_seeds, 2, env.board_size)
    python_workforce = PythonTerritorialWorkforcePlanner(
        shed_capacity=config.shed_capacity,
        turns_per_day=config.turns_per_day,
        episode_steps=config.episode_steps,
        role_bonus=0.0,
        zone_bonus=0.1,
    )
    native_workforce = TerritorialWorkforcePlanner(
        shed_capacity=config.shed_capacity,
        turns_per_day=config.turns_per_day,
        episode_steps=config.episode_steps,
        role_bonus=0.0,
        zone_bonus=0.1,
    )

    for turn in range(config.episode_steps - 1):
        expected_features = python_extract_rule_features(batch, config)
        native_features = extract_rule_features(batch, config)
        compare_features(turn, expected_features, native_features)

        expected_intent = intent_planner.from_features(batch, expected_features)
        native_intent = intent_planner.from_features(batch, native_features)
        for field in (
            "phase", "target_hands", "cash_reserve", "wheat_reserve",
            "target_crop_counts", "target_animal_counts", "liquidate",
        ):
            np.testing.assert_array_equal(
                getattr(expected_intent, field), getattr(native_intent, field),
                err_msg=f"turn {turn} intent {field}",
            )

        tasks.clear()
        farm_rule.propose(batch, native_intent, tasks)
        expected_workforce = python_workforce(batch, native_intent, tasks)
        native_plan = native_workforce(batch, native_intent, tasks)
        compare_workforce(turn, expected_workforce, native_plan)

        actions = driver.act(batch, max_orders=env.max_orders)
        batch = env.step(
            actions.unit_actions, actions.market_actions, actions.market_lengths
        )

    print(
        f"V13 parity passed: {args.num_seeds} environments x "
        f"{config.episode_steps - 1} turns"
    )


if __name__ == "__main__":
    main()
