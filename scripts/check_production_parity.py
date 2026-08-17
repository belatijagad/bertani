"""Exact parity check for native production-task generation.

Run after rebuilding the extension:
    uv run python scripts/check_production_parity.py --num-seeds 8
"""
from __future__ import annotations

import argparse

import numpy as np
from numpy.typing import NDArray

from bertani.tasks import TaskBatch, TaskKind, WorkRole
from bertani.vec_env import Item, VecEnv
from bertani_rules.agent import MaintenanceTaskRule, ProductionTaskRule, build_policy

PASTURE_SLOTS = (
    (3, 3),
    (4, 3),
    (3, 4),
    (4, 4),
    (2, 4),
    (4, 2),
    (5, 3),
    (5, 4),
    (6, 4),
    (6, 3),
    (5, 2),
    (7, 4),
    (3, 5),
    (4, 5),
    (6, 5),
    (6, 6),
    (5, 6),
    (5, 7),
)

class PythonProductionTaskRule:
    """Maintain crops, expand pasture, place livestock, and deposit goods."""

    profile_key = "production_tasks"

    def __init__(
        self,
        shed_capacity: int = 100,
        turns_per_day: int = 24,
        episode_steps: int = 720,
    ) -> None:
        self.shed_capacity = shed_capacity
        self.turns_per_day = turns_per_day
        self.last_step = max(1, episode_steps - 1)
        self.episode_days = max(
            1, (episode_steps + turns_per_day - 1) // turns_per_day
        )
        self._geometry_board_size: int | None = None
        self._center_distance: NDArray[np.int64] | None = None
        self._pasture_rank: NDArray[np.int16] | None = None

    def propose(
        self,
        batch: Batch,
        intent: StrategicIntent,
        tasks: TaskBatch,
    ) -> None:
        views = batch.observation_views
        tiles = views.tiles[:, :, 0]
        productive = ~intent.liquidate

        step = np.rint(
            views.global_features[..., 0] * self.last_step
        ).astype(np.int64)
        hour = step % self.turns_per_day
        day = step // self.turns_per_day
        weeds = (tiles[..., 2] > 0.5) & productive[..., None, None]
        crop_age = np.rint(tiles[..., 14] * self.episode_days).astype(np.int16)
        crop_channels = tiles[..., 9:14] > 0.5
        harvestable = tiles[..., 23] > 0.5
        exhausted_ongoing = (
            (
                (crop_channels[..., Item.TOMATO] & (crop_age >= 11))
                | (
                    crop_channels[..., Item.STRAWBERRY]
                    & (crop_age >= 16)
                )
            )
            & ~harvestable
            & (day <= self.episode_days - 5)[..., None, None]
            & productive[..., None, None]
        )
        weed_count = weeds.sum(axis=(2, 3))
        weed_priority = np.where(
            (day >= 22) & (weed_count >= 4), 109.0, 99.0
        )
        tasks.propose_tiles(
            TaskKind.CLEAR_WEED,
            weeds,
            weed_priority[..., None, None],
            work_role=WorkRole.FIELD,
        )
        tasks.propose_tiles(
            TaskKind.CLEAR_WEED,
            exhausted_ongoing,
            99.0,
            work_role=WorkRole.FIELD,
        )

        animal_counts = np.rint(tiles[..., 6:9].sum(axis=(2, 3))).astype(
            np.int64
        )
        pasture_count = np.rint(
            (tiles[..., 5] + tiles[..., 7] + tiles[..., 8]).sum(axis=(2, 3))
        ).astype(np.int64)
        target_pastures = intent.target_animal_counts[..., 1:].sum(axis=-1)
        missing_pastures = np.maximum(0, target_pastures - pasture_count)
        half, center_distance, pasture_rank = self._geometry(tasks.board_size)
        # The leader uses a stable center-out pasture template rather than
        # choosing whichever nearby tile happens to be empty. Fixed slots keep
        # animal routes compact and make every expansion deterministic.
        desired_pastures = (
            (pasture_rank >= 0)[None, None]
            & (pasture_rank[None, None] < target_pastures[..., None, None])
        )
        build_candidates = (
            (tiles[..., 0] > 0.5)
            & desired_pastures
            & productive[..., None, None]
        )
        existing_pastures = (
            tiles[..., 5] + tiles[..., 7] + tiles[..., 8]
        ) > 0.5
        build = self._select_limited_by_distance(
            build_candidates,
            missing_pastures,
            pasture_rank,
            existing_pastures,
        )
        tasks.propose_tiles(
            TaskKind.BUILD_PASTURE,
            build,
            105.0,
            work_role=WorkRole.LIVESTOCK,
        )

        # A plant starts with one missed watering day. Leave enough turns for a
        # worker to water it before the end-of-day refresh.
        safe_to_plant = hour < self.turns_per_day - 2
        empty = (
            (tiles[..., 0] > 0.5)
            & productive[..., None, None]
            & safe_to_plant[..., None, None]
        )
        shops = np.rint(views.global_features[..., 22:30] * 8).astype(np.int64)
        reserved_pasture_count = np.maximum(14, target_pastures)
        reserved_pastures = (
            (pasture_rank >= 0)[None, None]
            & (pasture_rank[None, None] < reserved_pasture_count[..., None, None])
        )
        empty &= ~reserved_pastures
        crop_counts = np.rint(tiles[..., 9:14].sum(axis=(2, 3))).astype(np.int64)
        seeds = np.rint(views.private[..., 12:17] * 10).astype(np.int64)
        # Do not let a higher-priority crop proposal overwrite pasture selected
        # for construction on this turn.
        claimed = build.copy()
        existing_production = tiles[..., 3:9].sum(axis=-1) > 0.5
        planned_seed_use = np.zeros_like(seeds)
        cash_slots = np.maximum(
            0,
            intent.target_crop_counts[..., 1:].sum(axis=-1)
            - crop_counts[..., 1:].sum(axis=-1),
        )
        # Wheat gets first choice because livestock survival depends on it.
        for crop in (
            Item.WHEAT,
            Item.MELON,
            Item.CARROT,
            Item.TOMATO,
            Item.STRAWBERRY,
        ):
            deficit = np.maximum(
                0, intent.target_crop_counts[..., crop] - crop_counts[..., crop]
            )
            available = np.minimum(deficit, seeds[..., crop])
            if crop != Item.WHEAT:
                available = np.minimum(available, cash_slots)
            selected = self._select_limited_by_distance(
                empty & ~claimed,
                available,
                center_distance,
                existing_production | claimed,
            )
            selected_count = selected.sum(axis=(2, 3), dtype=np.int64)
            claimed |= selected
            planned_seed_use[..., crop] += selected_count
            if crop != Item.WHEAT:
                cash_slots -= selected_count
            tasks.propose_tiles(
                TaskKind.PLANT,
                selected,
                np.where(
                    (
                        ((day >= 7) & (day <= 8))
                        | ((day >= 11) & (day <= 13))
                    )[..., None, None],
                    115.0,
                    97.0,
                ),
                item=crop,
                work_role=WorkRole.FIELD,
            )

        # Strategic targets govern purchases, but seeds already in storage are
        # sunk cost. Use any viable surplus to reclaim empty productive tiles
        # without causing the market rule to buy a larger seed inventory.
        remaining_days = self.episode_days - day
        for crop, maturity_days in (
            (Item.WHEAT, 4),
            (Item.CARROT, 3),
            (Item.MELON, 10),
            (Item.TOMATO, 8),
            (Item.STRAWBERRY, 10),
        ):
            surplus = np.maximum(0, seeds[..., crop] - planned_seed_use[..., crop])
            available = np.where(remaining_days > maturity_days, surplus, 0)
            selected = self._select_limited_by_distance(
                empty & ~claimed,
                available,
                center_distance,
                existing_production | claimed,
            )
            selected_count = selected.sum(axis=(2, 3), dtype=np.int64)
            claimed |= selected
            planned_seed_use[..., crop] += selected_count
            tasks.propose_tiles(
                TaskKind.PLANT,
                selected,
                np.where((day >= 22)[..., None, None], 105.0, 104.0),
                item=crop,
                work_role=WorkRole.FIELD,
            )

        units = views.units[:, :, 0]
        inventories = np.rint(
            units[..., 5:17] * self.shed_capacity
        ).astype(np.int64)
        empty_pasture = (tiles[..., 5] > 0.5) & productive[..., None, None]
        claimed_pasture = np.zeros_like(empty_pasture)
        for animal in (Item.COW, Item.SHEEP):
            animal_index = int(animal) - int(Item.GOOSE)
            carried_count = inventories[..., animal].sum(axis=-1)
            deficit = np.maximum(
                0,
                intent.target_animal_counts[..., animal_index]
                - animal_counts[..., animal_index],
            )
            place_count = np.minimum(carried_count, deficit)
            candidates = empty_pasture & ~claimed_pasture
            place = self._select_limited(candidates, place_count)
            claimed_pasture |= place
            tasks.propose_tiles(
                TaskKind.PLACE_ANIMAL,
                place,
                145.0,
                item=animal,
                required_item=animal,
                required_count=1,
                work_role=WorkRole.LIVESTOCK,
            )

        shed = np.rint(views.private[..., :12] * self.shed_capacity).astype(
            np.int64
        )
        carried_animals = inventories[..., Item.COW : Item.SHEEP + 1].sum(
            axis=-2
        )
        cow_needed = (
            intent.target_animal_counts[..., 1] - animal_counts[..., 1]
        ) > 0
        sheep_needed = (
            intent.target_animal_counts[..., 2] - animal_counts[..., 2]
        ) > 0
        fetch_cow = cow_needed & (shed[..., Item.COW] > 0) & (
            carried_animals[..., 0] == 0
        )
        fetch_sheep = (~fetch_cow) & sheep_needed & (shed[..., Item.SHEEP] > 0) & (
            carried_animals[..., 1] == 0
        )
        fetch_animal = productive & (fetch_cow | fetch_sheep)
        fetch_item = np.where(fetch_cow, Item.COW, Item.SHEEP)
        fetch_quantity = np.where(
            fetch_cow,
            np.minimum(
                shed[..., Item.COW],
                np.maximum(
                    0,
                    intent.target_animal_counts[..., 1]
                    - animal_counts[..., 1],
                ),
            ),
            np.minimum(
                shed[..., Item.SHEEP],
                np.maximum(
                    0,
                    intent.target_animal_counts[..., 2]
                    - animal_counts[..., 2],
                ),
            ),
        )
        half = tasks.board_size // 2
        tasks.set_global(
            2,
            fetch_animal,
            TaskKind.FETCH_ITEM,
            max(0, half - 1),
            max(0, half - 1),
            150.0,
            item=fetch_item,
            quantity=fetch_quantity,
            work_role=WorkRole.LOGISTICS,
        )

        carrying = (inventories.sum(axis=-1) > 0) & batch.active_units
        needs_deposit = carrying.any(axis=-1)
        premium_carried = inventories[
            ...,
            (Item.STRAWBERRY, Item.MELON, Item.MILK, Item.WOOL),
        ].sum(axis=(-1, -2)) > 0
        deposit_priority = np.where(
            premium_carried,
            np.where(day == 6, 145.0, 112.0),
            60.0,
        )
        half = tasks.board_size // 2
        tasks.set_global(
            1,
            needs_deposit,
            TaskKind.DEPOSIT_INVENTORY,
            max(0, half - 1),
            max(0, half - 1),
            deposit_priority,
            exclusive=False,
            work_role=WorkRole.LOGISTICS,
        )

    def _geometry(
        self,
        board_size: int,
    ) -> tuple[int, NDArray[np.int64], NDArray[np.int16]]:
        """Cache board-static routing geometry instead of rebuilding it each turn."""

        if self._geometry_board_size != board_size:
            y, x = np.indices((board_size, board_size))
            half = board_size // 2
            low_center = max(0, half - 1)
            distance_x = np.minimum(np.abs(x - low_center), np.abs(x - half))
            distance_y = np.minimum(np.abs(y - low_center), np.abs(y - half))
            self._center_distance = distance_x + distance_y

            pasture_rank = np.full(
                (board_size, board_size),
                -1,
                dtype=np.int16,
            )
            for rank, (slot_x, slot_y) in enumerate(PASTURE_SLOTS):
                if slot_x < board_size and slot_y < board_size:
                    pasture_rank[slot_y, slot_x] = rank
            self._pasture_rank = pasture_rank
            self._geometry_board_size = board_size

        assert self._center_distance is not None
        assert self._pasture_rank is not None
        return board_size // 2, self._center_distance, self._pasture_rank

    @staticmethod
    def _select_limited(
        candidates: np.ndarray,
        counts: np.ndarray,
    ) -> np.ndarray:
        selected = np.zeros_like(candidates)
        for environment, player in np.argwhere(counts > 0):
            indices = np.flatnonzero(candidates[environment, player].reshape(-1))
            count = min(int(counts[environment, player]), indices.size)
            selected[environment, player].reshape(-1)[indices[:count]] = True
        return selected

    @staticmethod
    def _select_limited_by_distance(
        candidates: np.ndarray,
        counts: np.ndarray,
        distance: np.ndarray,
        existing: np.ndarray,
    ) -> np.ndarray:
        """Select center-out tiles while balancing unlocked quadrants.

        This is exactly equivalent to the original list-scanning selector, but
        keeps four pre-sorted quadrant queues. Each choice then considers at
        most four queue heads instead of rescanning and removing from a Python
        list of every remaining tile.
        """

        selected = np.zeros_like(candidates)
        flat_distance = distance.reshape(-1)
        board_size = candidates.shape[-1]
        half = board_size // 2

        y, x = np.indices((board_size, board_size))
        flat_y = y.reshape(-1)
        flat_x = x.reshape(-1)
        quadrant = (
            (flat_y >= half).astype(np.int8) * 2
            + (flat_x >= half).astype(np.int8)
        )
        tie_y = np.minimum(
            np.abs(flat_y - (half - 1)),
            np.abs(flat_y - half),
        )
        tie_x = np.minimum(
            np.abs(flat_x - (half - 1)),
            np.abs(flat_x - half),
        )
        flat_index = np.arange(flat_distance.size, dtype=np.int64)

        # Primary key: distance. Tie-breaks reproduce the old chosen=min(...)
        # order exactly.
        base_order = np.lexsort(
            (flat_index, tie_x, tie_y, flat_distance)
        )
        quadrant_orders = tuple(
            base_order[quadrant[base_order] == quadrant_id]
            for quadrant_id in range(4)
        )

        for environment, player in np.argwhere(counts > 0):
            environment = int(environment)
            player = int(player)
            candidate_mask = candidates[environment, player].reshape(-1)
            if not np.any(candidate_mask):
                continue

            occupied = np.flatnonzero(
                existing[environment, player].reshape(-1)
            )
            quadrant_counts = np.bincount(
                quadrant[occupied],
                minlength=4,
            ).astype(np.int64)

            queues = tuple(
                order[candidate_mask[order]]
                for order in quadrant_orders
            )
            pointers = [0, 0, 0, 0]
            available_count = sum(queue.size for queue in queues)
            output = selected[environment, player].reshape(-1)

            for _ in range(
                min(int(counts[environment, player]), available_count)
            ):
                nearest: int | np.integer | None = None
                heads: list[tuple[int, int]] = []

                for quadrant_id, queue in enumerate(queues):
                    pointer = pointers[quadrant_id]
                    if pointer >= queue.size:
                        continue
                    index = int(queue[pointer])
                    candidate_distance = flat_distance[index]
                    if nearest is None or candidate_distance < nearest:
                        nearest = candidate_distance
                        heads = [(quadrant_id, index)]
                    elif candidate_distance == nearest:
                        heads.append((quadrant_id, index))

                least_occupied = min(
                    quadrant_counts[quadrant_id]
                    for quadrant_id, _ in heads
                )
                balanced = (
                    (quadrant_id, index)
                    for quadrant_id, index in heads
                    if quadrant_counts[quadrant_id] == least_occupied
                )
                chosen_quadrant, chosen = min(
                    balanced,
                    key=lambda candidate: (
                        tie_y[candidate[1]],
                        tie_x[candidate[1]],
                        candidate[1],
                    ),
                )

                output[chosen] = True
                quadrant_counts[chosen_quadrant] += 1
                pointers[chosen_quadrant] += 1

        return selected


def compare_tasks(reference: TaskBatch, native: TaskBatch, turn: int) -> None:
    exact_fields = (
        "active", "kind", "target_x", "target_y", "item", "quantity",
        "deadline", "required_item", "required_count", "exclusive", "work_role",
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
        close = np.isclose(left, right, rtol=0.0, atol=1e-6, equal_nan=True)
        if not np.all(close):
            mismatch = np.argwhere(~close)[0]
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
    env = VecEnv(args.num_seeds, seed=args.seed_source, auto_reset=False)
    batch = env.reset(seeds)
    policy = build_policy()
    assert policy.intent_planner is not None
    maintenance = MaintenanceTaskRule()
    python_rule = PythonProductionTaskRule()
    native_rule = ProductionTaskRule()
    reference = TaskBatch.allocate(args.num_seeds, 2, env.board_size)
    native = TaskBatch.allocate(args.num_seeds, 2, env.board_size)

    for turn in range(719):
        features = policy.extract_features(batch)
        planner_from_features = getattr(policy.intent_planner, "from_features")
        intent = planner_from_features(batch, features)

        reference.clear()
        native.clear()
        maintenance.propose(batch, intent, reference)
        maintenance.propose(batch, intent, native)
        python_rule.propose(batch, intent, reference)
        native_rule.propose(batch, intent, native)
        compare_tasks(reference, native, turn)

        actions = policy.act(batch, max_orders=env.max_orders)
        batch = env.step(
            actions.unit_actions,
            actions.market_actions,
            actions.market_lengths,
        )

    print(f"production parity passed: {args.num_seeds} environments x 719 turns")

if __name__ == "__main__":
    main()
