"""Current rule-based Kaggriculture policy.

All strategy choices live here. The bertani package supplies
only reusable planning, task, scheduling, encoding, and market abstractions.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from bertani.market import MarketPlanBatch, MarketRule
from bertani.opening import OpeningController, OpeningTurn
from bertani.rule_based import (
    RuleConfig,
    RulePhase,
    StrategicIntent,
    VectorRulePolicy,
    animal_counts_total,
    extract_rule_features,
)
from bertani.tasks import TaskBatch, TaskKind, TaskRule
from bertani.vec_env import Batch, Item, MarketOp


# Replay steps 1..72 from submission 55463512. Tuple position zero is the
# action emitted from the initial step-0 observation. Unit slot zero is the
# farmer; later slots are farm hands in their stable insertion order.
OPENING_BOOK: tuple[OpeningTurn, ...] = (
    OpeningTurn(((14, 0, 0),), ((1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (5, 10, 2), (5, 11, 2), (3, 0, 7), (3, 4, 12), (4, 0, 6))),  # 0:00
    OpeningTurn(((5, 11, 1), (4, 0, 0), (1, 0, 0), (4, 0, 0), (4, 0, 0), (4, 0, 0)), ((6, 0, 3),)),  # 0:01
    OpeningTurn(((1, 0, 0), (5, 11, 1), (1, 0, 0), (1, 0, 0), (14, 0, 0), (4, 0, 0)), ()),  # 0:02
    OpeningTurn(((14, 0, 0), (1, 0, 0), (1, 0, 0), (5, 10, 1), (1, 0, 0), (1, 0, 0)), ()),  # 0:03
    OpeningTurn(((7, 11, 1), (4, 0, 0), (1, 0, 0), (7, 10, 1), (14, 0, 0), (4, 0, 0)), ()),  # 0:04
    OpeningTurn(((4, 0, 0), (7, 11, 1), (8, 0, 0), (5, 0, 1), (1, 0, 0), (1, 0, 0)), ()),  # 0:05
    OpeningTurn(((3, 0, 0), (4, 0, 0), (9, 0, 0), (1, 0, 0), (8, 4, 0), (8, 0, 0)), ((4, 0, 2),)),  # 0:06
    OpeningTurn(((2, 0, 0), (8, 4, 0), (4, 0, 0), (15, 0, 0), (9, 0, 0), (9, 0, 0)), ()),  # 0:07
    OpeningTurn(((5, 0, 1), (9, 0, 0), (1, 0, 0), (17, 0, 0), (1, 0, 0), (1, 0, 0)), ()),  # 0:08
    OpeningTurn(((4, 0, 0), (4, 0, 0), (8, 0, 0), (2, 0, 0), (8, 4, 0), (1, 0, 0)), ()),  # 0:09
    OpeningTurn(((1, 0, 0), (8, 4, 0), (9, 0, 0), (5, 10, 1), (9, 0, 0), (8, 0, 0)), ()),  # 0:10
    OpeningTurn(((15, 0, 0), (9, 0, 0), (4, 0, 0), (4, 0, 0), (4, 0, 0), (9, 0, 0)), ()),  # 0:11
    OpeningTurn(((17, 0, 0), (1, 0, 0), (4, 0, 0), (7, 10, 1), (4, 0, 0), (4, 0, 0)), ((4, 0, 1),)),  # 0:12
    OpeningTurn(((1, 0, 0), (8, 4, 0), (8, 0, 0), (4, 0, 0), (8, 0, 0), (4, 0, 0)), ()),  # 0:13
    OpeningTurn(((4, 0, 0), (9, 0, 0), (9, 0, 0), (4, 0, 0), (9, 0, 0), (8, 0, 0)), ()),  # 0:14
    OpeningTurn(((1, 0, 0), (4, 0, 0), (3, 0, 0), (8, 4, 0), (4, 0, 0), (9, 0, 0)), ()),  # 0:15
    OpeningTurn(((8, 4, 0), (8, 4, 0), (3, 0, 0), (9, 0, 0), (8, 4, 0), (2, 0, 0)), ()),  # 0:16
    OpeningTurn(((9, 0, 0), (9, 0, 0), (3, 0, 0), (4, 0, 0), (9, 0, 0), (2, 0, 0)), ()),  # 0:17
    OpeningTurn(((0, 0, 0), (2, 0, 0), (8, 4, 0), (8, 4, 0), (0, 0, 0), (0, 0, 0)), ()),  # 0:18
    OpeningTurn(((0, 0, 0), (8, 4, 0), (9, 0, 0), (9, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 0:19
    OpeningTurn(((0, 0, 0), (9, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 0:20
    OpeningTurn(((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 0:21
    OpeningTurn(((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 0:22
    OpeningTurn(((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 0:23
    OpeningTurn(((5, 0, 3),), ()),  # 1:00
    OpeningTurn(((15, 0, 0),), ()),  # 1:01
    OpeningTurn(((4, 0, 0),), ()),  # 1:02
    OpeningTurn(((15, 0, 0),), ()),  # 1:03
    OpeningTurn(((17, 0, 0),), ()),  # 1:04
    OpeningTurn(((1, 0, 0),), ()),  # 1:05
    OpeningTurn(((15, 0, 0),), ()),  # 1:06
    OpeningTurn(((17, 0, 0),), ()),  # 1:07
    OpeningTurn(((3, 0, 0),), ()),  # 1:08
    OpeningTurn(((2, 0, 0),), ()),  # 1:09
    OpeningTurn(((17, 0, 0),), ()),  # 1:10
    OpeningTurn(((16, 0, 0),), ()),  # 1:11
    OpeningTurn(((1, 0, 0),), ()),  # 1:12
    OpeningTurn(((16, 0, 0),), ()),  # 1:13
    OpeningTurn(((4, 0, 0),), ()),  # 1:14
    OpeningTurn(((16, 0, 0),), ()),  # 1:15
    OpeningTurn(((3, 0, 0),), ()),  # 1:16
    OpeningTurn(((2, 0, 0),), ()),  # 1:17
    OpeningTurn(((7, 8, 3),), ((6, 8, 3), (4, 0, 5))),  # 1:18
    OpeningTurn(((5, 0, 1),), ()),  # 1:19
    OpeningTurn(((1, 0, 0),), ()),  # 1:20
    OpeningTurn(((15, 0, 0),), ()),  # 1:21
    OpeningTurn(((17, 0, 0),), ()),  # 1:22
    OpeningTurn(((4, 0, 0),), ()),  # 1:23
    OpeningTurn(((5, 0, 4),), ((1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0))),  # 2:00
    OpeningTurn(((15, 0, 0), (4, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0)), ()),  # 2:01
    OpeningTurn(((17, 0, 0), (4, 0, 0), (4, 0, 0), (1, 0, 0), (16, 0, 0)), ((4, 0, 2),)),  # 2:02
    OpeningTurn(((1, 0, 0), (16, 0, 0), (1, 0, 0), (1, 0, 0), (2, 0, 0)), ()),  # 2:03
    OpeningTurn(((15, 0, 0), (1, 0, 0), (16, 0, 0), (4, 0, 0), (16, 0, 0)), ()),  # 2:04
    OpeningTurn(((17, 0, 0), (1, 0, 0), (4, 0, 0), (1, 0, 0), (4, 0, 0)), ()),  # 2:05
    OpeningTurn(((4, 0, 0), (9, 0, 0), (9, 0, 0), (9, 0, 0), (4, 0, 0)), ()),  # 2:06
    OpeningTurn(((15, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (4, 0, 0)), ()),  # 2:07
    OpeningTurn(((17, 0, 0), (9, 0, 0), (9, 0, 0), (9, 0, 0), (9, 0, 0)), ((4, 0, 2),)),  # 2:08
    OpeningTurn(((2, 0, 0), (1, 0, 0), (1, 0, 0), (4, 0, 0), (1, 0, 0)), ()),  # 2:09
    OpeningTurn(((15, 0, 0), (9, 0, 0), (9, 0, 0), (4, 0, 0), (9, 0, 0)), ()),  # 2:10
    OpeningTurn(((17, 0, 0), (4, 0, 0), (4, 0, 0), (9, 0, 0), (1, 0, 0)), ()),  # 2:11
    OpeningTurn(((4, 0, 0), (4, 0, 0), (9, 0, 0), (4, 0, 0), (9, 0, 0)), ()),  # 2:12
    OpeningTurn(((4, 0, 0), (9, 0, 0), (4, 0, 0), (4, 0, 0), (4, 0, 0)), ()),  # 2:13
    OpeningTurn(((4, 0, 0), (2, 0, 0), (9, 0, 0), (9, 0, 0), (9, 0, 0)), ()),  # 2:14
    OpeningTurn(((9, 0, 0), (2, 0, 0), (0, 0, 0), (0, 0, 0), (2, 0, 0)), ()),  # 2:15
    OpeningTurn(((3, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (9, 0, 0)), ()),  # 2:16
    OpeningTurn(((3, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 2:17
    OpeningTurn(((14, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 2:18
    OpeningTurn(((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 2:19
    OpeningTurn(((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 2:20
    OpeningTurn(((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 2:21
    OpeningTurn(((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 2:22
    OpeningTurn(((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 2:23
)


class IntentPlanner:
    """Choose workforce, expansion, and production from live economics."""

    def __init__(self, config: RuleConfig) -> None:
        self.config = config

    def __call__(self, batch: Batch) -> StrategicIntent:
        features = extract_rule_features(batch, self.config)
        shape = features.step.shape
        total_days = (
            self.config.episode_steps + self.config.turns_per_day - 1
        ) // self.config.turns_per_day
        liquidation_start = max(0, total_days - self.config.liquidation_days)

        phase = np.full(shape, RulePhase.MIDGAME, dtype=np.int8)
        phase[features.day < 3] = RulePhase.OPENING
        phase[features.day >= liquidation_start] = RulePhase.LIQUIDATION
        shops = np.rint(
            batch.observation_views.global_features[..., 22:30] * 8
        ).astype(np.int64)
        labor_market = (
            (shops[..., 5] > 0)
            | (shops[..., 2] >= 2)
            | ((features.day < 10) & (shops[..., 2] > 0))
        )
        target_hands = np.where(labor_market, 9, 8).astype(np.int64)
        opening_hands = np.array([5, 0, 4], dtype=np.int64)
        opening = features.day < opening_hands.size
        target_hands[opening] = opening_hands[features.day[opening]]
        target_hands[phase == RulePhase.LIQUIDATION] = 0
        cash_reserve = np.full(shape, 1_000.0, dtype=np.float64)
        cash_reserve[phase != RulePhase.MIDGAME] = 0.0
        wheat_reserve = 2 * animal_counts_total(features.animal_counts)
        target_crop_counts = np.zeros((*shape, 5), dtype=np.int64)
        target_crop_counts[..., Item.WHEAT] = 15
        ratios = features.market_price_ratios
        candidate_items = np.asarray(
            (Item.CARROT, Item.TOMATO, Item.STRAWBERRY, Item.MELON),
            dtype=np.int64,
        )
        yields = np.asarray((4.0, 4.0, 4.0, 6.0))
        bases = np.asarray((35.0, 60.0, 120.0, 250.0))
        costs = np.asarray((20.0, 50.0, 100.0, 80.0))
        cycle_days = np.asarray((4.0, 12.0, 18.0, 13.0))
        scores = (
            yields * bases * ratios[..., candidate_items] - costs
        ) / cycle_days
        remaining_days = total_days - features.day
        scores = np.where(
            remaining_days[..., None] > cycle_days,
            scores,
            -np.inf,
        )
        best = np.argmax(scores, axis=-1)
        for option, crop in enumerate(candidate_items):
            target_crop_counts[..., crop] = np.where(best == option, 9, 0)
        target_animal_counts = np.broadcast_to(
            np.asarray(self.config.opening_animal_targets, dtype=np.int64),
            (*shape, 3),
        ).copy()
        return StrategicIntent(
            phase=phase,
            target_hands=target_hands,
            cash_reserve=cash_reserve,
            wheat_reserve=wheat_reserve,
            target_crop_counts=target_crop_counts,
            target_animal_counts=target_animal_counts,
            liquidate=phase == RulePhase.LIQUIDATION,
        )


class MaintenanceTaskRule:
    """Generate survival, collection, harvest, and shed-fetch tasks."""

    def __init__(
        self,
        turns_per_day: int = 24,
        shed_capacity: int = 100,
        episode_steps: int = 720,
    ) -> None:
        self.turns_per_day = turns_per_day
        self.shed_capacity = shed_capacity
        self.episode_days = max(
            1, (episode_steps + turns_per_day - 1) // turns_per_day
        )

    def propose(
        self,
        batch: Batch,
        intent: StrategicIntent,
        tasks: TaskBatch,
    ) -> None:
        del intent  # Maintenance depends only on current survival state.
        tiles = batch.observation_views.tiles[:, :, 0]
        plants = tiles[..., 3] > 0.5
        animals = tiles[..., 6:9].sum(axis=-1) > 0.5
        watered_or_fed = tiles[..., 15] > 0.5
        cared = tiles[..., 16] > 0.5
        consecutive_missed = tiles[..., 17] * 2.0
        harvestable = tiles[..., 23] > 0.5
        crop_age = np.rint(tiles[..., 14] * self.episode_days).astype(np.int16)
        crop_channels = tiles[..., 9:14]
        # One-time crops are worth waiting for until their maximum-yield day.
        # Ongoing crops and animal products should be collected as soon as held.
        one_time_ready = (
            (crop_channels[..., Item.WHEAT] > 0.5) & (crop_age >= 4)
        ) | ((crop_channels[..., Item.CARROT] > 0.5) & (crop_age >= 3)) | (
            (crop_channels[..., Item.MELON] > 0.5) & (crop_age >= 12)
        )
        ongoing = crop_channels[..., Item.TOMATO : Item.MELON].sum(axis=-1) > 0.5
        animal_product = animals & harvestable
        harvest_now = harvestable & (one_time_ready | ongoing | animal_product)
        fertilizer_available = tiles[..., 20] > 0.5

        # Lower-priority proposals are installed first; later urgent proposals
        # win the same tile through TaskBatch's priority arbitration.
        tasks.propose_tiles(
            TaskKind.CARE,
            animals & ~cared,
            85.0 + 5.0 * consecutive_missed,
            deadline=self.turns_per_day - 1,
        )
        tasks.propose_tiles(
            TaskKind.COLLECT_FERTILIZER,
            animals & fertilizer_available,
            80.0,
            estimated_value=100.0,
        )
        tasks.propose_tiles(
            TaskKind.WATER,
            plants & ~watered_or_fed,
            90.0 + 20.0 * consecutive_missed,
            deadline=self.turns_per_day - 1,
        )
        fertilizer_days = tiles[..., 19] * 3.0
        ongoing_needing_fertilizer = ongoing & (fertilizer_days < 1.0)
        tasks.propose_tiles(
            TaskKind.FERTILIZE,
            ongoing_needing_fertilizer,
            85.0,
            required_item=Item.FERTILIZER,
            required_count=1,
        )
        tasks.propose_tiles(
            TaskKind.HARVEST,
            harvest_now,
            100.0,
        )
        feed_priority = 110.0 + 30.0 * consecutive_missed
        needs_feed = animals & ~watered_or_fed
        tasks.propose_tiles(
            TaskKind.FEED,
            needs_feed,
            feed_priority,
            deadline=self.turns_per_day - 1,
            required_item=Item.WHEAT,
            required_count=1,
        )

        self._propose_wheat_fetch(batch, tasks, needs_feed, feed_priority)
        self._propose_fertilizer_fetch(
            batch, tasks, ongoing_needing_fertilizer
        )

    def _propose_fertilizer_fetch(
        self,
        batch: Batch,
        tasks: TaskBatch,
        needs_fertilizer: NDArray[np.bool_],
    ) -> None:
        views = batch.observation_views
        units = views.units[:, :, 0]
        carried = np.rint(
            units[..., 5 + int(Item.FERTILIZER)] * self.shed_capacity
        ).astype(np.int64)
        carried *= batch.active_units
        available = np.rint(
            views.private[..., int(Item.FERTILIZER)] * self.shed_capacity
        ).astype(np.int64)
        missing = np.maximum(
            0,
            needs_fertilizer.sum(axis=(2, 3), dtype=np.int64)
            - carried.sum(axis=-1),
        )
        total_fetch = np.minimum(missing, available)
        access = max(0, tasks.board_size // 2 - 1)
        for index, slot in enumerate((6, 7, 8)):
            quotient, remainder = np.divmod(total_fetch, 3)
            quantity = quotient + (remainder > index)
            tasks.set_global(
                slot,
                quantity > 0,
                TaskKind.FETCH_ITEM,
                access,
                access,
                86.0,
                item=Item.FERTILIZER,
                quantity=quantity,
            )

    def _propose_wheat_fetch(
        self,
        batch: Batch,
        tasks: TaskBatch,
        needs_feed: NDArray[np.bool_],
        feed_priority: NDArray[np.float32],
    ) -> None:
        views = batch.observation_views
        units = views.units[:, :, 0]
        carried_wheat = np.rint(
            units[..., 5 + int(Item.WHEAT)] * self.shed_capacity
        ).astype(np.int64)
        carried_wheat *= batch.active_units
        available_wheat = np.rint(
            views.private[..., int(Item.WHEAT)] * self.shed_capacity
        ).astype(np.int64)
        feed_count = needs_feed.sum(axis=(2, 3), dtype=np.int64)
        missing = np.maximum(0, feed_count - carried_wheat.sum(axis=-1))
        total_fetch = np.minimum(missing, available_wheat)
        half = tasks.board_size // 2
        access = max(0, half - 1)
        maximum_feed_priority = feed_priority.max(axis=(2, 3)) + 1.0
        fetch_slots = (0, 3, 4, 5)
        quotient, remainder = np.divmod(total_fetch, len(fetch_slots))
        for index, slot in enumerate(fetch_slots):
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
                deadline=self.turns_per_day - 1,
            )


class ProductionTaskRule:
    """Maintain crops, expand pasture, place livestock, and deposit goods."""

    def __init__(self, shed_capacity: int = 100) -> None:
        self.shed_capacity = shed_capacity

    def propose(
        self,
        batch: Batch,
        intent: StrategicIntent,
        tasks: TaskBatch,
    ) -> None:
        views = batch.observation_views
        tiles = views.tiles[:, :, 0]
        productive = ~intent.liquidate

        weeds = (tiles[..., 2] > 0.5) & productive[..., None, None]
        tasks.propose_tiles(TaskKind.CLEAR_WEED, weeds, 55.0)

        animal_counts = np.rint(tiles[..., 6:9].sum(axis=(2, 3))).astype(
            np.int64
        )
        pasture_count = np.rint(
            (tiles[..., 5] + tiles[..., 7] + tiles[..., 8]).sum(axis=(2, 3))
        ).astype(np.int64)
        target_pastures = intent.target_animal_counts[..., 1:].sum(axis=-1)
        missing_pastures = np.maximum(0, target_pastures - pasture_count)
        y, x = np.indices(tiles.shape[-3:-1])
        northeast = (x >= tasks.board_size // 2) & (y < tasks.board_size // 2)
        build_candidates = (
            (tiles[..., 0] > 0.5)
            & northeast
            & productive[..., None, None]
        )
        build = self._select_limited(build_candidates, missing_pastures)
        tasks.propose_tiles(TaskKind.BUILD_PASTURE, build, 76.0)

        empty = (tiles[..., 0] > 0.5) & productive[..., None, None]
        crop_counts = np.rint(tiles[..., 9:14].sum(axis=(2, 3))).astype(np.int64)
        seeds = np.rint(views.private[..., 12:17] * 10).astype(np.int64)
        claimed = np.zeros_like(empty)
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
            selected = np.zeros_like(empty)
            for environment, player in np.argwhere(available > 0):
                candidates = np.flatnonzero(
                    (
                        empty[environment, player]
                        & ~claimed[environment, player]
                    ).reshape(-1)
                )
                count = min(int(available[environment, player]), candidates.size)
                if count:
                    selected[environment, player].reshape(-1)[candidates[:count]] = True
                    claimed[environment, player].reshape(-1)[candidates[:count]] = True
                    if crop != Item.WHEAT:
                        cash_slots[environment, player] -= count
            priority = 68.0 if crop == Item.WHEAT else 64.0
            tasks.propose_tiles(
                TaskKind.PLANT,
                selected,
                priority,
                item=crop,
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
        )

        carrying = (inventories.sum(axis=-1) > 0) & batch.active_units
        needs_deposit = carrying.any(axis=-1) & productive
        half = tasks.board_size // 2
        tasks.set_global(
            1,
            needs_deposit,
            TaskKind.DEPOSIT_INVENTORY,
            max(0, half - 1),
            max(0, half - 1),
            60.0,
            exclusive=False,
        )

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

class EconomyMarketRule:
    """Fund daily work, replenish inputs, and sell inventory opportunistically."""

    def __init__(
        self,
        starting_money: int = 3_000,
        shed_capacity: int = 100,
        episode_steps: int = 720,
    ) -> None:
        self.starting_money = starting_money
        self.shed_capacity = shed_capacity
        self.last_step = max(1, episode_steps - 1)

    def propose(
        self,
        batch: Batch,
        intent: StrategicIntent,
        plan: MarketPlanBatch,
    ) -> None:
        views = batch.observation_views
        active = ~intent.liquidate
        shed = np.rint(views.private[..., :12] * self.shed_capacity).astype(
            np.int64
        )
        seeds = np.rint(views.private[..., 12:17] * 10).astype(np.int64)
        tiles = views.tiles[:, :, 0]
        crops = np.rint(tiles[..., 9:14].sum(axis=(2, 3))).astype(np.int64)
        units = views.units[:, :, 0]
        carried = np.rint(units[..., 5:17] * self.shed_capacity).astype(
            np.int64
        )
        carried *= batch.active_units[..., None]
        carried_wheat = carried[..., Item.WHEAT].sum(axis=-1)
        money = views.farms[:, :, 0, 0] * self.starting_money
        hands = np.maximum(0, batch.active_units.sum(axis=-1) - 1)
        shed_total = shed.sum(axis=-1)
        pressure = shed_total >= int(self.shed_capacity * 0.7)
        ratios = views.global_features[..., 5:22:2]
        step = np.rint(
            views.global_features[..., 0] * self.last_step
        ).astype(np.int64)
        post_town_demand = (step % 4) == 1

        # Sales lead purchases so current inventory can fund the day's inputs.
        # Premium goods tolerate a lower threshold; fertilizer is monetized
        # immediately until a fertilization rule is introduced.
        thresholds = np.asarray(
            (1.0, 1.0, 0.9, 0.7, 0.7, 0.9, 0.7, 0.7, 0.0),
            dtype=np.float32,
        )
        ongoing_count = (tiles[..., 11] + tiles[..., 12]).sum(axis=(2, 3))
        fertilizer_reserve = np.minimum(
            9, np.ceil(ongoing_count / 3.0)
        ).astype(np.int64)
        fertilizer = np.maximum(
            0, shed[..., Item.FERTILIZER] - fertilizer_reserve
        )
        plan.append(
            active & (fertilizer > 0),
            MarketOp.SELL,
            item=Item.FERTILIZER,
            count=fertilizer,
        )
        for item in (
            Item.MILK,
            Item.WOOL,
            Item.MELON,
            Item.STRAWBERRY,
            Item.TOMATO,
            Item.EGG,
            Item.CARROT,
        ):
            count = shed[..., item]
            sell = active & (count > 0) & (post_town_demand | pressure)
            plan.append(sell, MarketOp.SELL, item=item, count=count)

        wheat_surplus = np.maximum(0, shed[..., Item.WHEAT] - intent.wheat_reserve)
        plan.append(
            active & (wheat_surplus > 0) & (
                (
                    (ratios[..., Item.WHEAT] >= thresholds[Item.WHEAT])
                    & post_town_demand
                )
                | pressure
            ),
            MarketOp.SELL,
            item=Item.WHEAT,
            count=wheat_surplus,
        )

        wheat_owned = shed[..., Item.WHEAT] + carried_wheat
        wheat_shortfall = np.maximum(0, intent.wheat_reserve - wheat_owned)
        plan.append(
            active & (wheat_shortfall > 0),
            MarketOp.BUY_PRODUCT,
            item=Item.WHEAT,
            count=wheat_shortfall,
        )

        farms = views.farms[:, :, 0]
        unlocked = np.rint(farms[..., 4:8].sum(axis=-1)).astype(np.int64)
        land_buy = active & (unlocked < 2) & (money >= 1_500)
        plan.append(land_buy, MarketOp.BUY_LAND)

        animal_counts = np.rint(tiles[..., 6:9].sum(axis=(2, 3))).astype(
            np.int64
        )
        carried_animals = carried[..., Item.COW : Item.SHEEP + 1].sum(axis=-2)
        owned_cows = (
            animal_counts[..., 1]
            + shed[..., Item.COW]
            + carried_animals[..., 0]
        )
        owned_sheep = (
            animal_counts[..., 2]
            + shed[..., Item.SHEEP]
            + carried_animals[..., 1]
        )
        missing_cows = np.maximum(
            0, intent.target_animal_counts[..., 1] - owned_cows
        )
        missing_sheep = np.maximum(
            0, intent.target_animal_counts[..., 2] - owned_sheep
        )
        expansion_ready = active & ((unlocked >= 2) | land_buy)
        budget = np.maximum(0, money - 200 - land_buy * 1_000).astype(np.int64)
        buy_cows = np.minimum(missing_cows, budget // 400)
        budget -= buy_cows * 400
        buy_sheep = np.minimum(missing_sheep, budget // 500)
        plan.append(
            expansion_ready & (buy_cows > 0),
            MarketOp.BUY_ANIMAL,
            item=Item.COW,
            count=buy_cows,
        )
        plan.append(
            expansion_ready & (buy_sheep > 0),
            MarketOp.BUY_ANIMAL,
            item=Item.SHEEP,
            count=buy_sheep,
        )

        # Five hands cost only 12 coins at the default multiplier and turn the
        # maintenance backlog into parallel work. Orders remain individually
        # represented because HIRE has no count argument.
        missing_hands = np.maximum(0, intent.target_hands - hands)
        can_hire = active & (money >= 12)
        for hire_index in range(int(intent.target_hands.max(initial=0))):
            plan.append(can_hire & (missing_hands > hire_index), MarketOp.HIRE)

        # Buy exactly the missing crop stock. The unit executor will plant only
        # seeds already visible at the start of a turn, avoiding atomic overbid.
        for crop in (
            Item.WHEAT,
            Item.CARROT,
            Item.TOMATO,
            Item.STRAWBERRY,
            Item.MELON,
        ):
            missing = np.maximum(
                0,
                intent.target_crop_counts[..., crop]
                - crops[..., crop]
                - seeds[..., crop],
            )
            plan.append(
                active & (missing > 0),
                MarketOp.BUY_SEED,
                item=crop,
                count=missing,
            )


def build_policy(
    config: RuleConfig | None = None,
    *,
    use_opening: bool = True,
    liquidation_days: int = 1,
) -> VectorRulePolicy:
    """Construct the current strategy on the reusable policy engine."""
    resolved = replace(
        config or RuleConfig(),
        liquidation_days=liquidation_days,
        opening_crop_targets=(10, 0, 0, 0, 9),
        opening_animal_targets=(0, 8, 4),
    )
    return VectorRulePolicy(
        resolved,
        intent_planner=IntentPlanner(resolved),
        opening_controller=(
            OpeningController(
                resolved.episode_steps,
                OPENING_BOOK,
                pasture_recovery=(2, 4, 66),
            )
            if use_opening
            else None
        ),
        task_rules=(
            MaintenanceTaskRule(
                turns_per_day=resolved.turns_per_day,
                shed_capacity=resolved.shed_capacity,
                episode_steps=resolved.episode_steps,
            ),
            ProductionTaskRule(shed_capacity=resolved.shed_capacity),
        ),
        market_rules=(
            EconomyMarketRule(
                starting_money=resolved.starting_money,
                shed_capacity=resolved.shed_capacity,
                episode_steps=resolved.episode_steps,
            ),
        ),
    )


__all__ = [
    "EconomyMarketRule",
    "MaintenanceTaskRule",
    "OPENING_BOOK",
    "ProductionTaskRule",
    "IntentPlanner",
    "build_policy",
]
