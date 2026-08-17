"""Current rule-based Kaggriculture policy.

All strategy choices live here. The bertani package supplies
only reusable planning, task, scheduling, encoding, and market abstractions.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from bertani.market import MarketPlanBatch, propose_native_rule_market
from bertani.opening import OpeningController, OpeningTurn
from bertani.rule_based import (
    RuleConfig,
    RulePhase,
    StrategicIntent,
    VectorRulePolicy,
    animal_counts_total,
    extract_rule_features,
)
from bertani.tasks import (
    TaskBatch,
    TaskKind,
    TaskRule,
    WorkforcePlan,
    propose_native_maintenance_tasks,
    propose_native_production_tasks,
    WorkRole,
    WorkZone,
)
from bertani.vec_env import Batch, Item


# Earliest crop age at which this policy can realize a harvest.  Used only to
# stop starting new crop cohorts that cannot pay out before the season ends.
# Order matches the five crop Item indices used by target_crop_counts.
CROP_HARVEST_DAYS = np.asarray((4, 3, 8, 10, 10), dtype=np.int64)


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
        return self.from_features(batch, features)

    def from_features(
        self,
        batch: Batch,
        features,
    ) -> StrategicIntent:
        """Plan from already-decoded rule features.

        VectorRulePolicy uses this fast path so the same observation is not
        decoded twice on every non-opening turn. Calling the planner directly
        keeps the original behavior through ``__call__``.
        """

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
        # The leader commits to Yarn only when it appears among the first four
        # shop draws. Later Yarn unlocks do not rebuild an established dairy
        # farm. Sheep count latches the branch after that decision point.
        early_yarn = (
            ((features.day <= 12) & (shops[..., 7] >= 1))
            | (features.animal_counts[..., 2] > 4)
        )
        early_double_yarn = (
            ((features.day <= 12) & (shops[..., 7] >= 2))
            | (features.animal_counts[..., 2] > 8)
        )
        opponent_tiles = batch.observation_views.tiles[:, :, 1]
        opponent_crops = np.rint(
            opponent_tiles[..., 9:14].sum(axis=(2, 3))
        ).astype(np.int64)
        # The reference player uses two visibly different expansion books.
        # A wheat/livestock-heavy rival is answered with a 20-Melon cash
        # cohort.  Against another Melon/dairy opening it keeps only twelve
        # Melons and turns the second field into Strawberry earlier.  Crop
        # counts make this decision causal and seat-independent; opponent cash
        # happened to correlate with it in the downloaded games but is only a
        # transient consequence of the rival's purchases.
        opponent_field_heavy = (
            (opponent_crops[..., Item.WHEAT] >= 6)
            & (opponent_crops[..., Item.MELON] <= 8)
        )
        opponent_opening_known = opponent_crops.sum(axis=-1) >= 15

        # Daily workforce reconstructed from the replay median. Twelve hands
        # can maintain the stable board when crops are planted in cohorts.
        target_hands = np.full(shape, 12, dtype=np.int64)
        target_hands[(features.day >= 14) & (features.day <= 21)] = 13
        target_hands[features.day < 6] = 5
        target_hands[features.day == 6] = 4
        # Seven hands match the amount of immediately actionable expansion
        # work. The eighth hand previously cost another Fibonacci hire, then
        # spent the end of the day without a seed-backed task.
        target_hands[features.day == 7] = 7
        target_hands[features.day == 8] = 11
        opening_hands = np.array([5, 0, 4], dtype=np.int64)
        opening = features.day < opening_hands.size
        target_hands[opening] = opening_hands[features.day[opening]]
        target_hands[phase == RulePhase.LIQUIDATION] = 10

        cash_reserve = np.full(shape, 200.0, dtype=np.float64)
        cash_reserve[phase != RulePhase.MIDGAME] = 0.0
        wheat_reserve = np.maximum(
            6, animal_counts_total(features.animal_counts) + 2
        )
        # On expansion day, temporarily run a lean feed buffer so Wheat sales
        # can complete the $1,000 land fund. Replenishment resumes afterward.
        wheat_reserve = np.where(features.day == 6, 2, wheat_reserve)

        # Daily targets describe the board we want at the *next* day boundary.
        # The 313 downloaded leader replays use a second Melon cohort after the
        # first land purchase, then finance the day-11 Wheat/Strawberry rebuild
        # with that cohort's harvest. Replacing those tiles with Strawberry on
        # day eight was the largest early cash divergence in our old policy.
        target_crop_counts = np.zeros((*shape, 5), dtype=np.int64)
        target_crop_counts[..., Item.WHEAT] = np.where(
            features.day < 3, 7, np.where(features.day < 8, 3, 7)
        )
        target_crop_counts[..., Item.WHEAT] = np.select(
            (
                features.day == 10,
                features.day == 11,
                features.day == 12,
                features.day >= 13,
            ),
            (16, 19, 16, 19),
            default=target_crop_counts[..., Item.WHEAT],
        )

        target_crop_counts[..., Item.STRAWBERRY] = np.where(
            features.day >= 5, 4, 0
        )
        target_crop_counts[..., Item.STRAWBERRY] = np.select(
            (
                features.day == 6,
                (features.day >= 7) & (features.day <= 10),
                features.day == 11,
                features.day >= 12,
            ),
            (7, 8, 31, 34),
            default=target_crop_counts[..., Item.STRAWBERRY],
        )
        target_crop_counts[..., Item.STRAWBERRY] = np.where(
            (features.day >= 8) & (features.day <= 10),
            11,
            target_crop_counts[..., Item.STRAWBERRY],
        )

        target_crop_counts[..., Item.MELON] = np.select(
            (
                features.day == 6,
                features.day == 7,
                features.day == 8,
                features.day == 9,
                (features.day >= 10) & (features.day <= 13),
                features.day >= 14,
            ),
            (14, 18, 19, 20, 8, 0),
            default=12,
        )

        mirror_opening = (
            (features.day >= 6)
            & (features.day <= 10)
            & opponent_opening_known
            & ~opponent_field_heavy
        )
        target_crop_counts[..., Item.MELON] = np.where(
            mirror_opening, 12, target_crop_counts[..., Item.MELON]
        )
        target_crop_counts[..., Item.STRAWBERRY] = np.where(
            mirror_opening & (features.day >= 8),
            19,
            target_crop_counts[..., Item.STRAWBERRY],
        )

        # One early Yarn Store trades the remaining Melon cohort and four cows
        # for a compact 6-cow/8-sheep, 42-Strawberry board. Two early Yarn
        # Stores retain eight Melons and buy the fourth quadrant for Wheat and
        # four additional sheep.
        yarn_rebuild = early_yarn & (features.day >= 11)
        single_yarn = yarn_rebuild & ~early_double_yarn
        target_crop_counts[..., Item.STRAWBERRY] = np.where(
            single_yarn,
            np.where(features.day == 11, 34, 42),
            target_crop_counts[..., Item.STRAWBERRY],
        )
        target_crop_counts[..., Item.MELON] = np.where(
            single_yarn, 0, target_crop_counts[..., Item.MELON]
        )
        target_crop_counts[..., Item.STRAWBERRY] = np.where(
            yarn_rebuild & early_double_yarn,
            34,
            target_crop_counts[..., Item.STRAWBERRY],
        )
        target_crop_counts[..., Item.MELON] = np.where(
            yarn_rebuild & early_double_yarn,
            8,
            target_crop_counts[..., Item.MELON],
        )

        late = (features.day >= 22) & (features.day < liquidation_start)
        late_index = np.clip(features.day - 22, 0, 6)
        late_strawberry = np.asarray((38, 34, 30, 27, 24, 21, 18))
        late_wheat = np.asarray((23, 27, 31, 34, 34, 34, 34))
        late_carrot = np.asarray((1, 2, 3, 4, 4, 4, 4))
        target_crop_counts[..., Item.WHEAT] = np.where(
            late, late_wheat[late_index], target_crop_counts[..., Item.WHEAT]
        )
        target_crop_counts[..., Item.STRAWBERRY] = np.where(
            late,
            late_strawberry[late_index],
            target_crop_counts[..., Item.STRAWBERRY],
        )
        target_crop_counts[..., Item.CARROT] = np.where(
            late, late_carrot[late_index], 0
        )
        target_crop_counts[..., Item.MELON] = np.where(
            late, 0, target_crop_counts[..., Item.MELON]
        )

        # Four-field double-Yarn games use the extra acreage primarily for
        # Wheat and Sheep.
        target_crop_counts[..., Item.WHEAT] = np.where(
            early_double_yarn & (features.day >= 11) & (features.day < 22),
            26,
            target_crop_counts[..., Item.WHEAT],
        )

        target_animal_counts = np.zeros((*shape, 3), dtype=np.int64)
        target_animal_counts[..., 1] = 3
        target_animal_counts[..., 2] = 2
        target_animal_counts[..., 1] = np.select(
            (
                features.day >= 11,
                features.day >= 9,
                features.day >= 7,
                features.day >= 5,
                features.day >= 3,
            ),
            (10, 8, 6, 4, 3),
            default=2,
        )
        target_animal_counts[..., 2] = np.select(
            (
                features.day >= 8,
                features.day >= 7,
            ),
            (4, 3),
            default=2,
        )
        yarn_livestock = early_yarn & (features.day >= 9)
        target_animal_counts[..., 1] = np.where(
            yarn_livestock, 6, target_animal_counts[..., 1]
        )
        target_animal_counts[..., 2] = np.where(
            yarn_livestock, 8, target_animal_counts[..., 2]
        )
        target_animal_counts[..., 2] = np.where(
            early_double_yarn & (features.day >= 11),
            12,
            target_animal_counts[..., 2],
        )
        # Do not start a new crop cohort unless it can reach this policy's
        # harvest age before the final day.  This replaces the old blanket
        # two-day cutoff, which was far too late for Tomato/Strawberry/Melon.
        remaining_days = total_days - features.day
        crop_viable = remaining_days[..., None] > CROP_HARVEST_DAYS
        target_crop_counts = np.where(
            crop_viable,
            target_crop_counts,
            0,
        )
        target_crop_counts[phase == RulePhase.LIQUIDATION] = 0
        return StrategicIntent(
            phase=phase,
            target_hands=target_hands,
            cash_reserve=cash_reserve,
            wheat_reserve=wheat_reserve,
            target_crop_counts=target_crop_counts,
            target_animal_counts=target_animal_counts,
            liquidate=phase == RulePhase.LIQUIDATION,
        )


class TerritorialWorkforcePlanner:
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


class MaintenanceTaskRule:
    """Generate deterministic maintenance tasks in the native extension."""

    profile_key = "maintenance_tasks"

    def __init__(
        self,
        turns_per_day: int = 24,
        shed_capacity: int = 100,
        episode_steps: int = 720,
    ) -> None:
        self.turns_per_day = turns_per_day
        self.shed_capacity = shed_capacity
        self.episode_steps = episode_steps

    def propose(
        self,
        batch: Batch,
        intent: StrategicIntent,
        tasks: TaskBatch,
    ) -> None:
        del intent  # Maintenance depends only on current survival state.
        propose_native_maintenance_tasks(
            batch,
            tasks,
            turns_per_day=self.turns_per_day,
            shed_capacity=self.shed_capacity,
            episode_steps=self.episode_steps,
        )


class ProductionTaskRule:
    """Translate strategic farm targets into concrete tasks in Rust."""

    profile_key = "production_tasks"

    def __init__(
        self,
        shed_capacity: int = 100,
        turns_per_day: int = 24,
        episode_steps: int = 720,
    ) -> None:
        self.shed_capacity = shed_capacity
        self.turns_per_day = turns_per_day
        self.episode_steps = episode_steps

    def propose(
        self,
        batch: Batch,
        intent: StrategicIntent,
        tasks: TaskBatch,
    ) -> None:
        propose_native_production_tasks(
            batch,
            intent,
            tasks,
            turns_per_day=self.turns_per_day,
            shed_capacity=self.shed_capacity,
            episode_steps=self.episode_steps,
        )

class EconomyMarketRule:
    """Fast native backend for the current hand-written market strategy."""

    def __init__(
        self,
        starting_money: int = 3_000,
        shed_capacity: int = 100,
        episode_steps: int = 720,
        turns_per_day: int = 24,
    ) -> None:
        self.starting_money = starting_money
        self.shed_capacity = shed_capacity
        self.turns_per_day = turns_per_day
        self.episode_steps = episode_steps

    def propose(
        self,
        batch: Batch,
        intent: StrategicIntent,
        plan: MarketPlanBatch,
    ) -> None:
        propose_native_rule_market(
            batch,
            intent,
            plan,
            starting_money=self.starting_money,
            shed_capacity=self.shed_capacity,
            episode_steps=self.episode_steps,
            turns_per_day=self.turns_per_day,
        )


def build_policy(
    config: RuleConfig | None = None,
    *,
    use_opening: bool = True,
    liquidation_days: int = 1,
    profile: bool = False,
) -> VectorRulePolicy:
    """Construct the current strategy on the reusable policy engine."""
    resolved = replace(
        config or RuleConfig(),
        liquidation_days=liquidation_days,
        opening_crop_targets=(7, 0, 0, 0, 12),
        opening_animal_targets=(0, 10, 4),
    )
    return VectorRulePolicy(
        resolved,
        intent_planner=IntentPlanner(resolved),
        workforce_planner=TerritorialWorkforcePlanner(
            shed_capacity=resolved.shed_capacity,
            turns_per_day=resolved.turns_per_day,
            episode_steps=resolved.episode_steps,
            role_bonus=0.0,
            zone_bonus=0.1,
        ),
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
            ProductionTaskRule(
                shed_capacity=resolved.shed_capacity,
                turns_per_day=resolved.turns_per_day,
                episode_steps=resolved.episode_steps,
            ),
        ),
        market_rules=(
            EconomyMarketRule(
                starting_money=resolved.starting_money,
                shed_capacity=resolved.shed_capacity,
                episode_steps=resolved.episode_steps,
                turns_per_day=resolved.turns_per_day,
            ),
        ),
        profile=profile,
    )
__all__ = [
    "EconomyMarketRule",
    "MaintenanceTaskRule",
    "OPENING_BOOK",
    "ProductionTaskRule",
    "IntentPlanner",
    "build_policy",
]
