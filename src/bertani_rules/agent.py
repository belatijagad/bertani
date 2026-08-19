"""Current rule-based Kaggriculture policy.

All strategy choices live here. The bertani package supplies
only reusable planning, task, scheduling, encoding, and market abstractions.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

try:
    from bertani._rust import NativeWorkforcePlanner as _NativeWorkforcePlanner
except (ImportError, ModuleNotFoundError):
    _NativeWorkforcePlanner = None

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
    propose_native_farm_tasks,
    propose_native_maintenance_tasks,
    propose_native_production_tasks,
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
        shops = features.shop_counts
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
        opponent_crops = features.opponent_crop_counts
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
    """Thin Python wrapper around the persistent native workforce planner."""

    def __init__(
        self,
        shed_capacity: int = 100,
        turns_per_day: int = 24,
        episode_steps: int = 720,
        role_bonus: float = 0.0,
        zone_bonus: float = 0.0,
    ) -> None:
        if _NativeWorkforcePlanner is None:
            raise RuntimeError(
                "native workforce planning requires the bertani._rust extension"
            )
        self.role_bonus = role_bonus
        self.zone_bonus = zone_bonus
        self._native = _NativeWorkforcePlanner(
            shed_capacity, turns_per_day, episode_steps
        )
        self._shape: tuple[int, int, int] | None = None
        self._role: NDArray[np.int16] | None = None
        self._zone: NDArray[np.int16] | None = None
        self._reserved: NDArray[np.int16] | None = None

    def __call__(
        self,
        batch: Batch,
        intent: StrategicIntent,
        tasks: TaskBatch,
    ) -> WorkforcePlan:
        del intent
        shape = batch.active_units.shape
        if self._shape != shape:
            self._role = np.empty(shape, dtype=np.int16)
            self._zone = np.empty(shape, dtype=np.int16)
            self._reserved = np.empty(
                (*shape[:2], max(TaskKind) + 1), dtype=np.int16
            )
            self._shape = shape
        assert self._role is not None
        assert self._zone is not None
        assert self._reserved is not None

        views = batch.observation_views
        self._native.plan(
            views.global_features,
            views.farms,
            views.tiles,
            views.units,
            batch.active_units,
            tasks.active,
            tasks.kind,
            tasks.target_x,
            tasks.target_y,
            tasks.work_role,
            self._role,
            self._zone,
            self._reserved,
            tasks.board_size,
        )
        return WorkforcePlan(
            role=self._role,
            zone=self._zone,
            role_bonus=self.role_bonus,
            zone_bonus=self.zone_bonus,
            reserved_by_kind=self._reserved,
        )

    def plan_masked(
        self,
        batch: Batch,
        intent: StrategicIntent,
        tasks: TaskBatch,
        seat_mask: NDArray[np.bool_],
    ) -> WorkforcePlan:
        # Workforce state is cheap in Rust and day-persistent. Keep all seats
        # synchronized so a later change in controlled seat cannot inherit
        # stale territory state. The expensive Python implementation is gone.
        del seat_mask
        return self(batch, intent, tasks)


class FarmTaskRule:
    """Generate maintenance and production mechanics in one native call."""

    profile_key = "farm_tasks"

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
        propose_native_farm_tasks(
            batch,
            intent,
            tasks,
            turns_per_day=self.turns_per_day,
            shed_capacity=self.shed_capacity,
            episode_steps=self.episode_steps,
        )

    def propose_with_market_plan(
        self,
        batch: Batch,
        intent: StrategicIntent,
        tasks: TaskBatch,
        market_plan: MarketPlanBatch,
    ) -> None:
        propose_native_farm_tasks(
            batch,
            intent,
            tasks,
            market_plan=market_plan,
            turns_per_day=self.turns_per_day,
            shed_capacity=self.shed_capacity,
            episode_steps=self.episode_steps,
        )

    def propose_with_market_plan_masked(
        self,
        batch: Batch,
        intent: StrategicIntent,
        tasks: TaskBatch,
        market_plan: MarketPlanBatch,
        seat_mask: NDArray[np.bool_],
    ) -> None:
        propose_native_farm_tasks(
            batch,
            intent,
            tasks,
            market_plan=market_plan,
            seat_mask=seat_mask,
            turns_per_day=self.turns_per_day,
            shed_capacity=self.shed_capacity,
            episode_steps=self.episode_steps,
        )

    def propose_masked(
        self,
        batch: Batch,
        intent: StrategicIntent,
        tasks: TaskBatch,
        seat_mask: NDArray[np.bool_],
    ) -> None:
        propose_native_farm_tasks(
            batch,
            intent,
            tasks,
            seat_mask=seat_mask,
            turns_per_day=self.turns_per_day,
            shed_capacity=self.shed_capacity,
            episode_steps=self.episode_steps,
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

    def propose_with_market_plan(
        self,
        batch: Batch,
        intent: StrategicIntent,
        tasks: TaskBatch,
        market_plan: MarketPlanBatch,
    ) -> None:
        propose_native_production_tasks(
            batch,
            intent,
            tasks,
            market_plan=market_plan,
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

    def propose_masked(
        self,
        batch: Batch,
        intent: StrategicIntent,
        plan: MarketPlanBatch,
        seat_mask: NDArray[np.bool_],
    ) -> None:
        propose_native_rule_market(
            batch,
            intent,
            plan,
            seat_mask=seat_mask,
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
            FarmTaskRule(
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
    "FarmTaskRule",
    "MaintenanceTaskRule",
    "OPENING_BOOK",
    "ProductionTaskRule",
    "IntentPlanner",
    "build_policy",
]
