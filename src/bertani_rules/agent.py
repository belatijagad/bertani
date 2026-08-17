"""Current rule-based Kaggriculture policy.

All strategy choices live here. The bertani package supplies
only reusable planning, task, scheduling, encoding, and market abstractions.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

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
from bertani.tasks import (
    TaskBatch,
    TaskKind,
    TaskRule,
    WorkforcePlan,
    WorkRole,
    WorkZone,
)
from bertani.vec_env import Batch, Item, MarketOp


# SHOP_NAMES order from bertani.kaggle_agent. Keeping the demand table beside
# the strategy avoids coupling the native batch policy to the Kaggle adapter.
SHOP_DEMAND = np.asarray(
    (
        # wheat, carrot, tomato, strawberry, melon, egg, milk, wool, fertilizer
        (1, 0, 0, 0, 0, 1, 0, 0, 0),  # bakery
        (1, 0, 0, 1, 0, 1, 0, 0, 0),  # brunch spot
        (1, 1, 1, 1, 0, 0, 0, 0, 0),  # farmers market
        (1, 0, 0, 1, 0, 0, 1, 0, 0),  # ice cream shop
        (0, 2, 0, 0, 0, 0, 0, 0, 0),  # pet cafe
        (1, 0, 1, 0, 0, 0, 1, 0, 0),  # pizza shop
        (0, 0, 0, 1, 0, 0, 1, 0, 0),  # smoothie shop
        (0, 0, 0, 0, 0, 0, 0, 2, 0),  # yarn store
    ),
    dtype=np.int64,
)

# Median requested order sizes in the downloaded 55463512 replays.
SALE_BATCHES = {
    Item.WHEAT: 7,
    Item.CARROT: 4,
    Item.STRAWBERRY: 8,
    Item.MELON: 12,
    Item.MILK: 6,
    Item.WOOL: 4,
    Item.FERTILIZER: 18,
}

# Product base prices indexed by Item. Current market-price observations are
# encoded as price/base, so these reconstruct the current quotes closely enough
# for small, targeted land-financing sales.
MARKET_BASE_PRICES = np.asarray(
    (25, 35, 60, 120, 250, 50, 160, 200, 100),
    dtype=np.float64,
)

SEED_BUY_BATCHES = {
    Item.WHEAT: 8,
    Item.CARROT: 4,
    Item.TOMATO: 4,
    Item.STRAWBERRY: 4,
    Item.MELON: 12,
}

# Earliest crop age at which this policy can realize a harvest.  Used only to
# stop starting new crop cohorts that cannot pay out before the season ends.
# Order matches the five crop Item indices used by target_crop_counts.
CROP_HARVEST_DAYS = np.asarray((4, 3, 8, 10, 10), dtype=np.int64)

# Exact center-out structure order shared by the sampled 55463512 replays.
# The first 14 slots are the standard three-quadrant plan; the last four are
# added only by the duplicated-Yarn-Store sheep branch.
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
        self.last_step = max(1, episode_steps - 1)

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
        step = np.rint(
            batch.observation_views.global_features[..., 0] * self.last_step
        ).astype(np.int64)
        day = step // self.turns_per_day
        crop_channels = tiles[..., 9:14]
        # One-time crops are worth waiting for until their maximum-yield day.
        # Ongoing crops and animal products should be collected as soon as held.
        one_time_ready = (
            (crop_channels[..., Item.WHEAT] > 0.5) & (crop_age >= 4)
        ) | ((crop_channels[..., Item.CARROT] > 0.5) & (crop_age >= 3)) | (
            (crop_channels[..., Item.MELON] > 0.5) & (crop_age >= 10)
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
            100.0 + 5.0 * consecutive_missed,
            deadline=self.turns_per_day - 1,
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
            deadline=self.turns_per_day - 1,
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
            deadline=self.turns_per_day - 1,
            required_item=Item.WHEAT,
            required_count=1,
            work_role=WorkRole.LIVESTOCK,
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
        fetch_slots = (6,)
        for index, slot in enumerate(fetch_slots):
            quotient, remainder = np.divmod(total_fetch, len(fetch_slots))
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
                work_role=WorkRole.LOGISTICS,
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
        fetch_slots = (0, 3)
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
                work_role=WorkRole.LOGISTICS,
            )


class ProductionTaskRule:
    """Maintain crops, expand pasture, place livestock, and deposit goods."""

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

class EconomyMarketRule:
    """Fund daily work, replenish inputs, and sell inventory opportunistically."""

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
        self.last_step = max(1, episode_steps - 1)
        self.episode_days = max(1, (episode_steps + turns_per_day - 1) // turns_per_day)

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
        town_tick = (step % 4) == 0
        town_center_tick = (step % 24) == 0
        shops = np.rint(views.global_features[..., 22:30] * 8).astype(
            np.int64
        )
        farms = views.farms[:, :, 0]
        unlocked = np.rint(farms[..., 4:8].sum(axis=-1)).astype(np.int64)
        day = step // self.turns_per_day

        # Land expansion is scheduled rather than opportunistic:
        #   quadrant 2 -> day 7,  $1,000
        #   quadrant 3 -> day 11, $2,000
        #
        # On day 6/day 10 we only protect the future land bankroll from
        # discretionary livestock purchases. On the target day, if cash is
        # still slightly short, sell just enough premium inventory BEFORE
        # BUY_LAND. This preserves the existing economy instead of entering a
        # broad "expansion mode".
        next_land_cost = np.where(
            unlocked == 1,
            1_000,
            np.where(unlocked == 2, 2_000, 0),
        ).astype(np.int64)
        next_land_day = np.where(
            unlocked == 1,
            7,
            np.where(unlocked == 2, 11, -1),
        ).astype(np.int64)

        reserve_for_land = (
            active
            & (next_land_day >= 0)
            & (day == (next_land_day - 1))
        )
        target_land_day = (
            active
            & (next_land_day >= 0)
            & (day >= next_land_day)
        )

        land_shortfall = np.maximum(
            0,
            next_land_cost - money,
        ).astype(np.int64)

        current_prices = np.maximum(
            1,
            np.rint(ratios * MARKET_BASE_PRICES).astype(np.int64),
        )

        # Use Melon first because the existing strategy explicitly treats its
        # second cohort as bridge capital. If that is insufficient, use Wool.
        melon_price = current_prices[..., Item.MELON]
        melon_needed = np.where(
            melon_price > 0,
            (land_shortfall + melon_price - 1) // melon_price,
            0,
        )
        land_finance_melon = np.minimum(
            shed[..., Item.MELON],
            melon_needed,
        )
        estimated_after_melon = (
            money + land_finance_melon * melon_price
        )

        remaining_shortfall = np.maximum(
            0,
            next_land_cost - estimated_after_melon,
        ).astype(np.int64)
        wool_price = current_prices[..., Item.WOOL]
        wool_needed = np.where(
            wool_price > 0,
            (remaining_shortfall + wool_price - 1) // wool_price,
            0,
        )
        land_finance_wool = np.minimum(
            shed[..., Item.WOOL],
            wool_needed,
        )

        financing = target_land_day & (land_shortfall > 0)
        plan.append(
            financing & (land_finance_melon > 0),
            MarketOp.SELL,
            item=Item.MELON,
            count=land_finance_melon,
        )
        plan.append(
            financing & (land_finance_wool > 0),
            MarketOp.SELL,
            item=Item.WOOL,
            count=land_finance_wool,
        )

        estimated_land_cash = (
            money
            + np.where(financing, land_finance_melon * melon_price, 0)
            + np.where(financing, land_finance_wool * wool_price, 0)
        )
        scheduled_first_expansion = (
            target_land_day
            & (unlocked == 1)
            & (estimated_land_cash >= 1_000)
        )
        scheduled_third_expansion = (
            target_land_day
            & (unlocked == 2)
            & (estimated_land_cash >= 2_000)
        )

        # Keep the existing optional fourth-quadrant Yarn expansion unchanged.
        yarn_expansion_early = (
            active
            & (day >= 12)
            & (unlocked == 3)
            & (intent.target_animal_counts[..., 2] >= 12)
            & (money >= 4_000)
        )
        early_land_buy = (
            scheduled_first_expansion
            | scheduled_third_expansion
            | yarn_expansion_early
        )
        plan.append(early_land_buy, MarketOp.BUY_LAND)

        # Wool is the bridge between the livestock opening and the second
        # crop field. A normal four-unit sale is price-efficient later, but it
        # strands cash in the shed during expansion. The reference baseline
        # liquidates a large wool stack here and immediately converts it into
        # land, labor, and Strawberry seeds.
        expansion_financing = (
            active
            & (day >= 6)
            & (day <= 8)
            & (unlocked <= 2)
            & (money < 2_000)
        )

        # Sell in the leader's median batch sizes. Premium stock is normally
        # sold immediately after town demand. At a four-turn tick with no
        # demand for that product, selling now front-runs the opponent instead.
        ongoing_count = (tiles[..., 11] + tiles[..., 12]).sum(axis=(2, 3))
        fertilizer_reserve = np.minimum(
            9, np.ceil(ongoing_count / 3.0)
        ).astype(np.int64)
        fertilizer_surplus = np.maximum(
            0, shed[..., Item.FERTILIZER] - fertilizer_reserve
        )
        plan.append(
            active & (fertilizer_surplus > 0),
            MarketOp.SELL,
            item=Item.FERTILIZER,
            count=np.minimum(
                fertilizer_surplus, SALE_BATCHES[Item.FERTILIZER]
            ),
        )
        for item in (
            Item.MILK,
            Item.WOOL,
            Item.MELON,
            Item.STRAWBERRY,
            Item.CARROT,
        ):
            shop_demand = np.tensordot(
                shops, SHOP_DEMAND[:, int(item)], axes=([-1], [0])
            )
            demand_now = (
                town_center_tick.astype(np.int64)
                + np.where(town_tick, shop_demand, 0)
            )
            sale_window = post_town_demand | (town_tick & (demand_now == 0))
            normal_count = np.minimum(shed[..., item], SALE_BATCHES[item])
            count = np.where(
                (item == Item.WOOL) & expansion_financing,
                shed[..., item],
                normal_count,
            )
            sell = active & (count > 0) & (
                sale_window
                | pressure
                | ((item == Item.WOOL) & expansion_financing)
            )
            plan.append(sell, MarketOp.SELL, item=item, count=count)

        # These products are outside the intended strategy, but monetize any
        # accidental inventory instead of occupying shed space forever.
        for item in (Item.TOMATO, Item.EGG):
            count = shed[..., item]
            plan.append(
                active & (count > 0) & (post_town_demand | pressure),
                MarketOp.SELL,
                item=item,
                count=count,
            )

        wheat_surplus = np.maximum(0, shed[..., Item.WHEAT] - intent.wheat_reserve)
        wheat_count = np.minimum(wheat_surplus, SALE_BATCHES[Item.WHEAT])
        plan.append(
            active
            & (wheat_count > 0)
            & (((ratios[..., Item.WHEAT] >= 1.0) & post_town_demand) | pressure),
            MarketOp.SELL,
            item=Item.WHEAT,
            count=wheat_count,
        )

        # BUY_LAND was intentionally emitted before normal market sales so it
        # cannot be pushed past the ten-order cap. Keep this mask for downstream
        # cash budgeting.
        land_buy = early_land_buy

        # Expansion is processed before replenishing feed so a routine Wheat
        # purchase cannot consume the cash earmarked for the day's land unlock.
        wheat_owned = shed[..., Item.WHEAT] + carried_wheat
        wheat_shortfall = np.maximum(0, intent.wheat_reserve - wheat_owned)
        plan.append(
            active & (wheat_shortfall > 0),
            MarketOp.BUY_PRODUCT,
            item=Item.WHEAT,
            # Refill over multiple turns so feed cannot consume the entire
            # expansion bankroll before the premium seed order is processed.
            count=np.minimum(wheat_shortfall, 4),
        )

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
        # Once the first expansion date arrives, do not let livestock consume
        # its cash while BUY_LAND is still unaffordable.
        expansion_ready = active & (
            (
                (day < 6)
                | (unlocked >= 2)
                | land_buy
            )
            & ~reserve_for_land
        )
        land_cost = np.choose(np.minimum(unlocked, 3), (0, 1_000, 2_000, 4_000))
        # Sheep expansion is more expensive to maintain than its purchase
        # price suggests. Preserve a feed-and-seed buffer in Yarn branches;
        # otherwise seat-order price differences can spend the bank down to a
        # few coins and cascade into every animal escaping.
        animal_cash_reserve = np.where(
            intent.target_animal_counts[..., 2] >= 8,
            800,
            np.where(day < 6, 0, 200),
        )
        scheduled_land_reserve = np.where(
            reserve_for_land,
            next_land_cost,
            0,
        )

        budget = np.maximum(
            0,
            money
            - animal_cash_reserve
            - scheduled_land_reserve
            - land_buy * land_cost,
        ).astype(np.int64)
        buy_sheep = np.minimum(missing_sheep, budget // 500)
        budget -= buy_sheep * 500
        buy_cows = np.minimum(missing_cows, budget // 400)
        establishing_second_field = (day >= 7) & (day <= 9)
        plan.append(
            expansion_ready & ~establishing_second_field & (buy_sheep > 0),
            MarketOp.BUY_ANIMAL,
            item=Item.SHEEP,
            count=buy_sheep,
        )
        plan.append(
            expansion_ready & ~establishing_second_field & (buy_cows > 0),
            MarketOp.BUY_ANIMAL,
            item=Item.COW,
            count=buy_cows,
        )
        # Secure a minimum useful workforce before seed orders.  Remaining
        # hires come after seeds so neither survival work nor planting can
        # monopolize the ten available market-order slots.
        missing_hands = np.maximum(0, intent.target_hands - hands)
        can_hire = (intent.target_hands > 0) & (money >= 12)
        essential_hands = 5
        for hire_index in range(
            min(essential_hands, int(intent.target_hands.max(initial=0)))
        ):
            plan.append(can_hire & (missing_hands > hire_index), MarketOp.HIRE)

        # Buy exactly the missing crop stock. The unit executor will plant only
        # seeds already visible at the start of a turn.  Crops one day from
        # harvest also get replacement stock, eliminating the post-harvest day
        # where their tiles previously sat empty.  Seed orders precede HIRE so
        # the ten-order market cap cannot starve planting indefinitely.
        crop_age = np.rint(
            tiles[..., 14] * self.episode_days
        ).astype(np.int64)
        crop_channels = tiles[..., 9:14] > 0.5
        # Only recurring short crops need replacement stock. The replay plants
        # one Strawberry and one Melon cohort rather than rolling them forever.
        replacement_ages = np.asarray((3, 2, 99, 99, 99), dtype=np.int64)
        replacement_seeds = (
            crop_channels & (crop_age[..., None] >= replacement_ages)
        ).sum(axis=(2, 3), dtype=np.int64)
        market_crop_targets = intent.target_crop_counts.copy()
        total_missing = np.maximum(
            0,
            market_crop_targets.sum(axis=-1)
            + replacement_seeds.sum(axis=-1)
            - crops.sum(axis=-1)
            - seeds.sum(axis=-1),
        )
        wheat_missing = np.maximum(
            0,
            market_crop_targets[..., Item.WHEAT]
            + replacement_seeds[..., Item.WHEAT]
            - crops[..., Item.WHEAT]
            - seeds[..., Item.WHEAT],
        )
        cash_seed_missing = np.maximum(0, total_missing - wheat_missing)
        for crop in (
            Item.WHEAT,
            Item.CARROT,
            Item.TOMATO,
            Item.STRAWBERRY,
            Item.MELON,
        ):
            if crop == Item.WHEAT:
                missing = wheat_missing
            else:
                preferred_deficit = np.maximum(
                    0,
                    market_crop_targets[..., crop] - crops[..., crop],
                )
                missing = np.minimum(cash_seed_missing, preferred_deficit)
                cash_seed_missing -= missing
            batch_limit = np.full(missing.shape, SEED_BUY_BATCHES[crop])
            if crop == Item.WHEAT:
                batch_limit = np.where(day >= 11, 12, batch_limit)
            elif crop == Item.STRAWBERRY:
                batch_limit = np.where(
                    (day >= 7) & (day <= 9), 9, batch_limit
                )
                batch_limit = np.where(day >= 11, 16, batch_limit)
            plan.append(
                active & (missing > 0),
                MarketOp.BUY_SEED,
                item=crop,
                count=np.minimum(missing, batch_limit),
            )

        # Field establishment takes precedence over adding livestock. A large
        # animal purchase here used to consume the cash for Strawberry seeds,
        # leaving the newly unlocked quadrant idle for most of day eight.
        plan.append(
            expansion_ready & establishing_second_field & (buy_sheep > 0),
            MarketOp.BUY_ANIMAL,
            item=Item.SHEEP,
            count=buy_sheep,
        )
        plan.append(
            expansion_ready & establishing_second_field & (buy_cows > 0),
            MarketOp.BUY_ANIMAL,
            item=Item.COW,
            count=buy_cows,
        )

        # Daily labor must be present early enough to finish survival and field
        # work. HIRE has no count argument and consumes one market slot per hand.
        for hire_index in range(
            essential_hands, int(intent.target_hands.max(initial=0))
        ):
            plan.append(can_hire & (missing_hands > hire_index), MarketOp.HIRE)

def build_policy(
    config: RuleConfig | None = None,
    *,
    use_opening: bool = True,
    liquidation_days: int = 1,
    scheduler_mode: str = "route",
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
        scheduler_mode=scheduler_mode,
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
