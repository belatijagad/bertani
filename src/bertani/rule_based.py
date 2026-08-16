"""Vectorized scaffolding for hierarchical rule-based Kaggriculture agents.

The strategic layer operates on whole NumPy batches.  The executor converts
those intentions into the fixed action tensors accepted by :class:`VecEnv`.
Path assignment and other inherently ragged decisions can be added to the
executor without changing the planner interface used by a future learned
policy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from .vec_env import Batch, Item, MarketOp, UnitOp


Int8Array = NDArray[np.int8]
Int64Array = NDArray[np.int64]
Float64Array = NDArray[np.float64]


class RulePhase(IntEnum):
    """Coarse phase consumed by both rules and future learned policies."""

    OPENING = 0
    MIDGAME = 1
    LIQUIDATION = 2


@dataclass(frozen=True, slots=True)
class RuleConfig:
    """Game scales and initial strategic targets for the rule planner."""

    episode_steps: int = 720
    turns_per_day: int = 24
    starting_money: int = 3_000
    shed_capacity: int = 100
    liquidation_days: int = 3
    opening_crop_targets: tuple[int, int, int, int, int] = (7, 0, 0, 0, 12)
    # GOOSE, COW, SHEEP order.
    opening_animal_targets: tuple[int, int, int] = (0, 2, 2)

    def __post_init__(self) -> None:
        if self.episode_steps < 1:
            raise ValueError("episode_steps must be positive")
        if self.turns_per_day < 1:
            raise ValueError("turns_per_day must be positive")
        if self.starting_money < 1:
            raise ValueError("starting_money must be positive")
        if self.shed_capacity < 1:
            raise ValueError("shed_capacity must be positive")
        if self.liquidation_days < 0:
            raise ValueError("liquidation_days cannot be negative")


@dataclass(frozen=True, slots=True)
class RuleFeatures:
    """Dense batch features derived from the stable observation layout."""

    step: Int64Array
    day: Int64Array
    hour: Int64Array
    money: Float64Array
    crop_counts: Int64Array
    animal_counts: Int64Array
    shed: Int64Array
    seeds: Int64Array
    market_price_ratios: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class StrategicIntent:
    """High-level decisions independent of movement and action encoding."""

    phase: Int8Array
    target_hands: Int64Array
    cash_reserve: Float64Array
    wheat_reserve: Int64Array
    target_crop_counts: Int64Array
    target_animal_counts: Int64Array
    liquidate: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class RuleActions:
    """Reusable action buffers compatible with :meth:`VecEnv.step`."""

    unit_actions: Int64Array
    market_actions: Int64Array
    market_lengths: Int64Array


class VectorRulePolicy:
    """Batch-first rule planner with a conservative masked executor.

    The initial executor handles useful operations available on a unit's
    current tile.  Movement, opening-book choreography, inventory logistics,
    and market construction are deliberate extension points; strategic intent
    is already represented independently so those additions do not couple the
    rules to the simulator or to a neural-network implementation.
    """

    _LOCAL_OPERATION_PRIORITY = np.array(
        [
            1,  # PASS
            0,  # NORTH
            0,  # SOUTH
            0,  # EAST
            0,  # WEST
            0,  # PICKUP
            50,  # DROP
            0,  # PLACE
            0,  # PLANT
            90,  # WATER
            100,  # HARVEST
            0,  # FERTILIZE
            0,  # DIG -- never destroy a crop through a generic priority
            0,  # BUILD_COOP
            0,  # BUILD_PASTURE
            95,  # FEED
            80,  # COLLECT_FERTILIZER
            70,  # CARE
        ],
        dtype=np.int16,
    )

    def __init__(
        self,
        config: RuleConfig | None = None,
        intent_planner: Callable[[Batch], StrategicIntent] | None = None,
    ) -> None:
        self.config = config or RuleConfig()
        self.intent_planner = intent_planner
        self._shape: tuple[int, int, int, int] | None = None
        self._actions: RuleActions | None = None

    def extract_features(self, batch: Batch) -> RuleFeatures:
        """Extract planner features with batch-wide NumPy operations."""

        config = self.config
        views = batch.observation_views
        global_features = views.global_features
        own_farms = views.farms[:, :, 0]
        own_tiles = views.tiles[:, :, 0]

        last_step = max(1, config.episode_steps - 1)
        step = np.rint(global_features[..., 0] * last_step).astype(np.int64)
        day = step // config.turns_per_day
        hour = step % config.turns_per_day
        money = own_farms[..., 0].astype(np.float64) * config.starting_money

        # Tile channels 9..13 are WHEAT..MELON crop one-hots. Occupied animal
        # kind channels 6..8 are GOOSE, COW, SHEEP.
        crop_counts = np.rint(own_tiles[..., 9:14].sum(axis=(2, 3))).astype(
            np.int64
        )
        animal_counts = np.rint(own_tiles[..., 6:9].sum(axis=(2, 3))).astype(
            np.int64
        )
        shed = np.rint(views.private[..., :12] * config.shed_capacity).astype(
            np.int64
        )
        seeds = np.rint(views.private[..., 12:17] * 10).astype(np.int64)
        market_price_ratios = global_features[..., 5:22:2]

        return RuleFeatures(
            step=step,
            day=day,
            hour=hour,
            money=money,
            crop_counts=crop_counts,
            animal_counts=animal_counts,
            shed=shed,
            seeds=seeds,
            market_price_ratios=market_price_ratios,
        )

    def plan(self, batch: Batch) -> StrategicIntent:
        """Produce high-level intent for every environment and player."""

        if self.intent_planner is not None:
            return self.intent_planner(batch)
        features = self.extract_features(batch)
        return self._plan_features(features)

    def _plan_features(self, features: RuleFeatures) -> StrategicIntent:
        shape = features.step.shape
        total_days = (
            self.config.episode_steps + self.config.turns_per_day - 1
        ) // self.config.turns_per_day
        liquidation_start = max(0, total_days - self.config.liquidation_days)

        phase = np.full(shape, RulePhase.MIDGAME, dtype=np.int8)
        phase[features.day < 3] = RulePhase.OPENING
        phase[features.day >= liquidation_start] = RulePhase.LIQUIDATION

        target_hands = np.full(shape, 5, dtype=np.int64)
        opening_hands = np.array([5, 0, 4], dtype=np.int64)
        opening = features.day < opening_hands.size
        target_hands[opening] = opening_hands[features.day[opening]]
        target_hands[phase == RulePhase.LIQUIDATION] = 0

        cash_reserve = np.full(shape, 1_000.0, dtype=np.float64)
        cash_reserve[phase != RulePhase.MIDGAME] = 0.0
        wheat_reserve = 2 * animal_counts_total(features.animal_counts)

        target_crop_counts = np.broadcast_to(
            np.asarray(self.config.opening_crop_targets, dtype=np.int64),
            (*shape, 5),
        ).copy()
        target_animal_counts = np.broadcast_to(
            np.asarray(self.config.opening_animal_targets, dtype=np.int64),
            (*shape, 3),
        ).copy()
        liquidate = phase == RulePhase.LIQUIDATION

        return StrategicIntent(
            phase=phase,
            target_hands=target_hands,
            cash_reserve=cash_reserve,
            wheat_reserve=wheat_reserve,
            target_crop_counts=target_crop_counts,
            target_animal_counts=target_animal_counts,
            liquidate=liquidate,
        )

    def act(self, batch: Batch, max_orders: int = 10) -> RuleActions:
        """Return legal local maintenance actions for an entire batch.

        ``plan`` is called even though the first executor only consumes the
        liquidation flag. This keeps one stable seam for upcoming opening,
        routing, and economy rules and for a later neural strategy module.
        """

        intent = self.plan(batch)
        features = self.extract_features(batch)
        actions = self._action_buffers(batch, max_orders)
        actions.unit_actions.fill(0)
        actions.market_actions.fill(0)
        actions.market_lengths.fill(0)

        masks = batch.mask_views.unit_ops
        scores = masks * self._LOCAL_OPERATION_PRIORITY
        chosen = np.argmax(scores, axis=-1)
        chosen[~batch.active_units] = int(UnitOp.PASS)
        actions.unit_actions[..., 0] = chosen

        self._append_liquidation_sales(features, intent, actions)
        return actions

    def _action_buffers(self, batch: Batch, max_orders: int) -> RuleActions:
        n, players, units = batch.active_units.shape
        shape = (n, players, units, max_orders)
        if self._actions is None or self._shape != shape:
            self._actions = RuleActions(
                unit_actions=np.zeros((n, players, units, 3), dtype=np.int64),
                market_actions=np.zeros(
                    (n, players, max_orders, 3), dtype=np.int64
                ),
                market_lengths=np.zeros((n, players), dtype=np.int64),
            )
            self._shape = shape
        return self._actions

    def _append_liquidation_sales(
        self,
        features: RuleFeatures,
        intent: StrategicIntent,
        actions: RuleActions,
    ) -> None:
        """Serialize ragged sell orders after vectorized eligibility checks."""

        shed = features.shed
        # Products are WHEAT through FERTILIZER (item IDs 0..8). Animals cannot
        # be sold directly. This tiny ragged loop is intentionally isolated in
        # the executor; the expensive state evaluation remains batched.
        sellable = (shed[..., :9] > 0) & intent.liquidate[..., None]
        for environment, player in np.argwhere(sellable.any(axis=-1)):
            order = 0
            for item in np.flatnonzero(sellable[environment, player]):
                if order >= actions.market_actions.shape[2]:
                    break
                actions.market_actions[environment, player, order] = (
                    MarketOp.SELL,
                    item,
                    shed[environment, player, item],
                )
                order += 1
            actions.market_lengths[environment, player] = order


def animal_counts_total(animal_counts: Int64Array) -> Int64Array:
    """Return total livestock per environment/player without Python loops."""

    return animal_counts.sum(axis=-1, dtype=np.int64)


__all__ = [
    "RuleActions",
    "RuleConfig",
    "RuleFeatures",
    "RulePhase",
    "StrategicIntent",
    "VectorRulePolicy",
]
