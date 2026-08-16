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

from .market import MarketPlanBatch, MarketRule
from .opening import OpeningController, OpeningDiagnostics
from .tasks import (
    MaintenanceTaskRule,
    TaskAssignments,
    TaskBatch,
    TaskExecutor,
    TaskRule,
    TaskScheduler,
)
from .vec_env import Batch, Item, MarketOp


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

    def __init__(
        self,
        config: RuleConfig | None = None,
        intent_planner: Callable[[Batch], StrategicIntent] | None = None,
        use_opening: bool = True,
        task_rules: tuple[TaskRule, ...] | None = None,
        market_rules: tuple[MarketRule, ...] | None = None,
    ) -> None:
        self.config = config or RuleConfig()
        self.intent_planner = intent_planner
        self.opening_controller = (
            OpeningController(self.config.episode_steps) if use_opening else None
        )
        self.last_opening_diagnostics: OpeningDiagnostics | None = None
        self.task_rules = (
            (
                MaintenanceTaskRule(
                    turns_per_day=self.config.turns_per_day,
                    shed_capacity=self.config.shed_capacity,
                ),
            )
            if task_rules is None
            else task_rules
        )
        self.market_rules = () if market_rules is None else market_rules
        self._task_scheduler: TaskScheduler | None = None
        self._task_executor: TaskExecutor | None = None
        self.last_tasks: TaskBatch | None = None
        self.last_assignments: TaskAssignments | None = None
        self._shape: tuple[int, int, int, int] | None = None
        self._actions: RuleActions | None = None
        self.last_market_plan: MarketPlanBatch | None = None

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

        Opening-only batches take a fast path around task generation. Outside
        the opening, rules propose tasks, the scheduler assigns units, and the
        executor emits masked movement or interaction actions.
        """

        actions = self._action_buffers(batch, max_orders)
        actions.unit_actions.fill(0)
        assert self.last_market_plan is not None
        self.last_market_plan.clear()

        if (
            self.opening_controller is not None
            and self.opening_controller.active_mask(batch).all()
        ):
            self.last_opening_diagnostics = self.opening_controller.apply(
                batch,
                actions.unit_actions,
                actions.market_actions,
                actions.market_lengths,
            )
            self.last_tasks = None
            self.last_assignments = None
            return actions

        intent = self.plan(batch)
        features = self.extract_features(batch)

        tasks = self._task_buffers(batch)
        tasks.clear()
        for rule in self.task_rules:
            rule.propose(batch, intent, tasks)
        assert self._task_scheduler is not None
        assert self._task_executor is not None
        assignments = self._task_scheduler.assign(batch, tasks)
        self._task_executor.execute(
            batch, tasks, assignments, actions.unit_actions
        )
        self.last_tasks = tasks
        self.last_assignments = assignments

        for rule in self.market_rules:
            rule.propose(batch, intent, self.last_market_plan)
        self._append_liquidation_sales(features, intent, self.last_market_plan)
        if self.opening_controller is not None:
            self.last_opening_diagnostics = self.opening_controller.apply(
                batch,
                actions.unit_actions,
                actions.market_actions,
                actions.market_lengths,
            )
        else:
            self.last_opening_diagnostics = None
        return actions

    def _task_buffers(self, batch: Batch) -> TaskBatch:
        n, players, _ = batch.active_units.shape
        board_size = batch.observation_views.tiles.shape[3]
        expected_shape = (n, players, board_size * board_size + 12)
        if self.last_tasks is None or self.last_tasks.active.shape != expected_shape:
            self.last_tasks = TaskBatch.allocate(n, players, board_size)
            self._task_scheduler = TaskScheduler(
                board_size, shed_capacity=self.config.shed_capacity
            )
            self._task_executor = TaskExecutor(board_size)
        return self.last_tasks

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
            self.last_market_plan = MarketPlanBatch(
                actions=self._actions.market_actions,
                lengths=self._actions.market_lengths,
                reserved_cash=np.zeros((n, players), dtype=np.float64),
                reserved_items=np.zeros((n, players, 12), dtype=np.int64),
                overflow=np.zeros((n, players), dtype=np.bool_),
            )
            self._shape = shape
        return self._actions

    def _append_liquidation_sales(
        self,
        features: RuleFeatures,
        intent: StrategicIntent,
        plan: MarketPlanBatch,
    ) -> None:
        """Serialize ragged sell orders after vectorized eligibility checks."""

        shed = features.shed
        # Products are WHEAT through FERTILIZER (item IDs 0..8). Animals cannot
        # be sold directly. This tiny ragged loop is intentionally isolated in
        # the executor; the expensive state evaluation remains batched.
        sellable = (shed[..., :9] > 0) & intent.liquidate[..., None]
        for item in range(9):
            plan.append(
                sellable[..., item],
                MarketOp.SELL,
                item=item,
                count=shed[..., item],
            )


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
