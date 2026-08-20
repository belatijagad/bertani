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
import time

import numpy as np
from numpy.typing import NDArray

try:
    from ._rust import extract_rule_features as _native_extract_rule_features
except (ImportError, ModuleNotFoundError):
    _native_extract_rule_features = None

from .market import MarketPlanBatch, MarketRule
from .opening import OpeningController, OpeningDiagnostics
from .tasks import (
    TaskAssignments,
    TaskBatch,
    TaskExecutor,
    TaskRule,
    TaskScheduler,
    WorkforcePlanner,
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
    liquidation_days: int = 0
    opening_crop_targets: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
    # GOOSE, COW, SHEEP order.
    opening_animal_targets: tuple[int, int, int] = (0, 0, 0)
    town_shop_unlock_interval: int = 3
    town_shop_sell_interval: int = 4
    town_center_sell_interval: int = 24

    def __post_init__(self) -> None:
        if self.episode_steps < 1:
            raise ValueError("episode_steps must be positive")
        if self.turns_per_day < 1:
            raise ValueError("turns_per_day must be positive")
        if self.starting_money < 1:
            raise ValueError("starting_money must be positive")
        if self.shed_capacity < 1:
            raise ValueError("shed_capacity must be positive")
        if self.town_shop_unlock_interval < 1:
            raise ValueError("town_shop_unlock_interval must be positive")
        if self.town_shop_sell_interval < 1:
            raise ValueError("town_shop_sell_interval must be positive")
        if self.town_center_sell_interval < 1:
            raise ValueError("town_center_sell_interval must be positive")
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
    shop_counts: Int64Array
    opponent_crop_counts: Int64Array
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


def extract_rule_features(
    batch: Batch,
    config: RuleConfig,
    out: RuleFeatures | None = None,
) -> RuleFeatures:
    """Extract dense rule features through the native typed reduction kernel.

    ``out`` allows the vector policy to reuse every materialized feature
    buffer across turns, avoiding Python allocation and reduction overhead.
    """

    if _native_extract_rule_features is None:
        raise RuntimeError(
            "native rule feature extraction requires the bertani._rust extension"
        )

    views = batch.observation_views
    shape = batch.active_units.shape[:2]
    if out is None or out.step.shape != shape:
        out = RuleFeatures(
            step=np.empty(shape, dtype=np.int64),
            day=np.empty(shape, dtype=np.int64),
            hour=np.empty(shape, dtype=np.int64),
            money=np.empty(shape, dtype=np.float64),
            crop_counts=np.empty((*shape, 5), dtype=np.int64),
            animal_counts=np.empty((*shape, 3), dtype=np.int64),
            shed=np.empty((*shape, 12), dtype=np.int64),
            seeds=np.empty((*shape, 5), dtype=np.int64),
            shop_counts=np.empty((*shape, 8), dtype=np.int64),
            opponent_crop_counts=np.empty((*shape, 5), dtype=np.int64),
            market_price_ratios=np.empty((*shape, 9), dtype=np.float32),
        )

    _native_extract_rule_features(
        views.global_features,
        views.farms,
        views.tiles,
        views.private,
        out.step,
        out.day,
        out.hour,
        out.money,
        out.crop_counts,
        out.animal_counts,
        out.shed,
        out.seeds,
        out.shop_counts,
        out.opponent_crop_counts,
        out.market_price_ratios,
        config.episode_steps,
        config.turns_per_day,
        config.starting_money,
        config.shed_capacity,
    )
    return out


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
        opening_controller: OpeningController | None = None,
        task_rules: tuple[TaskRule, ...] | None = None,
        market_rules: tuple[MarketRule, ...] | None = None,
        workforce_planner: WorkforcePlanner | None = None,
        profile: bool = False,
    ) -> None:
        self.config = config or RuleConfig()
        self.profile = profile
        self.profile_ns: dict[str, int] = {
            "opening": 0,
            "features": 0,
            "intent": 0,
            "maintenance_tasks": 0,
            "production_tasks": 0,
            "farm_tasks": 0,
            "other_tasks": 0,
            "workforce": 0,
            "scheduler": 0,
            "executor": 0,
            "market": 0,
        }
        self.intent_planner = intent_planner
        self.opening_controller = opening_controller
        self.last_opening_diagnostics: OpeningDiagnostics | None = None
        self.task_rules = () if task_rules is None else task_rules
        self.market_rules = () if market_rules is None else market_rules
        self.workforce_planner = workforce_planner
        self._task_scheduler: TaskScheduler | None = None
        self._task_executor: TaskExecutor | None = None
        self.last_tasks: TaskBatch | None = None
        self.last_assignments: TaskAssignments | None = None
        self._shape: tuple[int, int, int, int] | None = None
        self._actions: RuleActions | None = None
        self._features: RuleFeatures | None = None
        self.last_market_plan: MarketPlanBatch | None = None

    def extract_features(self, batch: Batch) -> RuleFeatures:
        """Extract planner features with batch-wide NumPy operations."""

        self._features = extract_rule_features(batch, self.config, self._features)
        return self._features

    def plan(self, batch: Batch) -> StrategicIntent:
        """Produce high-level intent for every environment and player."""

        if self.intent_planner is not None:
            return self.intent_planner(batch)
        shape = batch.active_units.shape[:2]
        return StrategicIntent(
            phase=np.full(shape, RulePhase.MIDGAME, dtype=np.int8),
            target_hands=np.zeros(shape, dtype=np.int64),
            cash_reserve=np.zeros(shape, dtype=np.float64),
            wheat_reserve=np.zeros(shape, dtype=np.int64),
            target_crop_counts=np.zeros((*shape, 5), dtype=np.int64),
            target_animal_counts=np.zeros((*shape, 3), dtype=np.int64),
            liquidate=np.zeros(shape, dtype=np.bool_),
        )

    def act(
        self,
        batch: Batch,
        max_orders: int = 10,
        seat_mask: NDArray[np.bool_] | None = None,
    ) -> RuleActions:
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
            started = time.perf_counter_ns() if self.profile else 0
            self.last_opening_diagnostics = self.opening_controller.apply(
                batch,
                actions.unit_actions,
                actions.market_actions,
                actions.market_lengths,
            )
            if self.profile:
                self.profile_ns["opening"] += time.perf_counter_ns() - started
            self.last_tasks = None
            self.last_assignments = None
            return actions

        started = time.perf_counter_ns() if self.profile else 0
        features = self.extract_features(batch)
        if self.profile:
            self.profile_ns["features"] += time.perf_counter_ns() - started

        started = time.perf_counter_ns() if self.profile else 0
        planner_from_features = getattr(
            self.intent_planner, "from_features", None
        )
        if planner_from_features is not None:
            intent = planner_from_features(batch, features)
        else:
            intent = self.plan(batch)
        if self.profile:
            self.profile_ns["intent"] += time.perf_counter_ns() - started

        # Build the economic plan before routing so the tactical layer can use
        # its purchases as next-turn staging hints. Unit actions still resolve
        # before market orders, so current-turn tasks never consume them.
        started = time.perf_counter_ns() if self.profile else 0
        self._append_liquidation_sales(
            features, intent, self.last_market_plan, seat_mask=seat_mask
        )
        for rule in self.market_rules:
            masked_propose = getattr(rule, "propose_masked", None)
            if seat_mask is not None and masked_propose is not None:
                masked_propose(batch, intent, self.last_market_plan, seat_mask)
            else:
                rule.propose(batch, intent, self.last_market_plan)
        if self.profile:
            self.profile_ns["market"] += time.perf_counter_ns() - started

        tasks = self._task_buffers(batch)
        tasks.clear()
        for rule in self.task_rules:
            started = time.perf_counter_ns() if self.profile else 0
            with_market_plan = getattr(rule, "propose_with_market_plan", None)
            masked_with_market_plan = getattr(
                rule, "propose_with_market_plan_masked", None
            )
            masked_propose = getattr(rule, "propose_masked", None)
            if seat_mask is not None and masked_with_market_plan is not None:
                masked_with_market_plan(
                    batch, intent, tasks, self.last_market_plan, seat_mask
                )
            elif with_market_plan is not None:
                with_market_plan(batch, intent, tasks, self.last_market_plan)
            elif seat_mask is not None and masked_propose is not None:
                masked_propose(batch, intent, tasks, seat_mask)
            else:
                rule.propose(batch, intent, tasks)
            if self.profile:
                key = getattr(rule, "profile_key", "other_tasks")
                self.profile_ns.setdefault(key, 0)
                self.profile_ns[key] += time.perf_counter_ns() - started

        assert self._task_scheduler is not None
        assert self._task_executor is not None
        started = time.perf_counter_ns() if self.profile else 0
        if self.workforce_planner is None:
            workforce = None
        else:
            masked_workforce = getattr(self.workforce_planner, "plan_masked", None)
            if seat_mask is not None and masked_workforce is not None:
                workforce = masked_workforce(batch, intent, tasks, seat_mask)
            else:
                workforce = self.workforce_planner(batch, intent, tasks)
        if self.profile:
            self.profile_ns["workforce"] += time.perf_counter_ns() - started

        started = time.perf_counter_ns() if self.profile else 0
        assignments = self._task_scheduler.assign_and_execute(
            batch,
            tasks,
            actions.unit_actions,
            workforce,
            seat_mask=seat_mask,
        )
        if self.profile:
            # V12 fuses deterministic task execution into the native scheduler
            # call. Keep the existing key so historical pit profiles remain
            # directly comparable; executor should now stay at zero.
            self.profile_ns["scheduler"] += time.perf_counter_ns() - started
        self.last_tasks = tasks
        self.last_assignments = assignments

        if self.opening_controller is not None:
            started = time.perf_counter_ns() if self.profile else 0
            self.last_opening_diagnostics = self.opening_controller.apply(
                batch,
                actions.unit_actions,
                actions.market_actions,
                actions.market_lengths,
            )
            if self.profile:
                self.profile_ns["opening"] += time.perf_counter_ns() - started
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
                board_size,
                shed_capacity=self.config.shed_capacity,
                episode_steps=self.config.episode_steps,
                turns_per_day=self.config.turns_per_day,
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
        *,
        seat_mask: NDArray[np.bool_] | None = None,
    ) -> None:
        """Serialize ragged sell orders after vectorized eligibility checks."""

        shed = features.shed
        # Products are WHEAT through FERTILIZER (item IDs 0..8). Animals cannot
        # be sold directly. This tiny ragged loop is intentionally isolated in
        # the executor; the expensive state evaluation remains batched.
        sellable = (shed[..., :9] > 0) & intent.liquidate[..., None]
        if seat_mask is not None:
            sellable &= seat_mask[..., None]
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
    "extract_rule_features",
]
