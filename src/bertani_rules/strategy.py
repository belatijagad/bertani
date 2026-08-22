"""Public Python strategy interface backed by Bertani's native rule runtime.

Strategy authors only fill high-level targets for a whole batch. Rust keeps
ownership of feature extraction, farm-task generation, routing, workforce
assignment, action encoding, market execution, and simulation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from bertani.opening import OpeningController
from bertani.rule_based import (
    RuleConfig,
    RuleFeatures,
    RulePhase,
    StrategicIntent,
    VectorRulePolicy,
    extract_rule_features,
)
from bertani.vec_env import Batch, Item

from .agent import (
    OPENING_BOOK,
    EconomyMarketRule,
    FarmTaskRule,
    TerritorialWorkforcePlanner,
)

Int8Array = NDArray[np.int8]
Int64Array = NDArray[np.int64]
Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class RulePlan:
    """Mutable, reusable high-level targets filled by a Python strategy.

    Arrays use ``[environment, player]`` as their leading axes. Crop targets
    use the five crop ``Item`` IDs; animal targets use GOOSE/COW/SHEEP order.
    A plan is reused on every turn, so strategies must overwrite any targets
    they want to retain.
    """

    phase: Int8Array
    target_hands: Int64Array
    cash_reserve: Float64Array
    wheat_reserve: Int64Array
    target_crop_counts: Int64Array
    target_animal_counts: Int64Array
    liquidate: BoolArray

    @classmethod
    def allocate(cls, shape: tuple[int, int]) -> RulePlan:
        """Allocate one plan for a vector batch shape."""

        return cls(
            phase=np.empty(shape, dtype=np.int8),
            target_hands=np.empty(shape, dtype=np.int64),
            cash_reserve=np.empty(shape, dtype=np.float64),
            wheat_reserve=np.empty(shape, dtype=np.int64),
            target_crop_counts=np.empty((*shape, 5), dtype=np.int64),
            target_animal_counts=np.empty((*shape, 3), dtype=np.int64),
            liquidate=np.empty(shape, dtype=np.bool_),
        )

    def reset(self, features: RuleFeatures, config: RuleConfig) -> None:
        """Reset all targets to safe defaults before calling user code."""

        self.phase.fill(RulePhase.MIDGAME)
        self.target_hands.fill(0)
        self.cash_reserve.fill(0.0)
        self.wheat_reserve.fill(0)
        self.target_crop_counts.fill(0)
        self.target_animal_counts.fill(0)
        self.liquidate.fill(False)

        if config.liquidation_days:
            total_days = (
                config.episode_steps + config.turns_per_day - 1
            ) // config.turns_per_day
            liquidation_start = max(0, total_days - config.liquidation_days)
            closing = features.day >= liquidation_start
            self.phase[closing] = RulePhase.LIQUIDATION
            self.liquidate[closing] = True

    def crop(self, item: Item) -> Int64Array:
        """Return the writable target array for one crop."""

        index = int(item)
        if not int(Item.WHEAT) <= index <= int(Item.MELON):
            raise ValueError(f"{item.name} is not a crop")
        return self.target_crop_counts[..., index]

    def animal(self, item: Item) -> Int64Array:
        """Return the writable target array for one animal."""

        index = int(item) - int(Item.GOOSE)
        if not 0 <= index < 3:
            raise ValueError(f"{item.name} is not an animal")
        return self.target_animal_counts[..., index]

    def as_intent(self) -> StrategicIntent:
        """Expose this plan through the native runtime's stable intent type."""

        return StrategicIntent(
            phase=self.phase,
            target_hands=self.target_hands,
            cash_reserve=self.cash_reserve,
            wheat_reserve=self.wheat_reserve,
            target_crop_counts=self.target_crop_counts,
            target_animal_counts=self.target_animal_counts,
            liquidate=self.liquidate,
        )


class RuleStrategy(Protocol):
    """A batch-first Python strategy.

    Implement ``plan(features, plan)`` and mutate ``plan`` in place. Avoid
    loops over environments: NumPy masks keep one callback fast for large
    batches.
    """

    def plan(self, features: RuleFeatures, plan: RulePlan) -> None:
        """Fill high-level targets for the current batch."""


StrategyFunction = Callable[[RuleFeatures, RulePlan], None]


class PythonRulePlanner:
    """Adapt a friendly Python strategy to ``VectorRulePolicy``.

    Feature and plan buffers are retained across turns. The strategy callback
    runs once per vector batch, rather than once per game or worker.
    """

    def __init__(
        self,
        strategy: RuleStrategy | StrategyFunction,
        config: RuleConfig | None = None,
    ) -> None:
        self.strategy = strategy
        self.config = config or RuleConfig()
        self.last_plan: RulePlan | None = None

    def __call__(self, batch: Batch) -> StrategicIntent:
        features = extract_rule_features(batch, self.config)
        return self.from_features(batch, features)

    def from_features(self, batch: Batch, features: RuleFeatures) -> StrategicIntent:
        del batch
        shape = features.step.shape
        if self.last_plan is None or self.last_plan.phase.shape != shape:
            self.last_plan = RulePlan.allocate(shape)
        self.last_plan.reset(features, self.config)

        callback = getattr(self.strategy, "plan", self.strategy)
        if not callable(callback):
            raise TypeError("strategy must be callable or define plan(features, plan)")
        callback(features, self.last_plan)
        return self.last_plan.as_intent()


def build_python_policy(
    strategy: RuleStrategy | StrategyFunction,
    config: RuleConfig | None = None,
    *,
    use_current_opening: bool = False,
    liquidation_days: int = 1,
    profile: bool = False,
) -> VectorRulePolicy:
    """Build a Python-authored strategy on the native rule runtime.

    ``use_current_opening`` opts into the existing 72-turn competitive opening.
    Leave it disabled while developing an independent opening strategy.
    """

    if liquidation_days < 0:
        raise ValueError("liquidation_days cannot be negative")
    resolved = replace(
        config or RuleConfig(),
        liquidation_days=liquidation_days,
        opening_crop_targets=(7, 0, 0, 0, 12) if use_current_opening else (0,) * 5,
        opening_animal_targets=(0, 10, 4) if use_current_opening else (0,) * 3,
    )
    return VectorRulePolicy(
        resolved,
        intent_planner=PythonRulePlanner(strategy, resolved),
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
            if use_current_opening
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
    "PythonRulePlanner",
    "RulePlan",
    "RuleStrategy",
    "StrategyFunction",
    "build_python_policy",
]
