"""Ordered market-plan abstractions for rule-based policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from .vec_env import Batch, Item, MarketOp

try:
    from ._rust import propose_rule_market as _propose_rule_market
except (ImportError, ModuleNotFoundError):
    _propose_rule_market = None


if TYPE_CHECKING:
    from .rule_based import StrategicIntent


@dataclass(frozen=True, slots=True)
class MarketPlanBatch:
    """Reusable ordered market plans with explicit resource reservations."""

    actions: NDArray[np.int64]
    lengths: NDArray[np.int64]
    reserved_cash: NDArray[np.float64]
    reserved_items: NDArray[np.int64]
    overflow: NDArray[np.bool_]

    @classmethod
    def allocate(
        cls, num_envs: int, players: int, max_orders: int
    ) -> MarketPlanBatch:
        if min(num_envs, players, max_orders) < 1:
            raise ValueError("market plan dimensions must be positive")
        return cls(
            actions=np.zeros(
                (num_envs, players, max_orders, 3), dtype=np.int64
            ),
            lengths=np.zeros((num_envs, players), dtype=np.int64),
            reserved_cash=np.zeros((num_envs, players), dtype=np.float64),
            reserved_items=np.zeros((num_envs, players, 12), dtype=np.int64),
            overflow=np.zeros((num_envs, players), dtype=np.bool_),
        )

    @property
    def max_orders(self) -> int:
        return self.actions.shape[2]

    def clear(self) -> None:
        self.actions.fill(0)
        self.lengths.fill(0)
        self.reserved_cash.fill(0)
        self.reserved_items.fill(0)
        self.overflow.fill(False)

    def reserve_cash(self, mask: NDArray[np.bool_], amount: float) -> None:
        """Raise the minimum cash reservation for selected seats."""

        self._validate_mask(mask)
        self.reserved_cash[mask] = np.maximum(self.reserved_cash[mask], amount)

    def reserve_item(
        self, mask: NDArray[np.bool_], item: int, count: int | NDArray[np.int64]
    ) -> None:
        """Raise an inventory reservation used by later market rules."""

        self._validate_mask(mask)
        if not 0 <= item < self.reserved_items.shape[-1]:
            raise ValueError("reserved item ID is outside the item domain")
        self.reserved_items[..., item][mask] = np.maximum(
            self.reserved_items[..., item][mask], np.broadcast_to(count, mask.shape)[mask]
        )

    def append(
        self,
        mask: NDArray[np.bool_],
        operation: MarketOp,
        *,
        item: int = 0,
        count: int | NDArray[np.int64] = 0,
    ) -> None:
        """Append an ordered market row to each selected seat's active prefix."""

        self._validate_mask(mask)
        if not np.any(mask):
            return

        counts = np.broadcast_to(count, mask.shape)
        environments, players = np.nonzero(mask)
        slots = self.lengths[environments, players]
        valid = slots < self.max_orders

        if np.any(~valid):
            self.overflow[
                environments[~valid],
                players[~valid],
            ] = True

        if not np.any(valid):
            return

        environments = environments[valid]
        players = players[valid]
        slots = slots[valid]
        self.actions[environments, players, slots, 0] = int(operation)
        self.actions[environments, players, slots, 1] = int(item)
        self.actions[environments, players, slots, 2] = counts[
            environments,
            players,
        ]
        self.lengths[environments, players] += 1

    def _validate_mask(self, mask: NDArray[np.bool_]) -> None:
        if mask.shape != self.lengths.shape:
            raise ValueError(f"market plan mask must have shape {self.lengths.shape}")


def propose_native_rule_market(
    batch: Batch,
    intent: StrategicIntent,
    plan: MarketPlanBatch,
    *,
    seat_mask: NDArray[np.bool_] | None = None,
    starting_money: int,
    shed_capacity: int,
    episode_steps: int,
    turns_per_day: int,
) -> None:
    """Append the current rule-based market policy through Rust.

    This is a fast backend for the hand-written rule policy. A learned market
    policy can still implement :class:`MarketRule` directly without using it.
    """
    if _propose_rule_market is None:
        raise RuntimeError(
            "native rule market requires the bertani._rust extension"
        )
    views = batch.observation_views
    shape = batch.active_units.shape[:2]
    if seat_mask is None:
        controlled_seats = np.ones(shape, dtype=np.bool_)
    else:
        if seat_mask.shape != shape:
            raise ValueError(f"seat mask must have shape {shape}")
        controlled_seats = (
            seat_mask
            if seat_mask.dtype == np.bool_ and seat_mask.flags.c_contiguous
            else np.ascontiguousarray(seat_mask, dtype=np.bool_)
        )
    _propose_rule_market(
        views.global_features,
        views.farms,
        views.tiles,
        views.units,
        views.private,
        batch.active_units,
        controlled_seats,
        np.ascontiguousarray(intent.target_hands, dtype=np.int64),
        np.ascontiguousarray(intent.wheat_reserve, dtype=np.int64),
        np.ascontiguousarray(intent.target_crop_counts, dtype=np.int64),
        np.ascontiguousarray(intent.target_animal_counts, dtype=np.int64),
        np.ascontiguousarray(intent.liquidate, dtype=np.bool_),
        plan.actions,
        plan.lengths,
        plan.overflow,
        starting_money,
        shed_capacity,
        episode_steps,
        turns_per_day,
    )


class MarketRule(Protocol):
    """Extension point for rules that append orders or reserve resources."""

    def propose(
        self,
        batch: Batch,
        intent: StrategicIntent,
        plan: MarketPlanBatch,
    ) -> None:
        """Extend an ordered plan without constructing tensor rows directly."""


__all__ = ["MarketPlanBatch", "MarketRule", "propose_native_rule_market"]
