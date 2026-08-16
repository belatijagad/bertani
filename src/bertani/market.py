"""Ordered market-plan abstractions for rule-based policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from .vec_env import Batch, Item, MarketOp

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
        counts = np.broadcast_to(count, mask.shape)
        for environment, player in np.argwhere(mask):
            slot = self.lengths[environment, player]
            if slot >= self.max_orders:
                self.overflow[environment, player] = True
                continue
            self.actions[environment, player, slot] = (
                operation,
                item,
                counts[environment, player],
            )
            self.lengths[environment, player] += 1

    def _validate_mask(self, mask: NDArray[np.bool_]) -> None:
        if mask.shape != self.lengths.shape:
            raise ValueError(f"market plan mask must have shape {self.lengths.shape}")


class MarketRule(Protocol):
    """Extension point for rules that append orders or reserve resources."""

    def propose(
        self,
        batch: Batch,
        intent: StrategicIntent,
        plan: MarketPlanBatch,
    ) -> None:
        """Extend an ordered plan without constructing tensor rows directly."""


__all__ = ["MarketPlanBatch", "MarketRule"]
