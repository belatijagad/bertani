"""Thin Python boundary for the independent Rust V16 opponent."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .._rust import NativeV16Opponent
from ..actions import ActionBatch
from ..vec_env import Batch, VecEnv
from .trace import V16Trace, load_v16_trace


class V16CacheStats(NamedTuple):
    """V16 performs no state-cache lookup; values are always zero."""

    hits: int = 0
    misses: int = 0


class V16OpponentPolicy:
    """Emit V16 actions directly from Rust-owned simulator state."""

    def __init__(self, trace: V16Trace, *, max_orders: int = 10) -> None:
        self.max_orders = max_orders
        self._native = NativeV16Opponent(
            trace.unit_actions,
            trace.market_actions,
            trace.market_lengths,
        )
        self._actions: ActionBatch | None = None
        self._shape: tuple[int, int, int] | None = None

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        max_orders: int = 10,
    ) -> V16OpponentPolicy:
        return cls(load_v16_trace(path, max_orders=max_orders), max_orders=max_orders)

    @property
    def cache_stats(self) -> V16CacheStats:
        return V16CacheStats()

    def reset(self) -> None:
        self._native.reset()

    def act(
        self,
        environment: VecEnv,
        batch: Batch,
        *,
        seats: NDArray[np.int64],
    ) -> ActionBatch:
        if seats.shape != (environment.num_envs,):
            raise ValueError(f"seats must have shape ({environment.num_envs},)")
        actions = self._buffers(batch)
        self._native.act_into(
            environment.native,
            seats,
            actions.unit_actions,
            actions.market_actions,
            actions.market_lengths,
        )
        return actions

    def _buffers(self, batch: Batch) -> ActionBatch:
        shape = batch.active_units.shape
        if self._actions is None or self._shape != shape:
            environments, players, units = shape
            self._actions = ActionBatch(
                unit_actions=np.zeros(
                    (environments, players, units, 3), dtype=np.int16
                ),
                market_actions=np.zeros(
                    (environments, players, self.max_orders, 3), dtype=np.int16
                ),
                market_lengths=np.zeros((environments, players), dtype=np.int16),
            )
            self._shape = shape
        return self._actions


__all__ = ["V16CacheStats", "V16OpponentPolicy"]
