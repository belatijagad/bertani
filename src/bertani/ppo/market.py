"""Injectable market construction for the current worker/workforce network."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..rule_based import VectorRulePolicy
from ..vec_env import Batch, MarketOp


class LearnerMarketPolicy(Protocol):
    def actions(
        self,
        batch: Batch,
        seats: NDArray[np.int64],
        target_hands: NDArray[np.int64],
        *,
        max_orders: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Return learner-only market actions and active prefix lengths."""


class WorkforceMarketPolicy:
    """Translate the workforce head into hires, optionally over an economy policy."""

    def __init__(
        self,
        base_policy: VectorRulePolicy | None = None,
        *,
        max_hires_per_turn: int = 2,
    ) -> None:
        if max_hires_per_turn < 1:
            raise ValueError("max_hires_per_turn must be positive")
        self.base_policy = base_policy
        self.max_hires_per_turn = max_hires_per_turn
        self._actions: np.ndarray | None = None
        self._lengths: np.ndarray | None = None

    def actions(
        self,
        batch: Batch,
        seats: NDArray[np.int64],
        target_hands: NDArray[np.int64],
        *,
        max_orders: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        environments = batch.active_units.shape[0]
        if seats.shape != (environments,) or target_hands.shape != (environments,):
            raise ValueError("seats and target_hands must have shape [environments]")
        self._ensure_buffers(environments, max_orders)
        assert self._actions is not None and self._lengths is not None
        self._actions.fill(0)
        self._lengths.fill(0)
        games = np.arange(environments, dtype=np.int64)

        base_actions = None
        if self.base_policy is not None:
            seat_mask = np.zeros((environments, 2), dtype=np.bool_)
            seat_mask[games, seats] = True
            base_actions = self.base_policy.act(
                batch, max_orders=max_orders, seat_mask=seat_mask
            )

        current_hands = (
            batch.active_units[games, seats].sum(axis=-1).astype(np.int64) - 1
        )
        hire_counts = np.minimum(
            np.maximum(target_hands - current_hands, 0),
            self.max_hires_per_turn,
        )
        for environment in range(environments):
            output: list[tuple[int, int, int]] = [
                (int(MarketOp.HIRE), 0, 0)
            ] * int(hire_counts[environment])
            if base_actions is not None:
                seat = int(seats[environment])
                length = int(base_actions.market_lengths[environment, seat])
                output.extend(
                    tuple(int(value) for value in row)
                    for row in base_actions.market_actions[
                        environment, seat, :length
                    ]
                    if int(row[0]) != int(MarketOp.HIRE)
                )
            output = output[:max_orders]
            self._lengths[environment] = len(output)
            if output:
                self._actions[environment, : len(output)] = output
        return self._actions, self._lengths

    def _ensure_buffers(self, environments: int, max_orders: int) -> None:
        shape = (environments, max_orders, 3)
        if self._actions is None or self._actions.shape != shape:
            self._actions = np.zeros(shape, dtype=np.int64)
            self._lengths = np.zeros(environments, dtype=np.int64)


__all__ = ["LearnerMarketPolicy", "WorkforceMarketPolicy"]
