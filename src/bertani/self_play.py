"""Policy-neutral composition of learner and frozen-opponent actions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .actions import ActionBatch, IntegerArray
from .vec_env import Batch, MarketOp, VecEnv


@dataclass(frozen=True)
class SelfPlayStepProfile:
    """Wall-clock split for the most recent composed self-play step."""

    opponent_seconds: float = 0.0
    composition_seconds: float = 0.0
    environment_seconds: float = 0.0
    total_seconds: float = 0.0


class OpponentCacheStats(Protocol):
    hits: int
    misses: int


class OpponentPolicy(Protocol):
    """Interface implemented by batched frozen self-play opponents."""

    @property
    def cache_stats(self) -> OpponentCacheStats: ...

    def reset(self) -> None: ...

    def act(
        self,
        environment: VecEnv,
        batch: Batch,
        *,
        seats: NDArray[np.int64],
    ) -> ActionBatch: ...


class SelfPlayEnv:
    """Compose one learner seat and one frozen opponent in every vector slot."""

    def __init__(self, environment: VecEnv, opponent: OpponentPolicy) -> None:
        self.environment = environment
        self.opponent = opponent
        self.games = np.arange(environment.num_envs, dtype=np.int64)
        self.learner_seats = self.games % 2
        self.opponent_seats = 1 - self.learner_seats
        self.learner_flat_indices = 2 * self.games + self.learner_seats
        self._batch: Batch | None = None
        self.last_step_profile = SelfPlayStepProfile()

    @property
    def batch(self) -> Batch:
        if self._batch is None:
            raise RuntimeError("reset must be called before accessing the batch")
        return self._batch

    def reset(self, seeds: object | None = None) -> Batch:
        self.opponent.reset()
        self._batch = self.environment.reset(seeds)
        return self._batch

    def step(
        self,
        learner_unit_actions: IntegerArray,
        learner_market_actions: IntegerArray | None = None,
        learner_market_lengths: IntegerArray | None = None,
    ) -> Batch:
        """Advance using compact learner-only tensors shaped with leading ``N``."""

        step_started = time.perf_counter()
        batch = self.batch
        expected_units = (
            self.environment.num_envs,
            self.environment.max_units,
            3,
        )
        if learner_unit_actions.shape != expected_units:
            raise ValueError(
                f"learner_unit_actions must have shape {expected_units}"
            )
        if learner_market_lengths is not None and learner_market_actions is None:
            raise ValueError("learner_market_lengths requires learner_market_actions")
        unit_actions, market_actions, market_lengths = (
            self.environment.clear_actions()
        )
        opponent_started = time.perf_counter()
        opponent = self.opponent.act(
            self.environment, batch, seats=self.opponent_seats
        )
        opponent_finished = time.perf_counter()
        index = (self.games, self.learner_seats)
        opponent_index = (self.games, self.opponent_seats)
        unit_actions[index] = learner_unit_actions
        unit_actions[opponent_index] = opponent.unit_actions[opponent_index]
        if learner_market_actions is not None:
            expected_market = (
                self.environment.num_envs,
                self.environment.max_orders,
                3,
            )
            if learner_market_actions.shape != expected_market:
                raise ValueError(
                    f"learner_market_actions must have shape {expected_market}"
                )
            market_actions[index] = learner_market_actions
            if learner_market_lengths is None:
                active = learner_market_actions[..., 0] != int(MarketOp.NONE)
                any_active = active.any(axis=-1)
                reversed_last = np.argmax(active[:, ::-1], axis=-1)
                learner_market_lengths = np.where(
                    any_active,
                    self.environment.max_orders - reversed_last,
                    0,
                ).astype(np.int64)
        if learner_market_lengths is not None:
            if learner_market_lengths.shape != (self.environment.num_envs,):
                raise ValueError(
                    "learner_market_lengths must have shape "
                    f"({self.environment.num_envs},)"
                )
            market_lengths[index] = learner_market_lengths
        market_actions[opponent_index] = opponent.market_actions[opponent_index]
        market_lengths[opponent_index] = opponent.market_lengths[opponent_index]
        environment_started = time.perf_counter()
        self._batch = self.environment.step(
            unit_actions, market_actions, market_lengths
        )
        finished = time.perf_counter()
        self.last_step_profile = SelfPlayStepProfile(
            opponent_seconds=opponent_finished - opponent_started,
            composition_seconds=environment_started - opponent_finished,
            environment_seconds=finished - environment_started,
            total_seconds=finished - step_started,
        )
        return self._batch

    def learner_rewards(self) -> NDArray[np.float64]:
        return self.batch.rewards[self.games, self.learner_seats]

    def learner_dones(self) -> NDArray[np.bool_]:
        return self.batch.dones[self.games, self.learner_seats]


__all__ = [
    "OpponentCacheStats",
    "OpponentPolicy",
    "SelfPlayEnv",
    "SelfPlayStepProfile",
]
