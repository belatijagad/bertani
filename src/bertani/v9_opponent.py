"""Efficient frozen-V9 opponent integration for PPO rollouts.

The preserved V9 submission contains one large decoded replay ensemble and a
few module globals that track the active trajectory. Loading an isolated copy
for every vector slot is prohibitively expensive. This adapter loads the replay
bank once, while saving the small mutable state independently for every slot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

from .actions import ActionBatch
from .native_agent import load_agent_module, snapshot_observation
from .vec_env import Batch, Item, MarketOp, UnitOp, VecEnv

DEFAULT_V9_PATH = Path(__file__).resolve().parents[2] / "references" / (
    "v9_main_restarted.py"
)
_STATE_NAMES = (
    "_R9_MODE",
    "_R9_LAST_TOWN",
    "_R9_LAST_STEP",
    "_V8_COMMIT",
    "_V8_LAST_STEP",
)


@dataclass
class _V9State:
    r9_mode: int = 5
    r9_last_town: tuple[str, ...] | None = None
    r9_last_step: int = -1
    v8_commit: object = None
    v8_last_step: int = -1
    town_fingerprint: tuple[int, int] | None = None

    def values(self) -> tuple[object, ...]:
        return (
            self.r9_mode,
            self.r9_last_town,
            self.r9_last_step,
            self.v8_commit,
            self.v8_last_step,
        )

    @classmethod
    def from_module(
        cls,
        module: ModuleType,
        *,
        town_fingerprint: tuple[int, int],
    ) -> _V9State:
        values = [getattr(module, name) for name in _STATE_NAMES]
        return cls(*values, town_fingerprint=town_fingerprint)

    def copy(self) -> _V9State:
        return _V9State(*self.values(), town_fingerprint=self.town_fingerprint)


@dataclass(frozen=True)
class V9CacheStats:
    """Same-step cache diagnostics for rollout profiling."""

    hits: int
    misses: int


def _unit_row(raw: Sequence[object] | None) -> tuple[int, int, int]:
    action = list(raw or ("PASS",))
    operation = UnitOp[str(action[0])]
    if operation in {UnitOp.PICKUP, UnitOp.PLACE, UnitOp.PLANT}:
        count = int(action[2]) if len(action) >= 3 else 1
        return int(operation), int(Item[str(action[1])]), count
    return int(operation), 0, 0


def _market_row(raw: Sequence[object]) -> tuple[int, int, int]:
    action = list(raw)
    operation = MarketOp[str(action[0])]
    if operation in {MarketOp.HIRE, MarketOp.BUY_LAND}:
        return int(operation), 0, 0
    return int(operation), int(Item[str(action[1])]), int(action[2])


class V9OpponentPolicy:
    """Run one shared V9 replay bank as an isolated opponent in every slot."""

    def __init__(
        self,
        module: ModuleType,
        *,
        configuration: Mapping[str, object],
        max_orders: int,
        cache_identical_states: bool = True,
    ) -> None:
        required = (*_STATE_NAMES, "_feature", "_R9_BANK", "agent")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise TypeError(f"module is not a V9 agent; missing {missing!r}")
        self.module = module
        self.configuration = dict(configuration)
        self.max_orders = max_orders
        self.cache_identical_states = cache_identical_states
        self._states: list[_V9State] = []
        self._episode_ids = np.empty(0, dtype=np.uint64)
        self._actions: ActionBatch | None = None
        self._shape: tuple[int, int, int] | None = None
        self._hits = 0
        self._misses = 0

    @classmethod
    def from_path(
        cls,
        path: Path = DEFAULT_V9_PATH,
        *,
        configuration: Mapping[str, object],
        max_orders: int,
        cache_identical_states: bool = True,
    ) -> V9OpponentPolicy:
        """Load V9 once and construct the batched opponent adapter."""

        return cls(
            load_agent_module(path),
            configuration=configuration,
            max_orders=max_orders,
            cache_identical_states=cache_identical_states,
        )

    @property
    def cache_stats(self) -> V9CacheStats:
        return V9CacheStats(self._hits, self._misses)

    def reset(self) -> None:
        """Forget all per-slot episode state and cache diagnostics."""

        self._states.clear()
        self._episode_ids = np.empty(0, dtype=np.uint64)
        self._hits = 0
        self._misses = 0

    def act(
        self,
        environment: VecEnv,
        batch: Batch,
        *,
        seats: NDArray[np.int64],
    ) -> ActionBatch:
        """Return V9 actions for exactly one selected seat per environment."""

        if seats.shape != (environment.num_envs,):
            raise ValueError(f"seats must have shape ({environment.num_envs},)")
        if np.any((seats < 0) | (seats > 1)):
            raise ValueError("seats must contain only 0 or 1")

        self._sync_slots(batch)
        actions = self._buffers(batch)
        actions.unit_actions.fill(0)
        actions.market_actions.fill(0)
        actions.market_lengths.fill(0)
        # The cache is deliberately scoped to one synchronous vector step. It
        # captures duplicated openings without retaining full farm signatures.
        cache: dict[object, tuple[np.ndarray, np.ndarray, int, _V9State]] = {}
        fingerprints = environment.v9_fingerprints(seats)

        for environment_index in range(environment.num_envs):
            seat = int(seats[environment_index])
            state = self._states[environment_index]
            row = fingerprints[environment_index]
            state_fingerprint = (int(row[0]), int(row[1]))
            town_fingerprint = (int(row[2]), int(row[3]))
            step = int(row[4])
            staff = int(row[5])
            rows = (
                self.module._R9_BANK[step].get(staff, ())
                if 0 <= step < len(self.module._R9_BANK)
                else ()
            )
            effective_mode = (
                5
                if town_fingerprint != state.town_fingerprint
                else state.r9_mode
            )
            key = None
            if self.cache_identical_states and rows:
                key = (state_fingerprint, town_fingerprint, effective_mode)
                cached = cache.get(key)
                if cached is not None:
                    units, market, market_length, post_state = cached
                    self._write_encoded(
                        actions,
                        batch,
                        environment_index,
                        seat,
                        units,
                        market,
                        market_length,
                    )
                    self._states[environment_index] = post_state.copy()
                    self._hits += 1
                    continue

            snapshot = environment.state_snapshot(environment_index)
            observation = snapshot_observation(
                snapshot, seat, include_opponent=not bool(rows)
            )
            feature = self.module._feature(observation)
            self._restore_state(state)
            original_feature = self.module._feature
            if rows:
                # _r9_step would otherwise rebuild the exact feature that was
                # just needed for the cache key. Calls are serial because V9's
                # replay selector itself uses module-global trajectory state.
                self.module._feature = lambda _observation, value=feature: value
            try:
                raw = self.module.agent(observation, self.configuration) or {}
            finally:
                self.module._feature = original_feature
            post_state = _V9State.from_module(
                self.module, town_fingerprint=town_fingerprint
            )
            self._states[environment_index] = post_state
            units, market, market_length = self._encode(raw)
            self._write_encoded(
                actions,
                batch,
                environment_index,
                seat,
                units,
                market,
                market_length,
            )
            if key is not None:
                cache[key] = (units, market, market_length, post_state)
            self._misses += 1
        return actions

    def _sync_slots(self, batch: Batch) -> None:
        count = len(batch.episode_ids)
        if len(self._states) != count:
            self._states = [_V9State() for _ in range(count)]
            self._episode_ids = np.full(count, np.iinfo(np.uint64).max, np.uint64)
        changed = batch.episode_ids != self._episode_ids
        for index in np.flatnonzero(changed):
            self._states[int(index)] = _V9State()
        self._episode_ids[:] = batch.episode_ids

    def _restore_state(self, state: _V9State) -> None:
        for name, value in zip(_STATE_NAMES, state.values()):
            setattr(self.module, name, value)

    def _encode(
        self, raw: Mapping[str, object]
    ) -> tuple[np.ndarray, np.ndarray, int]:
        units_raw = [raw.get("farmer") or ["PASS"], *(raw.get("hands") or [])]
        units = np.zeros((len(units_raw), 3), dtype=np.int64)
        for index, unit_action in enumerate(units_raw):
            units[index] = _unit_row(unit_action)
        market_raw = list(raw.get("market") or [])[: self.max_orders]
        market = np.zeros((len(market_raw), 3), dtype=np.int64)
        for index, market_action in enumerate(market_raw):
            market[index] = _market_row(market_action)
        return units, market, len(market_raw)

    @staticmethod
    def _write_encoded(
        actions: ActionBatch,
        batch: Batch,
        environment_index: int,
        seat: int,
        units: np.ndarray,
        market: np.ndarray,
        market_length: int,
    ) -> None:
        unit_limit = min(len(units), actions.unit_actions.shape[2])
        if unit_limit:
            active = batch.active_units[
                environment_index, seat, :unit_limit
            ]
            actions.unit_actions[
                environment_index, seat, :unit_limit
            ][active] = units[:unit_limit][active]
        if market_length:
            actions.market_actions[
                environment_index, seat, :market_length
            ] = market
        actions.market_lengths[environment_index, seat] = market_length

    def _buffers(self, batch: Batch) -> ActionBatch:
        shape = batch.active_units.shape
        if self._actions is None or self._shape != shape:
            environments, players, units = shape
            self._actions = ActionBatch(
                unit_actions=np.zeros(
                    (environments, players, units, 3), dtype=np.int64
                ),
                market_actions=np.zeros(
                    (environments, players, self.max_orders, 3), dtype=np.int64
                ),
                market_lengths=np.zeros((environments, players), dtype=np.int64),
            )
            self._shape = shape
        return self._actions


__all__ = [
    "DEFAULT_V9_PATH",
    "V9CacheStats",
    "V9OpponentPolicy",
]
