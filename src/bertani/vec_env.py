"""NumPy-first vector environment for the Rust Kaggriculture simulator.

The arrays in a :class:`Batch` are owned by the environment and reused.  A
subsequent call to :meth:`VecEnv.reset` or :meth:`VecEnv.step` overwrites their
contents, so copy values that need to outlive the next environment call.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ._rust import (
    ITEM_COUNT,
    MARKET_ACTION_COUNT,
    RL_API_VERSION,
    UNIT_ACTION_COUNT,
    NativeVecEnv,
)


class UnitOp(IntEnum):
    """Stable operation IDs for the last axis of a unit action row."""

    PASS = 0
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4
    PICKUP = 5
    DROP = 6
    PLACE = 7
    PLANT = 8
    WATER = 9
    HARVEST = 10
    FERTILIZE = 11
    DIG = 12
    BUILD_COOP = 13
    BUILD_PASTURE = 14
    FEED = 15
    COLLECT_FERTILIZER = 16
    CARE = 17


class MarketOp(IntEnum):
    """Stable operation IDs for the last axis of a market action row."""

    NONE = 0
    HIRE = 1
    BUY_LAND = 2
    BUY_SEED = 3
    BUY_PRODUCT = 4
    BUY_ANIMAL = 5
    SELL = 6


class Item(IntEnum):
    """Shared item IDs used by unit and market action arguments."""

    WHEAT = 0
    CARROT = 1
    TOMATO = 2
    STRAWBERRY = 3
    MELON = 4
    EGG = 5
    MILK = 6
    WOOL = 7
    FERTILIZER = 8
    GOOSE = 9
    COW = 10
    SHEEP = 11


if RL_API_VERSION != 1:
    raise RuntimeError(
        f"unsupported native RL API version {RL_API_VERSION}; expected version 1"
    )
if len(UnitOp) != UNIT_ACTION_COUNT:
    raise RuntimeError("Python UnitOp IDs do not match the native extension")
if len(MarketOp) != MARKET_ACTION_COUNT:
    raise RuntimeError("Python MarketOp IDs do not match the native extension")
if len(Item) != ITEM_COUNT:
    raise RuntimeError("Python Item IDs do not match the native extension")


Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]
UInt64Array = NDArray[np.uint64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ObservationViews:
    """Structured, zero-copy views into a flattened observation tensor."""

    global_features: Float32Array
    farms: Float32Array
    tiles: Float32Array
    units: Float32Array
    private: Float32Array


@dataclass(frozen=True, slots=True)
class MaskViews:
    """Structured, zero-copy views into a flattened action-mask tensor."""

    unit_ops: BoolArray
    unit_args: BoolArray
    market_ops: BoolArray
    market_args: BoolArray


@dataclass(frozen=True, slots=True)
class Batch:
    """Reusable output buffers from one vector-environment call.

    Every field, including all structured views, is overwritten by the next
    ``reset`` or ``step`` call on the environment that created it.
    """

    observations: Float32Array
    action_masks: BoolArray
    active_units: BoolArray
    rewards: Float64Array
    dones: BoolArray
    episode_ids: UInt64Array
    overflow: BoolArray
    observation_views: ObservationViews
    mask_views: MaskViews


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> NDArray[Any]:
    """Validate without asking NumPy to allocate or coerce an input."""

    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    array = value
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype.name}, got {array.dtype}")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return array


def _nested_integer(mapping: Mapping[str, Any], *paths: tuple[str, ...]) -> int | None:
    for path in paths:
        value: Any = mapping
        for part in path:
            if not isinstance(value, Mapping) or part not in value:
                break
            value = value[part]
        else:
            if isinstance(value, int):
                return value
    return None


def _offset(
    specs: Mapping[str, Any], family: str, name: str, default: int
) -> int:
    value = _nested_integer(
        specs,
        (f"{family}_{name}",),
        (f"{family}_{name}_offset",),
        (family, name),
        (family, f"{name}_offset"),
        (f"{family}_offsets", name),
    )
    return default if value is None else value


class VecEnv:
    """Batched Kaggriculture simulations backed by the Rust rules engine.

    Unit actions have shape ``[N, 2, max_units, 3]`` and market actions have
    shape ``[N, 2, max_orders, 3]``.  Their final axis is ``(op, arg, count)``.
    Inputs are required to be C-contiguous ``int64`` arrays; the wrapper never
    silently casts an action tensor.
    """

    def __init__(
        self,
        num_envs: int,
        seed: int = 0,
        max_units: int = 0,
        auto_reset: bool = True,
        episode_steps: int = 720,
        board_size: int = 10,
        starting_money: int = 3_000,
        max_market_orders: int = 10,
        turns_per_day: int = 24,
        shed_capacity: int = 100,
        weed_spawn_chance: float = 0.005,
        town_shop_unlock_interval: int = 3,
        town_shop_sell_interval: int = 4,
        town_center_sell_interval: int = 24,
        farm_hand_cost_multiplier: int = 1,
    ) -> None:
        self._native = NativeVecEnv(
            num_envs,
            seed,
            max_units,
            auto_reset,
            episode_steps,
            board_size,
            starting_money,
            max_market_orders,
            turns_per_day,
            shed_capacity,
            weed_spawn_chance,
            town_shop_unlock_interval,
            town_shop_sell_interval,
            town_center_sell_interval,
            farm_hand_cost_multiplier,
        )

        n = self._native.num_envs
        units = self._native.max_units
        orders = self._native.max_orders
        observation_size = self._native.observation_size
        mask_size = self._native.mask_size

        self._observations = np.empty((n, 2, observation_size), dtype=np.float32)
        self._action_mask_bytes = np.zeros((n, 2, mask_size), dtype=np.uint8)
        self._action_masks = self._action_mask_bytes.view(np.bool_)
        self._active_unit_bytes = np.zeros((n, 2, units), dtype=np.uint8)
        self._active_units = self._active_unit_bytes.view(np.bool_)
        self._rewards = np.empty((n, 2), dtype=np.float64)
        self._done_bytes = np.zeros((n, 2), dtype=np.uint8)
        self._dones = self._done_bytes.view(np.bool_)
        self._episode_ids = np.empty((n,), dtype=np.uint64)
        self._overflow_bytes = np.zeros((n, 2), dtype=np.uint8)
        self._overflow = self._overflow_bytes.view(np.bool_)

        # These immutable defaults avoid allocating pass actions per step.
        self._pass_unit_actions = np.zeros((n, 2, units, 3), dtype=np.int64)
        self._pass_market_actions = np.zeros((n, 2, orders, 3), dtype=np.int64)
        self._empty_market_lengths = np.zeros((n, 2), dtype=np.int64)

        observations = self._observation_views(self._native.buffer_specs())
        masks = self._mask_views(self._native.buffer_specs())
        self._batch = Batch(
            observations=self._observations,
            action_masks=self._action_masks,
            active_units=self._active_units,
            rewards=self._rewards,
            dones=self._dones,
            episode_ids=self._episode_ids,
            overflow=self._overflow,
            observation_views=observations,
            mask_views=masks,
        )

    @property
    def native(self) -> NativeVecEnv:
        """The low-level extension object, for diagnostics and integrations."""

        return self._native

    @property
    def num_envs(self) -> int:
        return self._native.num_envs

    @property
    def max_units(self) -> int:
        return self._native.max_units

    @property
    def max_orders(self) -> int:
        return self._native.max_orders

    @property
    def observation_size(self) -> int:
        return self._native.observation_size

    @property
    def mask_size(self) -> int:
        return self._native.mask_size

    @property
    def board_size(self) -> int:
        return self._native.board_size

    @property
    def auto_reset(self) -> bool:
        return self._native.auto_reset

    @auto_reset.setter
    def auto_reset(self, value: bool) -> None:
        self._native.auto_reset = value

    @property
    def buffer_specs(self) -> dict[str, Any]:
        """Native flattened-buffer offsets used to construct the named views."""

        return self._native.buffer_specs()

    @property
    def unit_actions(self) -> Int64Array:
        """A reusable, initially all-pass unit-action tensor."""

        return self._pass_unit_actions

    @property
    def market_actions(self) -> Int64Array:
        """A reusable, initially all-NONE market-action tensor."""

        return self._pass_market_actions

    @property
    def market_lengths(self) -> Int64Array:
        """A reusable, initially empty market-prefix tensor."""

        return self._empty_market_lengths

    def clear_actions(self) -> tuple[Int64Array, Int64Array, Int64Array]:
        """Clear and return the three reusable action buffers."""

        self._pass_unit_actions.fill(0)
        self._pass_market_actions.fill(0)
        self._empty_market_lengths.fill(0)
        return (
            self._pass_unit_actions,
            self._pass_market_actions,
            self._empty_market_lengths,
        )

    def reset(self, seeds: object | None = None) -> Batch:
        """Reset every slot, optionally with one explicit ``uint64`` seed each."""

        checked_seeds: UInt64Array | None
        if seeds is None:
            checked_seeds = None
        else:
            checked_seeds = _require_array(
                seeds,
                name="seeds",
                shape=(self.num_envs,),
                dtype=np.dtype(np.uint64),
            )
        self._native.reset_into(
            checked_seeds,
            self._observations,
            self._action_mask_bytes,
            self._active_unit_bytes,
            self._rewards,
            self._done_bytes,
            self._episode_ids,
            self._overflow_bytes,
        )
        return self._batch

    def step(
        self,
        unit_actions: object | None = None,
        market_actions: object | None = None,
        market_lengths: object | None = None,
    ) -> Batch:
        """Advance every slot once and overwrite the reusable output batch.

        If a market tensor is supplied without lengths, the active prefix is
        inferred through the last row whose operation is not ``MarketOp.NONE``.
        """

        if unit_actions is None:
            units = self._pass_unit_actions
        else:
            units = _require_array(
                unit_actions,
                name="unit_actions",
                shape=(self.num_envs, 2, self.max_units, 3),
                dtype=np.dtype(np.int64),
            )

        if market_actions is None:
            market = self._pass_market_actions
        else:
            market = _require_array(
                market_actions,
                name="market_actions",
                shape=(self.num_envs, 2, self.max_orders, 3),
                dtype=np.dtype(np.int64),
            )

        if market_lengths is None:
            if market_actions is None:
                lengths = self._empty_market_lengths
            else:
                lengths = self._infer_market_lengths(market)
        else:
            lengths = _require_array(
                market_lengths,
                name="market_lengths",
                shape=(self.num_envs, 2),
                dtype=np.dtype(np.int64),
            )

        self._native.step_into(
            units,
            market,
            lengths,
            self._observations,
            self._action_mask_bytes,
            self._active_unit_bytes,
            self._rewards,
            self._done_bytes,
            self._episode_ids,
            self._overflow_bytes,
        )
        return self._batch

    def state_snapshot(self, index: int) -> dict[str, Any]:
        """Return a JSON-compatible copy of the current state for one slot."""

        return self._native.state_snapshot(index)

    def terminal_snapshot(self, index: int) -> dict[str, Any] | None:
        """Return the last terminal state retained by auto-reset, if any."""

        return self._native.terminal_snapshot(index)

    def _infer_market_lengths(self, actions: Int64Array) -> Int64Array:
        self._empty_market_lengths.fill(0)
        nonempty = actions[..., 0] != int(MarketOp.NONE)
        for environment in range(self.num_envs):
            for player in range(2):
                indices = np.flatnonzero(nonempty[environment, player])
                if indices.size:
                    self._empty_market_lengths[environment, player] = indices[-1] + 1
        return self._empty_market_lengths

    def _observation_views(self, specs: Mapping[str, Any]) -> ObservationViews:
        n, players = self.num_envs, 2
        board = self.board_size
        units = self.max_units

        # These defaults are the public v1 layout. Native-provided offsets win,
        # allowing the extension to add layouts while retaining one view builder.
        global_offset = _offset(specs, "observation", "global", 0)
        farms_offset = _offset(specs, "observation", "farms", 30)
        tiles_offset = _offset(specs, "observation", "tiles", farms_offset + 2 * 9)
        units_offset = _offset(
            specs,
            "observation",
            "units",
            tiles_offset + 2 * board * board * 24,
        )
        private_offset = _offset(
            specs,
            "observation",
            "private",
            units_offset + 2 * units * 29,
        )

        global_channels = farms_offset - global_offset
        farm_channels = (tiles_offset - farms_offset) // 2
        tile_channels = (units_offset - tiles_offset) // (2 * board * board)
        unit_channels = (private_offset - units_offset) // (2 * units)
        private_channels = self.observation_size - private_offset

        return ObservationViews(
            global_features=self._observations[
                ..., global_offset:farms_offset
            ].reshape(n, players, global_channels),
            farms=self._observations[..., farms_offset:tiles_offset].reshape(
                n, players, 2, farm_channels
            ),
            tiles=self._observations[..., tiles_offset:units_offset].reshape(
                n, players, 2, board, board, tile_channels
            ),
            units=self._observations[..., units_offset:private_offset].reshape(
                n, players, 2, units, unit_channels
            ),
            private=self._observations[..., private_offset:].reshape(
                n, players, private_channels
            ),
        )

    def _mask_views(self, specs: Mapping[str, Any]) -> MaskViews:
        n, players = self.num_envs, 2
        units = self.max_units
        unit_ops_offset = _offset(specs, "mask", "unit_ops", 0)
        unit_args_offset = _offset(
            specs, "mask", "unit_args", units * UNIT_ACTION_COUNT
        )
        market_ops_offset = _offset(
            specs,
            "mask",
            "market_ops",
            unit_args_offset + units * UNIT_ACTION_COUNT * ITEM_COUNT,
        )
        market_args_offset = _offset(
            specs,
            "mask",
            "market_args",
            market_ops_offset + MARKET_ACTION_COUNT,
        )

        return MaskViews(
            unit_ops=self._action_masks[
                ..., unit_ops_offset:unit_args_offset
            ].reshape(n, players, units, UNIT_ACTION_COUNT),
            unit_args=self._action_masks[
                ..., unit_args_offset:market_ops_offset
            ].reshape(n, players, units, UNIT_ACTION_COUNT, ITEM_COUNT),
            market_ops=self._action_masks[
                ..., market_ops_offset:market_args_offset
            ].reshape(n, players, MARKET_ACTION_COUNT),
            market_args=self._action_masks[..., market_args_offset:].reshape(
                n, players, MARKET_ACTION_COUNT, ITEM_COUNT
            ),
        )


__all__ = [
    "Batch",
    "Item",
    "MarketOp",
    "MaskViews",
    "ObservationViews",
    "UnitOp",
    "VecEnv",
]
