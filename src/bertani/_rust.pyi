"""Type declarations for the native PyO3 extension."""

from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

UNIT_ACTION_COUNT: Final[int]
MARKET_ACTION_COUNT: Final[int]
ITEM_COUNT: Final[int]
BUILD_PROFILE: Final[str]
RL_API_VERSION: Final[int]
ITEM_WHEAT: Final[int]
ITEM_CARROT: Final[int]
ITEM_TOMATO: Final[int]
ITEM_STRAWBERRY: Final[int]
ITEM_MELON: Final[int]
ITEM_EGG: Final[int]
ITEM_MILK: Final[int]
ITEM_WOOL: Final[int]
ITEM_FERTILIZER: Final[int]
ITEM_GOOSE: Final[int]
ITEM_COW: Final[int]
ITEM_SHEEP: Final[int]


class NativeVecEnv:
    def __init__(
        self,
        num_envs: int,
        seed: int = ...,
        max_units: int = ...,
        auto_reset: bool = ...,
        episode_steps: int = ...,
        board_size: int = ...,
        starting_money: int = ...,
        max_market_orders: int = ...,
        turns_per_day: int = ...,
        shed_capacity: int = ...,
        weed_spawn_chance: float = ...,
        town_shop_unlock_interval: int = ...,
        town_shop_sell_interval: int = ...,
        town_center_sell_interval: int = ...,
        farm_hand_cost_multiplier: int = ...,
    ) -> None: ...

    @property
    def num_envs(self) -> int: ...

    @property
    def max_units(self) -> int: ...

    @property
    def max_orders(self) -> int: ...

    @property
    def observation_size(self) -> int: ...

    @property
    def mask_size(self) -> int: ...

    @property
    def board_size(self) -> int: ...

    @property
    def auto_reset(self) -> bool: ...

    @auto_reset.setter
    def auto_reset(self, value: bool) -> None: ...

    def buffer_specs(self) -> dict[str, Any]: ...

    def reset_into(
        self,
        seeds: NDArray[np.uint64] | None,
        observations: NDArray[np.float32],
        action_masks: NDArray[np.uint8],
        unit_active: NDArray[np.uint8],
        rewards: NDArray[np.float64],
        dones: NDArray[np.uint8],
        episode_ids: NDArray[np.uint64],
        overflows: NDArray[np.uint8],
    ) -> None: ...

    def step_into(
        self,
        unit_actions: NDArray[np.int64],
        market_actions: NDArray[np.int64],
        market_lengths: NDArray[np.int64],
        observations: NDArray[np.float32],
        action_masks: NDArray[np.uint8],
        unit_active: NDArray[np.uint8],
        rewards: NDArray[np.float64],
        dones: NDArray[np.uint8],
        episode_ids: NDArray[np.uint64],
        overflows: NDArray[np.uint8],
    ) -> None: ...

    def state_snapshot(self, index: int) -> dict[str, Any]: ...
    def v9_fingerprints(
        self,
        seats: NDArray[np.int64],
        output: NDArray[np.uint64],
    ) -> None: ...
    def terminal_snapshot(self, index: int) -> dict[str, Any] | None: ...
