"""Run submission-style Kaggle agents against :class:`bertani.VecEnv`.

The native simulator exposes compact numeric state snapshots for diagnostics.
This module converts those snapshots back to the public Kaggriculture
observation schema and converts an agent's action dictionaries to native action
tensors.  It is intended for evaluation of preserved Python agents, not for
submission packaging.
"""

from __future__ import annotations

import gc
import importlib.util
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .actions import ActionBatch
from .vec_env import Batch, Item, MarketOp, UnitOp, VecEnv

Agent = Callable[..., dict[str, object]]
ITEM_NAMES = tuple(item.name for item in Item)
CROP_NAMES = ITEM_NAMES[:5]
PRODUCT_NAMES = ITEM_NAMES[:9]
ANIMAL_NAMES = ITEM_NAMES[9:12]
QUADRANT_NAMES = ("NW", "NE", "SW", "SE")
SHOP_NAMES = (
    "BAKERY",
    "BRUNCH_SPOT",
    "FARMERS_MARKET",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "PIZZA_SHOP",
    "SMOOTHIE_SHOP",
    "YARN_STORE",
)


def load_agent_module(path: Path) -> ModuleType:
    """Load an isolated submission module without leaking its GC preference."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"agent does not exist: {resolved}")
    spec = importlib.util.spec_from_file_location(
        f"bertani_native_agent_{uuid.uuid4().hex}", resolved
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load agent module: {resolved}")
    module = importlib.util.module_from_spec(spec)
    module_directory = str(resolved.parent)
    gc_was_enabled = gc.isenabled()
    sys.path.insert(0, module_directory)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(module_directory)
        # Some preserved competition agents disable cyclic collection to avoid
        # inference jitter. That process-global choice must not leak into a PPO
        # trainer which creates longer-lived Python and Torch object graphs.
        if gc_was_enabled and not gc.isenabled():
            gc.enable()
    return module


def load_agent_file(path: Path) -> Agent:
    """Load an isolated submission module and return its ``agent`` callable."""

    resolved = path.resolve()
    module = load_agent_module(resolved)
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise TypeError(f"{resolved} does not define a callable agent")
    return agent


def _named_counts(values: Sequence[object], names: Sequence[str]) -> dict[str, int]:
    return {name: int(values[index]) for index, name in enumerate(names)}


def _inventory(snapshot: Mapping[str, object]) -> dict[str, int]:
    counts = list(snapshot.get("counts", ()))
    insertion_order = list(snapshot.get("insertion_order", ()))
    return {
        ITEM_NAMES[int(item)]: int(counts[int(item)])
        for item in insertion_order
        if int(counts[int(item)]) > 0
    }


def _tile(snapshot: Mapping[str, object]) -> object:
    kind = str(snapshot["kind"])
    if kind == "EMPTY":
        return None
    if kind == "LOCKED":
        return "LOCKED"
    if kind == "WEED":
        return {"kind": "WEED"}
    output = dict(snapshot)
    if kind == "PLANT":
        output["crop"] = CROP_NAMES[int(output["crop"])]
    elif kind in {"COOP", "PASTURE"} and "animal" in output:
        output["animal"] = ANIMAL_NAMES[int(output["animal"])]
    return output


def snapshot_observation(
    snapshot: Mapping[str, object],
    seat: int,
    *,
    include_opponent: bool = True,
) -> dict[str, object]:
    """Convert one native state snapshot to a seat's public observation.

    ``include_opponent=False`` keeps the opponent's public scalars and units but
    omits its tile conversion. It is an internal fast path for policies, such as
    V9's normal R9 branch, that provably inspect only their own farm.
    """

    if seat not in (0, 1):
        raise ValueError("seat must be 0 or 1")
    farms_raw = list(snapshot["farms"])
    farms: list[dict[str, object]] = []
    for farm_index, raw in enumerate(farms_raw):
        farm = dict(raw)
        farms.append(
            {
                "money": farm["money"],
                "tiles": (
                    [
                        [_tile(tile) for tile in row]
                        for row in farm["tiles"]
                    ]
                    if include_opponent or farm_index == seat
                    else []
                ),
                "farmer": list(farm["farmer"]),
                "hands": [list(position) for position in farm["hands"]],
                "unlocked_quadrants": [
                    QUADRANT_NAMES[int(value)]
                    for value in farm["unlocked_quadrants"]
                ],
                "hires_today": int(farm["hires_today"]),
            }
        )

    own = dict(farms_raw[seat])
    market_raw = dict(snapshot["market"])
    town_raw = dict(snapshot["town"])
    return {
        "player": seat,
        "step": int(snapshot["step"]),
        "day": int(snapshot["day"]),
        "hour": int(snapshot["hour"]),
        "farms": farms,
        "private": {
            "shed": _named_counts(list(own["shed"]), ITEM_NAMES),
            "seeds": _named_counts(list(own["seeds"]), CROP_NAMES),
            "inventories": [
                _inventory(inventory) for inventory in own["inventories"]
            ],
        },
        "market": {
            "inventory": _named_counts(
                list(market_raw["inventory"]), PRODUCT_NAMES
            ),
            "prices": {
                name: round(float(list(market_raw["prices"])[index]))
                for index, name in enumerate(PRODUCT_NAMES)
            },
        },
        "town": {
            "unlocked_shops": [
                SHOP_NAMES[int(value)] for value in town_raw["unlocked_shops"]
            ]
        },
        "remainingOverageTime": 60,
    }


def _unit_row(raw: Sequence[object] | None) -> tuple[int, int, int]:
    action = list(raw or ("PASS",))
    operation = UnitOp[str(action[0])]
    item = 0
    count = 0
    if operation in {UnitOp.PICKUP, UnitOp.PLACE, UnitOp.PLANT}:
        item = int(Item[str(action[1])])
        count = int(action[2]) if len(action) >= 3 else 1
    return int(operation), item, count


def _market_row(raw: Sequence[object]) -> tuple[int, int, int]:
    action = list(raw)
    operation = MarketOp[str(action[0])]
    if operation in {MarketOp.HIRE, MarketOp.BUY_LAND}:
        return int(operation), 0, 0
    return int(operation), int(Item[str(action[1])]), int(action[2])


class NativeFileAgentPolicy:
    """Adapt one loaded Kaggle agent to selected seats in a native batch."""

    def __init__(
        self,
        agent: Agent,
        *,
        configuration: Mapping[str, object],
        max_orders: int,
    ) -> None:
        self.agent = agent
        self.configuration = dict(configuration)
        self.max_orders = max_orders
        self._shape: tuple[int, int, int] | None = None
        self._actions: ActionBatch | None = None

    def act(
        self,
        environment: VecEnv,
        batch: Batch,
        *,
        seat_mask: np.ndarray[Any, np.dtype[np.bool_]],
    ) -> ActionBatch:
        """Call the file agent for each selected environment and seat."""

        actions = self._buffers(batch)
        actions.unit_actions.fill(0)
        actions.market_actions.fill(0)
        actions.market_lengths.fill(0)
        for environment_index, seat in zip(*np.nonzero(seat_mask)):
            observation = snapshot_observation(
                environment.state_snapshot(int(environment_index)), int(seat)
            )
            arguments: tuple[object, ...] = (observation, self.configuration)
            code = getattr(self.agent, "__code__", None)
            if code is not None:
                arguments = arguments[: code.co_argcount]
            raw = self.agent(*arguments) or {}
            units = [raw.get("farmer") or ["PASS"], *(raw.get("hands") or [])]
            unit_limit = min(len(units), actions.unit_actions.shape[2])
            for unit, unit_action in enumerate(units[:unit_limit]):
                if batch.active_units[environment_index, seat, unit]:
                    actions.unit_actions[environment_index, seat, unit] = _unit_row(
                        unit_action
                    )
            market = list(raw.get("market") or [])[: self.max_orders]
            actions.market_lengths[environment_index, seat] = len(market)
            for slot, market_action in enumerate(market):
                actions.market_actions[environment_index, seat, slot] = _market_row(
                    market_action
                )
        return actions

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
    "NativeFileAgentPolicy",
    "load_agent_file",
    "load_agent_module",
    "snapshot_observation",
]
