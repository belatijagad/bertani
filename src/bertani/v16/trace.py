"""Load and encode the immutable action trace embedded in V16-RC5."""

from __future__ import annotations

import importlib.util
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..vec_env import Item, MarketOp, UnitOp


@dataclass(frozen=True, slots=True)
class V16Trace:
    """Contiguous one-time initialization arrays consumed by Rust."""

    unit_actions: NDArray[np.int16]
    market_actions: NDArray[np.int16]
    market_lengths: NDArray[np.int16]


def load_v16_actions(path: Path) -> Sequence[dict[str, Any]]:
    """Load the immutable raw action trace embedded in the V16 submission."""

    resolved = path.resolve()
    spec = importlib.util.spec_from_file_location(
        f"bertani_v16_trace_{uuid.uuid4().hex}", resolved
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load V16 baseline: {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actions = getattr(module, "_ACTIONS", None)
    if not isinstance(actions, (list, tuple)) or not actions:
        raise ValueError(f"V16 baseline has no decoded action trace: {resolved}")
    return actions


def _unit_row(raw: Sequence[Any] | None) -> tuple[int, int, int]:
    action = list(raw or ("PASS",))
    operation = UnitOp[str(action[0])]
    item = 0
    count = 0
    if operation in {UnitOp.PICKUP, UnitOp.PLACE, UnitOp.PLANT}:
        item = int(Item[str(action[1])])
        count = int(action[2]) if len(action) >= 3 else 1
    return int(operation), item, count


def _market_row(raw: Sequence[Any]) -> tuple[int, int, int]:
    action = list(raw)
    operation = MarketOp[str(action[0])]
    if operation in {MarketOp.HIRE, MarketOp.BUY_LAND}:
        return int(operation), 0, 0
    return int(operation), int(Item[str(action[1])]), int(action[2])


def encode_v16_trace(
    actions: Sequence[dict[str, Any]], *, max_orders: int = 10
) -> V16Trace:
    """Encode submission dictionaries once into stable native action IDs."""

    if not actions:
        raise ValueError("V16 action trace cannot be empty")
    if max_orders <= 0:
        raise ValueError("max_orders must be positive")
    steps = len(actions)
    max_units = max(
        1 + len((action or {}).get("hands", ()) or ()) for action in actions
    )
    units = np.zeros((steps, max_units, 3), dtype=np.int16)
    market = np.zeros((steps, max_orders, 3), dtype=np.int16)
    lengths = np.zeros(steps, dtype=np.int16)
    for step, raw_action in enumerate(actions):
        action = raw_action or {}
        unit_rows = [
            action.get("farmer") or ["PASS"],
            *(action.get("hands") or []),
        ]
        for unit, raw in enumerate(unit_rows[:max_units]):
            units[step, unit] = _unit_row(raw)
        market_rows = list(action.get("market") or [])[:max_orders]
        lengths[step] = len(market_rows)
        for order, raw in enumerate(market_rows):
            market[step, order] = _market_row(raw)
    return V16Trace(units, market, lengths)


def load_v16_trace(path: Path, *, max_orders: int = 10) -> V16Trace:
    """Load and encode a preserved V16 submission."""

    return encode_v16_trace(load_v16_actions(path), max_orders=max_orders)


__all__ = [
    "V16Trace",
    "encode_v16_trace",
    "load_v16_actions",
    "load_v16_trace",
]
