"""Batched native adapter for the preserved V16-RC5 baseline."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Sequence
import uuid

import numpy as np
from numpy.typing import NDArray

from .rule_based import RuleActions
from .vec_env import Batch, Item, MarketOp, UnitOp


FRONT_RUN_ITEMS = (Item.MELON, Item.MILK, Item.STRAWBERRY, Item.WOOL)
SHOP_DEMAND = np.asarray(
    (
        # wheat, carrot, tomato, strawberry, melon, egg, milk, wool, fertilizer
        (1, 0, 0, 0, 0, 1, 0, 0, 0),  # bakery
        (1, 0, 0, 1, 0, 1, 0, 0, 0),  # brunch spot
        (1, 1, 1, 1, 0, 0, 0, 0, 0),  # farmers market
        (1, 0, 0, 1, 0, 0, 1, 0, 0),  # ice cream shop
        (0, 2, 0, 0, 0, 0, 0, 0, 0),  # pet cafe
        (1, 0, 1, 0, 0, 0, 1, 0, 0),  # pizza shop
        (0, 0, 0, 1, 0, 0, 1, 0, 0),  # smoothie shop
        (0, 0, 0, 0, 0, 0, 0, 2, 0),  # yarn store
    ),
    dtype=np.int64,
)


def load_v16_actions(path: Path) -> Sequence[dict[str, Any]]:
    """Load the immutable action trace embedded in the V16 submission."""

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


class NativeV16Policy:
    """Emit V16 actions directly into tensors for every batch slot."""

    def __init__(
        self,
        actions: Sequence[dict[str, Any]],
        *,
        episode_steps: int = 720,
        board_size: int = 10,
        shed_capacity: int = 100,
        max_orders: int = 10,
    ) -> None:
        self.episode_steps = episode_steps
        self.board_size = board_size
        self.shed_capacity = shed_capacity
        self.max_orders = max_orders
        self.trace_steps = len(actions)
        max_units = max(
            1 + len((action or {}).get("hands", ()) or ()) for action in actions
        )
        self.trace_units = np.zeros(
            (self.trace_steps, max_units, 3), dtype=np.int64
        )
        self.trace_market = np.zeros(
            (self.trace_steps, max_orders, 3), dtype=np.int64
        )
        self.trace_market_lengths = np.zeros(self.trace_steps, dtype=np.int64)
        for step, action in enumerate(actions):
            action = action or {}
            units = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
            for unit, raw in enumerate(units[:max_units]):
                self.trace_units[step, unit] = _unit_row(raw)
            market = list(action.get("market") or [])[:max_orders]
            self.trace_market_lengths[step] = len(market)
            for slot, raw in enumerate(market):
                self.trace_market[step, slot] = _market_row(raw)

        self._shape: tuple[int, int, int] | None = None
        self._actions: RuleActions | None = None
        self._repair_active: NDArray[np.bool_] | None = None
        self._repair_start: NDArray[np.int64] | None = None
        self._repair_intended: NDArray[np.int64] | None = None
        self._due_step: NDArray[np.int64] | None = None
        self._due: NDArray[np.int64] | None = None

    def reset(self) -> None:
        """Clear per-environment weed and market transactions."""

        if self._repair_active is not None:
            self._repair_active.fill(False)
        if self._repair_start is not None:
            self._repair_start.fill(-1)
        if self._repair_intended is not None:
            self._repair_intended.fill(0)
        if self._due_step is not None:
            self._due_step.fill(-1)
        if self._due is not None:
            self._due.fill(0)

    def act(self, batch: Batch) -> RuleActions:
        """Return V16 tensor actions for the current synchronous batch step."""

        actions = self._buffers(batch)
        assert self._repair_active is not None
        assert self._repair_start is not None
        assert self._repair_intended is not None
        assert self._due_step is not None
        assert self._due is not None
        step = int(
            np.rint(
                batch.observation_views.global_features[0, 0, 0]
                * max(1, self.episode_steps - 1)
            )
        )
        trace_step = min(max(step, 0), self.trace_steps - 1)
        actions.unit_actions.fill(0)
        unit_limit = min(actions.unit_actions.shape[2], self.trace_units.shape[1])
        actions.unit_actions[:, :, :unit_limit] = self.trace_units[
            trace_step, :unit_limit
        ]
        actions.unit_actions[~batch.active_units] = 0
        actions.market_actions[:] = self.trace_market[trace_step]
        actions.market_lengths.fill(self.trace_market_lengths[trace_step])

        self._apply_weed_repair(batch, actions.unit_actions, step)
        self._apply_repayment(actions, step)
        self._apply_front_run(batch, actions, step)
        return actions

    def _buffers(self, batch: Batch) -> RuleActions:
        shape = batch.active_units.shape
        if self._actions is None or self._shape != shape:
            n, players, units = shape
            self._actions = RuleActions(
                unit_actions=np.zeros((n, players, units, 3), dtype=np.int64),
                market_actions=np.zeros(
                    (n, players, self.max_orders, 3), dtype=np.int64
                ),
                market_lengths=np.zeros((n, players), dtype=np.int64),
            )
            self._repair_active = np.zeros(shape, dtype=np.bool_)
            self._repair_start = np.full(shape, -1, dtype=np.int64)
            self._repair_intended = np.zeros((*shape, 3), dtype=np.int64)
            self._due_step = np.full((n, players), -1, dtype=np.int64)
            self._due = np.zeros((n, players, 9), dtype=np.int64)
            self._shape = shape
        return self._actions

    def _apply_weed_repair(
        self,
        batch: Batch,
        unit_actions: NDArray[np.int64],
        step: int,
    ) -> None:
        assert self._repair_active is not None
        assert self._repair_start is not None
        assert self._repair_intended is not None
        active = self._repair_active & batch.active_units
        age = step - self._repair_start
        intended = active & (age == 1)
        unit_actions[intended] = self._repair_intended[intended]
        replay = active & (age >= 2) & (age <= 9)
        if replay.any():
            trace_step = min(max(step - 1, 0), self.trace_steps - 1)
            _, _, unit_indices = np.nonzero(replay)
            valid = unit_indices < self.trace_units.shape[1]
            environments, players, units = np.nonzero(replay)
            unit_actions[
                environments[valid], players[valid], units[valid]
            ] = self.trace_units[trace_step, units[valid]]
        expired = self._repair_active & (
            ~batch.active_units | (age > 9)
        )
        self._repair_active[expired] = False

        candidate = (
            batch.active_units
            & ~self._repair_active
            & (
                (unit_actions[..., 0] == UnitOp.BUILD_PASTURE)
                | (unit_actions[..., 0] == UnitOp.PLANT)
            )
        )
        if not candidate.any():
            return
        units = batch.observation_views.units[:, :, 0]
        scale = max(1, self.board_size - 1)
        x = np.rint(units[..., 2] * scale).astype(np.int64)
        y = np.rint(units[..., 3] * scale).astype(np.int64)
        environments, players, unit_indices = np.nonzero(candidate)
        weeds = (
            batch.observation_views.tiles[
                environments,
                players,
                0,
                y[environments, players, unit_indices],
                x[environments, players, unit_indices],
                2,
            ]
            > 0.5
        )
        environments = environments[weeds]
        players = players[weeds]
        unit_indices = unit_indices[weeds]
        if not environments.size:
            return
        index = (environments, players, unit_indices)
        self._repair_active[index] = True
        self._repair_start[index] = step
        self._repair_intended[index] = unit_actions[index]
        unit_actions[index] = (UnitOp.DIG, 0, 0)

    def _apply_repayment(self, actions: RuleActions, step: int) -> None:
        assert self._due_step is not None
        assert self._due is not None
        repay = self._due_step == step
        if not repay.any():
            return
        for item in FRONT_RUN_ITEMS:
            remaining = np.where(repay, self._due[..., item], 0)
            for slot in range(self.max_orders):
                row = actions.market_actions[..., slot, :]
                matching = (
                    (row[..., 0] == MarketOp.SELL)
                    & (row[..., 1] == item)
                    & (remaining > 0)
                )
                reduction = np.minimum(row[..., 2], remaining)
                row[..., 2] -= np.where(matching, reduction, 0)
                remaining -= np.where(matching, reduction, 0)
                remove = matching & (row[..., 2] <= 0)
                row[remove] = 0
        self._compact_market(actions)
        self._due_step[repay] = -1
        self._due[repay] = 0

    def _apply_front_run(
        self, batch: Batch, actions: RuleActions, step: int
    ) -> None:
        assert self._due_step is not None
        assert self._due is not None
        expired = (self._due_step >= 0) & (self._due_step < step)
        self._due_step[expired] = -1
        self._due[expired] = 0
        shed = np.rint(
            batch.observation_views.private[..., :9] * self.shed_capacity
        ).astype(np.int64)
        shops = np.rint(
            batch.observation_views.global_features[..., 22:30] * 8
        ).astype(np.int64)
        moved_any = np.zeros(actions.market_lengths.shape, dtype=np.bool_)
        for item in FRONT_RUN_ITEMS:
            future = min(step + 1, self.trace_steps - 1)
            future_rows = self.trace_market[future]
            target = int(
                future_rows[
                    (future_rows[:, 0] == MarketOp.SELL)
                    & (future_rows[:, 1] == item),
                    2,
                ].sum()
            )
            if target <= 0:
                continue
            demand = np.zeros(actions.market_lengths.shape, dtype=np.int64)
            if item != Item.FERTILIZER and step % 24 == 0:
                demand += 1
            if step % 4 == 0:
                demand += np.tensordot(
                    shops, SHOP_DEMAND[:, int(item)], axes=([-1], [0])
                )
            pickup = (
                (actions.unit_actions[..., 0] == UnitOp.PICKUP)
                & (actions.unit_actions[..., 1] == item)
            )
            pickup_reserve = (
                actions.unit_actions[..., 2] * pickup
            ).sum(axis=-1)
            rows = actions.market_actions
            selling = (rows[..., 0] == MarketOp.SELL) & (
                rows[..., 1] == item
            )
            existing_quantity = (rows[..., 2] * selling).sum(axis=-1)
            quantity = np.minimum(
                target,
                np.maximum(0, shed[..., item] - pickup_reserve - existing_quantity),
            )
            quantity = np.where(demand == 0, quantity, 0)
            add_existing = (quantity > 0) & selling.any(axis=-1)
            environments, players = np.nonzero(add_existing)
            if environments.size:
                slots = np.argmax(selling[environments, players], axis=-1)
                actions.market_actions[environments, players, slots, 2] += quantity[
                    environments, players
                ]
            append = (quantity > 0) & ~selling.any(axis=-1) & (
                actions.market_lengths < self.max_orders
            )
            environments, players = np.nonzero(append)
            if environments.size:
                slots = actions.market_lengths[environments, players]
                actions.market_actions[environments, players, slots] = np.stack(
                    (
                        np.full(slots.shape, MarketOp.SELL, dtype=np.int64),
                        np.full(slots.shape, item, dtype=np.int64),
                        quantity[environments, players],
                    ),
                    axis=-1,
                )
                actions.market_lengths[environments, players] += 1
            moved = add_existing | append
            self._due[..., item] = np.where(moved, quantity, self._due[..., item])
            moved_any |= moved
        self._due_step[moved_any] = step + 1

    @staticmethod
    def _compact_market(actions: RuleActions) -> None:
        valid = actions.market_actions[..., 0] != MarketOp.NONE
        order = np.argsort(~valid, axis=-1, kind="stable")
        actions.market_actions[:] = np.take_along_axis(
            actions.market_actions, order[..., None], axis=-2
        )
        actions.market_lengths[:] = valid.sum(axis=-1)


__all__ = ["NativeV16Policy", "load_v16_actions"]
