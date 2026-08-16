"""Task-level action abstractions for extensible rule-based policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from .vec_env import Batch, Item, UnitOp

if TYPE_CHECKING:
    from .rule_based import StrategicIntent


class TaskKind(IntEnum):
    """Stable objectives proposed by rules and fulfilled by the executor."""

    NONE = 0
    WATER = 1
    FEED = 2
    CARE = 3
    HARVEST = 4
    COLLECT_FERTILIZER = 5
    CLEAR_WEED = 6
    PLANT = 7
    FERTILIZE = 8
    BUILD_COOP = 9
    BUILD_PASTURE = 10
    PLACE_ANIMAL = 11
    FETCH_ITEM = 12
    DEPOSIT_INVENTORY = 13


@dataclass(frozen=True, slots=True)
class TaskBatch:
    """Fixed-capacity task proposals for every environment and player.

    The first ``board_size**2`` slots correspond directly to board tiles. Rules
    arbitrate for those slots by priority, so at most one current objective is
    attached to a tile. Extra slots hold global logistics tasks such as fetching
    wheat from the shed.
    """

    active: NDArray[np.bool_]
    kind: NDArray[np.int16]
    target_x: NDArray[np.int16]
    target_y: NDArray[np.int16]
    item: NDArray[np.int16]
    quantity: NDArray[np.int64]
    priority: NDArray[np.float32]
    deadline: NDArray[np.int16]
    estimated_value: NDArray[np.float32]
    required_item: NDArray[np.int16]
    required_count: NDArray[np.int64]
    exclusive: NDArray[np.bool_]
    board_size: int
    tile_slots: int

    @classmethod
    def allocate(
        cls,
        num_envs: int,
        players: int,
        board_size: int,
        extra_slots: int = 12,
    ) -> TaskBatch:
        if min(num_envs, players, board_size) < 1:
            raise ValueError("task batch dimensions must be positive")
        if extra_slots < 0:
            raise ValueError("extra_slots cannot be negative")
        tile_slots = board_size * board_size
        shape = (num_envs, players, tile_slots + extra_slots)
        tasks = cls(
            active=np.zeros(shape, dtype=np.bool_),
            kind=np.zeros(shape, dtype=np.int16),
            target_x=np.zeros(shape, dtype=np.int16),
            target_y=np.zeros(shape, dtype=np.int16),
            item=np.full(shape, -1, dtype=np.int16),
            quantity=np.ones(shape, dtype=np.int64),
            priority=np.full(shape, -np.inf, dtype=np.float32),
            deadline=np.full(shape, -1, dtype=np.int16),
            estimated_value=np.zeros(shape, dtype=np.float32),
            required_item=np.full(shape, -1, dtype=np.int16),
            required_count=np.zeros(shape, dtype=np.int64),
            exclusive=np.ones(shape, dtype=np.bool_),
            board_size=board_size,
            tile_slots=tile_slots,
        )
        y, x = np.indices((board_size, board_size), dtype=np.int16)
        tasks.target_x[..., :tile_slots] = x.reshape(-1)
        tasks.target_y[..., :tile_slots] = y.reshape(-1)
        return tasks

    @property
    def capacity(self) -> int:
        return self.active.shape[-1]

    def clear(self) -> None:
        """Clear proposals while retaining tile coordinates and allocations."""

        self.active.fill(False)
        self.kind.fill(TaskKind.NONE)
        self.item.fill(-1)
        self.quantity.fill(1)
        self.priority.fill(-np.inf)
        self.deadline.fill(-1)
        self.estimated_value.fill(0)
        self.required_item.fill(-1)
        self.required_count.fill(0)
        self.exclusive.fill(True)

    def propose_tiles(
        self,
        kind: TaskKind,
        mask: NDArray[np.bool_],
        priority: float | NDArray[np.float32],
        *,
        item: int = -1,
        quantity: int = 1,
        deadline: int = -1,
        estimated_value: float = 0.0,
        required_item: int = -1,
        required_count: int = 0,
        exclusive: bool = True,
    ) -> None:
        """Propose one task per matching tile, replacing lower priorities."""

        expected = (*self.active.shape[:2], self.board_size, self.board_size)
        if mask.shape != expected:
            raise ValueError(f"tile proposal mask must have shape {expected}")
        flat_mask = mask.reshape(*mask.shape[:2], self.tile_slots)
        flat_priority = np.broadcast_to(priority, mask.shape).reshape(
            *mask.shape[:2], self.tile_slots
        )
        replace = flat_mask & (flat_priority > self.priority[..., : self.tile_slots])
        slots = np.nonzero(replace)
        self.active[..., : self.tile_slots][slots] = True
        self.kind[..., : self.tile_slots][slots] = kind
        self.item[..., : self.tile_slots][slots] = item
        self.quantity[..., : self.tile_slots][slots] = quantity
        self.priority[..., : self.tile_slots][slots] = flat_priority[slots]
        self.deadline[..., : self.tile_slots][slots] = deadline
        self.estimated_value[..., : self.tile_slots][slots] = estimated_value
        self.required_item[..., : self.tile_slots][slots] = required_item
        self.required_count[..., : self.tile_slots][slots] = required_count
        self.exclusive[..., : self.tile_slots][slots] = exclusive

    def set_global(
        self,
        extra_slot: int,
        active: NDArray[np.bool_],
        kind: TaskKind,
        target_x: int | NDArray[np.int16],
        target_y: int | NDArray[np.int16],
        priority: float | NDArray[np.float32],
        *,
        item: int = -1,
        quantity: int | NDArray[np.int64] = 1,
        deadline: int = -1,
        estimated_value: float = 0.0,
        required_item: int = -1,
        required_count: int = 0,
        exclusive: bool = True,
    ) -> None:
        """Set one named extra-slot task across a batch."""

        if extra_slot < 0 or self.tile_slots + extra_slot >= self.capacity:
            raise IndexError("global task slot is outside the configured capacity")
        if active.shape != self.active.shape[:2]:
            raise ValueError("global task mask does not match the task batch")
        slot = self.tile_slots + extra_slot
        self.active[..., slot] = active
        self.kind[..., slot] = np.where(active, kind, TaskKind.NONE)
        self.target_x[..., slot] = target_x
        self.target_y[..., slot] = target_y
        self.item[..., slot] = item
        self.quantity[..., slot] = quantity
        self.priority[..., slot] = np.where(active, priority, -np.inf)
        self.deadline[..., slot] = deadline
        self.estimated_value[..., slot] = estimated_value
        self.required_item[..., slot] = required_item
        self.required_count[..., slot] = required_count
        self.exclusive[..., slot] = exclusive


@dataclass(frozen=True, slots=True)
class TaskAssignments:
    """One exclusive task index per active unit, or ``-1`` when unassigned."""

    task_index: NDArray[np.int64]
    score: NDArray[np.float32]


class TaskRule(Protocol):
    """Extension point for independent rules that propose farm tasks."""

    def propose(
        self,
        batch: Batch,
        intent: StrategicIntent,
        tasks: TaskBatch,
    ) -> None:
        """Add or replace task proposals using task priorities."""


class TaskScheduler:
    """Assign exclusive tasks to units by priority, eligibility, and distance."""

    def __init__(self, board_size: int, shed_capacity: int = 100) -> None:
        self.board_size = board_size
        self.shed_capacity = shed_capacity
        self._shape: tuple[int, int, int] | None = None
        self._assignments: TaskAssignments | None = None

    def assign(self, batch: Batch, tasks: TaskBatch) -> TaskAssignments:
        n, players, unit_count = batch.active_units.shape
        shape = (n, players, unit_count)
        if self._assignments is None or self._shape != shape:
            self._assignments = TaskAssignments(
                task_index=np.full(shape, -1, dtype=np.int64),
                score=np.full(shape, -np.inf, dtype=np.float32),
            )
            self._shape = shape
        assignments = self._assignments
        assignments.task_index.fill(-1)
        assignments.score.fill(-np.inf)

        units = batch.observation_views.units[:, :, 0]
        scale = max(1, self.board_size - 1)
        unit_x = np.rint(units[..., 2] * scale).astype(np.int16)
        unit_y = np.rint(units[..., 3] * scale).astype(np.int16)
        inventories = np.rint(units[..., 5:17] * self.shed_capacity).astype(
            np.int64
        )

        # Task scoring is dense and batched. The final conflict resolution is a
        # deliberately small ragged loop because active unit/task counts vary.
        distance = np.abs(unit_x[..., None] - tasks.target_x[..., None, :])
        distance += np.abs(unit_y[..., None] - tasks.target_y[..., None, :])
        scores = tasks.priority[..., None, :] * 1_000.0 - distance
        scores = scores.astype(np.float32, copy=False)
        eligible = batch.active_units[..., None] & tasks.active[..., None, :]
        for item in range(12):
            required = tasks.required_item == item
            if required.any():
                enough = inventories[..., item, None] >= tasks.required_count[..., None, :]
                eligible &= ~required[..., None, :] | enough
        deposit = tasks.kind == TaskKind.DEPOSIT_INVENTORY
        carrying_anything = inventories.sum(axis=-1) > 0
        eligible &= ~deposit[..., None, :] | carrying_anything[..., None]
        scores[~eligible] = -np.inf

        for environment in range(n):
            for player in range(players):
                available_units = set(np.flatnonzero(batch.active_units[environment, player]))
                ordered_tasks = np.flatnonzero(tasks.active[environment, player])
                ordered_tasks = ordered_tasks[
                    np.argsort(
                        -tasks.priority[environment, player, ordered_tasks],
                        kind="stable",
                    )
                ]
                for task in ordered_tasks:
                    if not available_units:
                        break
                    candidates = np.asarray(sorted(available_units), dtype=np.int64)
                    candidate_scores = scores[environment, player, candidates, task]
                    candidate_order = np.argsort(-candidate_scores, kind="stable")
                    for best_offset in candidate_order:
                        if not np.isfinite(candidate_scores[best_offset]):
                            break
                        unit = int(candidates[best_offset])
                        assignments.task_index[environment, player, unit] = task
                        assignments.score[environment, player, unit] = candidate_scores[
                            best_offset
                        ]
                        available_units.remove(unit)
                        if tasks.exclusive[environment, player, task]:
                            break
        return assignments


class TaskExecutor:
    """Convert task assignments into masked movement or interaction actions."""

    _OPERATIONS = {
        TaskKind.WATER: UnitOp.WATER,
        TaskKind.FEED: UnitOp.FEED,
        TaskKind.CARE: UnitOp.CARE,
        TaskKind.HARVEST: UnitOp.HARVEST,
        TaskKind.COLLECT_FERTILIZER: UnitOp.COLLECT_FERTILIZER,
        TaskKind.CLEAR_WEED: UnitOp.DIG,
        TaskKind.PLANT: UnitOp.PLANT,
        TaskKind.FERTILIZE: UnitOp.FERTILIZE,
        TaskKind.BUILD_COOP: UnitOp.BUILD_COOP,
        TaskKind.BUILD_PASTURE: UnitOp.BUILD_PASTURE,
        TaskKind.PLACE_ANIMAL: UnitOp.PLACE,
        TaskKind.FETCH_ITEM: UnitOp.PICKUP,
        TaskKind.DEPOSIT_INVENTORY: UnitOp.DROP,
    }
    _ARGUMENT_OPERATIONS = {
        UnitOp.PICKUP,
        UnitOp.PLACE,
        UnitOp.PLANT,
    }

    def __init__(self, board_size: int) -> None:
        self.board_size = board_size

    def execute(
        self,
        batch: Batch,
        tasks: TaskBatch,
        assignments: TaskAssignments,
        unit_actions: NDArray[np.int64],
    ) -> None:
        unit_actions.fill(0)
        units = batch.observation_views.units[:, :, 0]
        scale = max(1, self.board_size - 1)
        unit_x = np.rint(units[..., 2] * scale).astype(np.int16)
        unit_y = np.rint(units[..., 3] * scale).astype(np.int16)
        n, players, unit_count = batch.active_units.shape

        for environment in range(n):
            for player in range(players):
                for unit in range(unit_count):
                    task_index = assignments.task_index[environment, player, unit]
                    if task_index < 0:
                        continue
                    target_x = tasks.target_x[environment, player, task_index]
                    target_y = tasks.target_y[environment, player, task_index]
                    x = unit_x[environment, player, unit]
                    y = unit_y[environment, player, unit]
                    kind = TaskKind(tasks.kind[environment, player, task_index])
                    if kind == TaskKind.DEPOSIT_INVENTORY:
                        half = self.board_size // 2
                        centers = (max(0, half - 1), half)
                        if x in centers and y in centers:
                            if batch.mask_views.unit_ops[
                                environment, player, unit, UnitOp.DROP
                            ]:
                                unit_actions[environment, player, unit, 0] = UnitOp.DROP
                            continue
                        target_x = centers[0] if x <= centers[0] else centers[1]
                        target_y = centers[0] if y <= centers[0] else centers[1]
                    if x != target_x or y != target_y:
                        operation = self._movement(x, y, target_x, target_y)
                        if batch.mask_views.unit_ops[
                            environment, player, unit, operation
                        ]:
                            unit_actions[environment, player, unit, 0] = operation
                        continue

                    operation = self._OPERATIONS.get(kind, UnitOp.PASS)
                    item = int(tasks.item[environment, player, task_index])
                    count = int(tasks.quantity[environment, player, task_index])
                    if not batch.mask_views.unit_ops[
                        environment, player, unit, operation
                    ]:
                        continue
                    if operation in self._ARGUMENT_OPERATIONS:
                        if item < 0 or not batch.mask_views.unit_args[
                            environment, player, unit, operation, item
                        ]:
                            continue
                    unit_actions[environment, player, unit] = (
                        operation,
                        max(0, item),
                        count,
                    )

    @staticmethod
    def _movement(x: int, y: int, target_x: int, target_y: int) -> UnitOp:
        if x < target_x:
            return UnitOp.EAST
        if x > target_x:
            return UnitOp.WEST
        if y < target_y:
            return UnitOp.SOUTH
        if y > target_y:
            return UnitOp.NORTH
        return UnitOp.PASS


__all__ = [
    "TaskAssignments",
    "TaskBatch",
    "TaskExecutor",
    "TaskKind",
    "TaskRule",
    "TaskScheduler",
]
