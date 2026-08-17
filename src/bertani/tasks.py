"""Task-level action abstractions for extensible rule-based policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

try:
    from ._rust import schedule_tasks as _native_schedule_tasks
except (ImportError, ModuleNotFoundError):
    _native_schedule_tasks = None

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


class WorkRole(IntEnum):
    """Soft worker specialization used by the task scheduler."""

    ANY = 0
    LOGISTICS = 1
    LIVESTOCK = 2
    FIELD = 3


class WorkZone(IntEnum):
    """Preferred board region for field workers."""

    ANY = -1
    NW = 0
    NE = 1
    SW = 2
    SE = 3


@dataclass(frozen=True, slots=True)
class WorkforcePlan:
    """Per-unit soft role and territory preferences.

    Preferences affect routing only among tasks in the same urgency band, so
    every worker remains available for survival-critical work.
    """

    role: NDArray[np.int16]
    zone: NDArray[np.int16]
    role_bonus: float = 4.0
    zone_bonus: float = 3.0
    reserved_by_kind: NDArray[np.int16] | None = None


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
    work_role: NDArray[np.int16]
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
            work_role=np.full(shape, WorkRole.ANY, dtype=np.int16),
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
        self.work_role.fill(WorkRole.ANY)

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
        work_role: WorkRole = WorkRole.ANY,
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
        self.work_role[..., : self.tile_slots][slots] = work_role

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
        work_role: WorkRole = WorkRole.ANY,
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
        self.work_role[..., slot] = np.where(active, work_role, WorkRole.ANY)


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


class WorkforcePlanner(Protocol):
    """Extension point for assigning soft roles and territories to units."""

    def __call__(
        self,
        batch: Batch,
        intent: StrategicIntent,
        tasks: TaskBatch,
    ) -> WorkforcePlan:
        """Return per-unit preferences for the current scheduling turn."""


class TaskScheduler:
    """Assign urgency bands while preserving valid within-day worker jobs."""

    def __init__(
        self,
        board_size: int,
        shed_capacity: int = 100,
        continuity_bonus: float = 1.0,
        episode_steps: int = 720,
        turns_per_day: int = 24,
    ) -> None:
        self.board_size = board_size
        self.shed_capacity = shed_capacity
        self.continuity_bonus = continuity_bonus
        self.last_step = max(1, episode_steps - 1)
        self.turns_per_day = turns_per_day
        self._shape: tuple[int, int, int] | None = None
        self._assignments: TaskAssignments | None = None
        self._previous_task: NDArray[np.int64] | None = None

    def assign(
        self,
        batch: Batch,
        tasks: TaskBatch,
        workforce: WorkforcePlan | None = None,
    ) -> TaskAssignments:
        n, players, unit_count = batch.active_units.shape
        shape = (n, players, unit_count)
        if self._assignments is None or self._shape != shape:
            self._assignments = TaskAssignments(
                task_index=np.full(shape, -1, dtype=np.int64),
                score=np.full(shape, -np.inf, dtype=np.float32),
            )
            self._previous_task = np.full(shape, -1, dtype=np.int64)
            self._shape = shape
        assignments = self._assignments
        assert self._previous_task is not None
        assignments.task_index.fill(-1)
        assignments.score.fill(-np.inf)

        step = np.rint(
            batch.observation_views.global_features[..., 0] * self.last_step
        ).astype(np.int64)
        new_day = (step % self.turns_per_day) == 0
        self._previous_task[new_day] = -1
        self._previous_task[~batch.active_units] = -1

        units = batch.observation_views.units[:, :, 0]
        scale = max(1, self.board_size - 1)
        unit_x = np.rint(units[..., 2] * scale).astype(np.int16)
        unit_y = np.rint(units[..., 3] * scale).astype(np.int16)
        inventories = np.rint(units[..., 5:17] * self.shed_capacity).astype(
            np.int64
        )
        if workforce is None:
            unit_role = np.full(shape, WorkRole.ANY, dtype=np.int16)
            unit_zone = np.full(shape, WorkZone.ANY, dtype=np.int16)
            role_bonus = 0.0
            zone_bonus = 0.0
        else:
            if workforce.role.shape != shape or workforce.zone.shape != shape:
                raise ValueError("workforce plan must match active unit shape")
            unit_role = np.ascontiguousarray(workforce.role, dtype=np.int16)
            unit_zone = np.ascontiguousarray(workforce.zone, dtype=np.int16)
            role_bonus = workforce.role_bonus
            zone_bonus = workforce.zone_bonus
        if workforce is None or workforce.reserved_by_kind is None:
            reserved_by_kind = np.zeros(
                (*shape[:2], max(TaskKind) + 1), dtype=np.int16
            )
        else:
            expected = (*shape[:2], max(TaskKind) + 1)
            if workforce.reserved_by_kind.shape != expected:
                raise ValueError(
                    f"capacity reservation must have shape {expected}"
                )
            reserved_by_kind = np.ascontiguousarray(
                workforce.reserved_by_kind, dtype=np.int16
            )

        half = self.board_size // 2
        task_zone = (
            (tasks.target_y >= half).astype(np.int16) * 2
            + (tasks.target_x >= half).astype(np.int16)
        )

        if _native_schedule_tasks is not None:
            _native_schedule_tasks(
                unit_x,
                unit_y,
                inventories,
                tasks.priority,
                batch.active_units,
                tasks.active,
                tasks.exclusive,
                tasks.target_x,
                tasks.target_y,
                tasks.required_item,
                tasks.required_count,
                tasks.kind,
                tasks.work_role,
                unit_role,
                unit_zone,
                task_zone,
                role_bonus,
                zone_bonus,
                reserved_by_kind,
                self._previous_task,
                self.continuity_bonus,
                tasks.board_size,
                assignments.task_index,
                assignments.score,
            )
        else:
            distance = np.abs(unit_x[..., None] - tasks.target_x[..., None, :])
            distance += np.abs(unit_y[..., None] - tasks.target_y[..., None, :])
            scores = (
                tasks.priority[..., None, :] - distance
            ).astype(np.float32, copy=False)
            role_match = (
                (unit_role[..., None] != WorkRole.ANY)
                & (unit_role[..., None] == tasks.work_role[..., None, :])
            )
            field_task = (
                (tasks.kind == TaskKind.CLEAR_WEED)
                & (np.arange(tasks.capacity) < tasks.tile_slots)
            )
            zone_match = (
                (unit_zone[..., None] != WorkZone.ANY)
                & field_task[..., None, :]
                & (unit_zone[..., None] == task_zone[..., None, :])
            )
            scores += role_match * role_bonus + zone_match * zone_bonus
            task_indices = np.arange(tasks.capacity, dtype=np.int64)
            continuing = (
                self._previous_task[..., None] == task_indices
            )
            scores += continuing * self.continuity_bonus
            eligible = batch.active_units[..., None] & tasks.active[..., None, :]
            for item_index in range(inventories.shape[-1]):
                required = tasks.required_item == item_index
                if required.any():
                    enough = (
                        inventories[..., item_index, None]
                        >= tasks.required_count[..., None, :]
                    )
                    eligible &= ~required[..., None, :] | enough
            deposit = tasks.kind == TaskKind.DEPOSIT_INVENTORY
            carrying_anything = inventories.sum(axis=-1) > 0
            eligible &= ~deposit[..., None, :] | carrying_anything[..., None]
            scores[~eligible] = -np.inf
            self._assign_python(
                batch, tasks, scores, assignments, reserved_by_kind
            )
            self._prefer_unclaimed_local_tasks(
                batch,
                tasks,
                assignments,
                unit_x,
                unit_y,
                inventories,
            )
        self._preserve_workflow_contracts(
            batch, tasks, assignments, unit_x, unit_y, inventories
        )
        self._previous_task[...] = assignments.task_index
        self._previous_task[~batch.active_units] = -1
        return assignments

    def _preserve_workflow_contracts(
        self,
        batch: Batch,
        tasks: TaskBatch,
        assignments: TaskAssignments,
        unit_x: NDArray[np.int16],
        unit_y: NDArray[np.int16],
        inventories: NDArray[np.int64],
    ) -> None:
        """Keep field workflows owned while they remain in the top urgency band."""

        assert self._previous_task is not None
        workflow_kinds = {
            int(TaskKind.HARVEST),
            int(TaskKind.CLEAR_WEED),
            int(TaskKind.PLANT),
        }
        n, players, _ = batch.active_units.shape
        for environment in range(n):
            for player in range(players):
                active_tasks = np.flatnonzero(tasks.active[environment, player])
                if active_tasks.size == 0:
                    continue
                top_band = np.floor(
                    tasks.priority[environment, player, active_tasks] / 10.0
                ).max()
                locked_units: set[int] = set()
                for unit in np.flatnonzero(batch.active_units[environment, player]):
                    previous = int(self._previous_task[environment, player, unit])
                    if previous < 0 or previous >= tasks.tile_slots:
                        continue
                    if not tasks.active[environment, player, previous]:
                        continue
                    if int(tasks.kind[environment, player, previous]) not in workflow_kinds:
                        continue
                    if np.floor(tasks.priority[environment, player, previous] / 10.0) < top_band:
                        continue
                    required = int(tasks.required_item[environment, player, previous])
                    if required >= 0 and inventories[
                        environment, player, unit, required
                    ] < tasks.required_count[environment, player, previous]:
                        continue
                    current = int(assignments.task_index[environment, player, unit])
                    if current == previous:
                        locked_units.add(int(unit))
                        continue
                    owners = np.flatnonzero(
                        assignments.task_index[environment, player] == previous
                    )
                    owner = int(owners[0]) if owners.size else -1
                    if owner >= 0:
                        worker_distance = abs(
                            int(unit_x[environment, player, unit])
                            - int(tasks.target_x[environment, player, previous])
                        ) + abs(
                            int(unit_y[environment, player, unit])
                            - int(tasks.target_y[environment, player, previous])
                        )
                        owner_distance = abs(
                            int(unit_x[environment, player, owner])
                            - int(tasks.target_x[environment, player, previous])
                        ) + abs(
                            int(unit_y[environment, player, owner])
                            - int(tasks.target_y[environment, player, previous])
                        )
                        if worker_distance > owner_distance:
                            continue
                    assignments.task_index[environment, player, unit] = previous
                    assignments.score[environment, player, unit] = tasks.priority[
                        environment, player, previous
                    ]
                    locked_units.add(int(unit))
                    if owner >= 0 and owner not in locked_units:
                        assignments.task_index[environment, player, owner] = current
                        assignments.score[environment, player, owner] = (
                            tasks.priority[environment, player, current]
                            if current >= 0
                            else -np.inf
                        )

    @staticmethod
    def _prefer_unclaimed_local_tasks(
        batch: Batch,
        tasks: TaskBatch,
        assignments: TaskAssignments,
        unit_x: NDArray[np.int16],
        unit_y: NDArray[np.int16],
        inventories: NDArray[np.int64],
        priority_slack: float = 5.0,
    ) -> None:
        """Do useful work underfoot when no other unit owns that tile task."""

        n, players, _ = batch.active_units.shape
        for environment in range(n):
            for player in range(players):
                active_units = np.flatnonzero(
                    batch.active_units[environment, player]
                )
                claimed = {
                    int(task)
                    for task in assignments.task_index[environment, player]
                    if task >= 0
                }
                for unit in active_units:
                    local = (
                        int(unit_y[environment, player, unit])
                        * tasks.board_size
                        + int(unit_x[environment, player, unit])
                    )
                    if (
                        local >= tasks.tile_slots
                        or not tasks.active[environment, player, local]
                        or local in claimed
                    ):
                        continue
                    required = int(
                        tasks.required_item[environment, player, local]
                    )
                    if required >= 0 and inventories[
                        environment, player, unit, required
                    ] < tasks.required_count[environment, player, local]:
                        continue
                    current = int(
                        assignments.task_index[environment, player, unit]
                    )
                    current_priority = (
                        float(tasks.priority[environment, player, current])
                        if current >= 0
                        else -np.inf
                    )
                    local_priority = float(
                        tasks.priority[environment, player, local]
                    )
                    if local_priority + priority_slack < current_priority:
                        continue
                    if current >= 0:
                        claimed.discard(current)
                    assignments.task_index[environment, player, unit] = local
                    assignments.score[environment, player, unit] = (
                        local_priority * 1_000.0
                    )
                    claimed.add(local)

    @staticmethod
    def _assign_python(
        batch: Batch,
        tasks: TaskBatch,
        scores: NDArray[np.float32],
        assignments: TaskAssignments,
        reserved_by_kind: NDArray[np.int16],
    ) -> None:
        """Portable fallback used by submission archives without the extension."""

        n, players, _ = batch.active_units.shape
        for environment in range(n):
            for player in range(players):
                available_units = set(np.flatnonzero(batch.active_units[environment, player]))
                active_tasks = np.flatnonzero(tasks.active[environment, player])
                urgency = np.floor(
                    tasks.priority[environment, player, active_tasks] / 10.0
                )
                urgency_bands = np.unique(urgency)[::-1]
                for urgency_band in urgency_bands:
                    tier_tasks = set(
                        active_tasks[urgency == urgency_band].tolist()
                    )
                    lower_tasks = active_tasks[urgency < urgency_band]
                    future_reserve = 0
                    for kind in range(reserved_by_kind.shape[-1]):
                        if urgency_band >= 12:
                            break
                        requested = int(
                            reserved_by_kind[environment, player, kind]
                        )
                        if requested <= 0:
                            continue
                        available_kind = int(
                            np.count_nonzero(
                                tasks.kind[environment, player, lower_tasks]
                                == kind
                            )
                        )
                        future_reserve += min(requested, available_kind)
                    while len(available_units) > future_reserve and tier_tasks:
                        candidates = np.asarray(sorted(available_units), dtype=np.int64)
                        candidate_tasks = np.asarray(sorted(tier_tasks), dtype=np.int64)
                        tier_scores = scores[environment, player][
                            np.ix_(candidates, candidate_tasks)
                        ]
                        best_flat = int(np.argmax(tier_scores))
                        best_score = float(tier_scores.flat[best_flat])
                        if not np.isfinite(best_score):
                            break
                        unit_offset, task_offset = np.unravel_index(best_flat, tier_scores.shape)
                        unit = int(candidates[unit_offset])
                        task = int(candidate_tasks[task_offset])
                        assignments.task_index[environment, player, unit] = task
                        assignments.score[environment, player, unit] = best_score
                        available_units.remove(unit)
                        if tasks.exclusive[environment, player, task]:
                            tier_tasks.remove(task)


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
        assigned = assignments.task_index >= 0
        safe_task = np.maximum(assignments.task_index, 0)

        def task_field(values: NDArray) -> NDArray:
            return np.take_along_axis(values, safe_task, axis=2)

        target_x = task_field(tasks.target_x).copy()
        target_y = task_field(tasks.target_y).copy()
        kind = task_field(tasks.kind)
        item = task_field(tasks.item)
        count = task_field(tasks.quantity)

        deposit = assigned & (kind == TaskKind.DEPOSIT_INVENTORY)
        half = self.board_size // 2
        low_center, high_center = max(0, half - 1), half
        at_shed = deposit & np.isin(unit_x, (low_center, high_center)) & np.isin(
            unit_y, (low_center, high_center)
        )
        target_x[deposit] = np.where(
            unit_x[deposit] <= low_center, low_center, high_center
        )
        target_y[deposit] = np.where(
            unit_y[deposit] <= low_center, low_center, high_center
        )

        moving = assigned & ~at_shed & (
            (unit_x != target_x) | (unit_y != target_y)
        )
        movement = np.where(
            unit_x < target_x,
            UnitOp.EAST,
            np.where(
                unit_x > target_x,
                UnitOp.WEST,
                np.where(unit_y < target_y, UnitOp.SOUTH, UnitOp.NORTH),
            ),
        ).astype(np.int64)

        operation_lookup = np.full(max(TaskKind) + 1, UnitOp.PASS, dtype=np.int64)
        for task_kind, operation in self._OPERATIONS.items():
            operation_lookup[task_kind] = operation
        operation = operation_lookup[kind]
        operation = np.where(moving, movement, operation)
        operation = np.where(at_shed, UnitOp.DROP, operation)

        grid = np.indices(assigned.shape)
        legal = assigned & batch.mask_views.unit_ops[
            grid[0], grid[1], grid[2], operation
        ]
        interaction = assigned & ~moving & ~at_shed
        needs_argument = interaction & np.isin(
            operation, tuple(self._ARGUMENT_OPERATIONS)
        )
        safe_item = np.maximum(item, 0)
        argument_legal = batch.mask_views.unit_args[
            grid[0], grid[1], grid[2], operation, safe_item
        ]
        legal &= ~needs_argument | ((item >= 0) & argument_legal)

        unit_actions[..., 0][legal] = operation[legal]
        write_arguments = legal & interaction
        unit_actions[..., 1][write_arguments] = safe_item[write_arguments]
        unit_actions[..., 2][write_arguments] = count[write_arguments]

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
    "WorkforcePlan",
    "WorkforcePlanner",
    "WorkRole",
    "WorkZone",
]
