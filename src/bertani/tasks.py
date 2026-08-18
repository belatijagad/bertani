"""Task-level action abstractions for extensible rule-based policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

try:
    from ._rust import execute_assignments as _execute_assignments
    from ._rust import propose_farm_tasks as _propose_farm_tasks
    from ._rust import propose_maintenance_tasks as _propose_maintenance_tasks
    from ._rust import propose_production_tasks as _propose_production_tasks
    from ._rust import NativeTaskScheduler as _NativeTaskScheduler
except (ImportError, ModuleNotFoundError):
    _execute_assignments = None
    _propose_farm_tasks = None
    _propose_maintenance_tasks = None
    _propose_production_tasks = None
    _NativeTaskScheduler = None

from .vec_env import Batch, Item, UnitOp

if TYPE_CHECKING:
    from .rule_based import StrategicIntent


def _active_unit_limit(active_units: NDArray[np.bool_]) -> int:
    """Return the smallest prefix containing every active unit slot.

    The vector environment reserves 231 unit slots under the default
    configuration, while normal rule-policy games use only a small prefix.
    Restricting temporary decoding/execution arrays to that live prefix avoids
    doing the same NumPy work over hundreds of guaranteed-inactive slots.
    """

    active_slots = np.flatnonzero(np.any(active_units, axis=(0, 1)))
    return int(active_slots[-1]) + 1 if active_slots.size else 0


def _normalized_seat_mask(
    batch: Batch, seat_mask: NDArray[np.bool_] | None
) -> NDArray[np.bool_]:
    shape = batch.active_units.shape[:2]
    if seat_mask is None:
        return np.ones(shape, dtype=np.bool_)
    if seat_mask.shape != shape:
        raise ValueError(f"seat mask must have shape {shape}")
    if seat_mask.dtype == np.bool_ and seat_mask.flags.c_contiguous:
        return seat_mask
    return np.ascontiguousarray(seat_mask, dtype=np.bool_)


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


def propose_native_farm_tasks(
    batch: Batch,
    intent: StrategicIntent,
    tasks: TaskBatch,
    *,
    seat_mask: NDArray[np.bool_] | None = None,
    turns_per_day: int,
    shed_capacity: int,
    episode_steps: int,
) -> None:
    """Populate maintenance and production tasks through one native call."""

    if _propose_farm_tasks is None:
        raise RuntimeError(
            "native farm tasks require the bertani._rust extension"
        )
    views = batch.observation_views
    _propose_farm_tasks(
        views.tiles,
        views.global_features,
        views.units,
        views.private,
        batch.active_units,
        _normalized_seat_mask(batch, seat_mask),
        np.ascontiguousarray(intent.target_crop_counts, dtype=np.int64),
        np.ascontiguousarray(intent.target_animal_counts, dtype=np.int64),
        np.ascontiguousarray(intent.liquidate, dtype=np.bool_),
        tasks.active,
        tasks.kind,
        tasks.target_x,
        tasks.target_y,
        tasks.item,
        tasks.quantity,
        tasks.priority,
        tasks.deadline,
        tasks.estimated_value,
        tasks.required_item,
        tasks.required_count,
        tasks.exclusive,
        tasks.work_role,
        tasks.board_size,
        tasks.tile_slots,
        turns_per_day,
        shed_capacity,
        episode_steps,
    )


def propose_native_maintenance_tasks(
    batch: Batch,
    tasks: TaskBatch,
    *,
    seat_mask: NDArray[np.bool_] | None = None,
    turns_per_day: int,
    shed_capacity: int,
    episode_steps: int,
) -> None:
    """Populate deterministic maintenance tasks through the native kernel.

    Production/portfolio decisions deliberately remain in Python. The native
    function only mirrors the mechanical maintenance rule and writes directly
    into the reusable :class:`TaskBatch` buffers.
    """

    if _propose_maintenance_tasks is None:
        raise RuntimeError(
            "native maintenance tasks require the bertani._rust extension"
        )
    views = batch.observation_views
    _propose_maintenance_tasks(
        views.tiles,
        views.global_features,
        views.units,
        views.private,
        batch.active_units,
        _normalized_seat_mask(batch, seat_mask),
        tasks.active,
        tasks.kind,
        tasks.target_x,
        tasks.target_y,
        tasks.item,
        tasks.quantity,
        tasks.priority,
        tasks.deadline,
        tasks.estimated_value,
        tasks.required_item,
        tasks.required_count,
        tasks.exclusive,
        tasks.work_role,
        tasks.board_size,
        tasks.tile_slots,
        turns_per_day,
        shed_capacity,
        episode_steps,
    )



def propose_native_production_tasks(
    batch: Batch,
    intent: StrategicIntent,
    tasks: TaskBatch,
    *,
    seat_mask: NDArray[np.bool_] | None = None,
    turns_per_day: int,
    shed_capacity: int,
    episode_steps: int,
) -> None:
    """Populate deterministic production tasks through the native kernel.

    Strategic targets remain in Python; Rust only maps those targets to exact
    tile/logistics tasks using the current farm state.
    """
    if _propose_production_tasks is None:
        raise RuntimeError(
            "native production tasks require the bertani._rust extension"
        )
    views = batch.observation_views
    _propose_production_tasks(
        views.tiles,
        views.global_features,
        views.units,
        views.private,
        batch.active_units,
        _normalized_seat_mask(batch, seat_mask),
        np.ascontiguousarray(intent.target_crop_counts, dtype=np.int64),
        np.ascontiguousarray(intent.target_animal_counts, dtype=np.int64),
        np.ascontiguousarray(intent.liquidate, dtype=np.bool_),
        tasks.active,
        tasks.kind,
        tasks.target_x,
        tasks.target_y,
        tasks.item,
        tasks.quantity,
        tasks.priority,
        tasks.deadline,
        tasks.estimated_value,
        tasks.required_item,
        tasks.required_count,
        tasks.exclusive,
        tasks.work_role,
        tasks.board_size,
        tasks.tile_slots,
        turns_per_day,
        shed_capacity,
        episode_steps,
    )

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
    """Persistent Rust-backed rolling-horizon task scheduler.

    The native controller owns route-cache validation, cached-route serving,
    non-exclusive logistics, local overrides, assignment scoring, replan
    invalidation, and route construction. Python retains only the reusable
    NumPy output buffers and the public diagnostics interface.
    """

    _CACHE_MISS_NAMES = (
        "no_route",
        "day_changed",
        "unit_set_changed",
        "forced",
        "missing_worker_route",
        "new_exclusive",
        "required_item",
        "empty_deposit",
    )
    _FORCE_REPLAN_NAMES = (
        "fetch_item",
        "local_override",
    )

    def __init__(
        self,
        board_size: int,
        shed_capacity: int = 100,
        continuity_bonus: float = 1.0,
        episode_steps: int = 720,
        turns_per_day: int = 24,
    ) -> None:
        if _NativeTaskScheduler is None:
            raise RuntimeError(
                "bertani._rust.NativeTaskScheduler is unavailable; rebuild the "
                "extension with `uv run maturin develop --release`"
            )
        self.board_size = board_size
        self.shed_capacity = shed_capacity
        self.continuity_bonus = continuity_bonus
        self.last_step = max(1, episode_steps - 1)
        self.turns_per_day = turns_per_day
        self._native = _NativeTaskScheduler(
            board_size,
            shed_capacity,
            continuity_bonus,
            episode_steps,
            turns_per_day,
        )
        self._shape: tuple[int, int, int] | None = None
        self._assignments: TaskAssignments | None = None
        self._all_seats: NDArray[np.bool_] | None = None
        self._default_role: NDArray[np.int16] | None = None
        self._default_zone: NDArray[np.int16] | None = None
        self._default_reserved: NDArray[np.int16] | None = None

    @property
    def full_solves(self) -> int:
        return int(self._native.full_solves)

    @property
    def cache_hits(self) -> int:
        return int(self._native.cache_hits)

    @property
    def idle_worker_steals(self) -> int:
        return int(self._native.idle_worker_steals)

    @property
    def cache_miss_reasons(self) -> dict[str, int]:
        counts = self._native.cache_miss_counts()
        return {
            name: int(count)
            for name, count in zip(self._CACHE_MISS_NAMES, counts, strict=True)
        }

    @property
    def force_replan_reasons(self) -> dict[str, int]:
        counts = self._native.force_replan_counts()
        return {
            name: int(count)
            for name, count in zip(self._FORCE_REPLAN_NAMES, counts, strict=True)
        }

    def _prepare_inputs(
        self,
        batch: Batch,
        workforce: WorkforcePlan | None,
        seat_mask: NDArray[np.bool_] | None,
    ):
        n, players, unit_count = batch.active_units.shape
        shape = (n, players, unit_count)
        if self._assignments is None or self._shape != shape:
            self._assignments = TaskAssignments(
                task_index=np.full(shape, -1, dtype=np.int64),
                score=np.full(shape, -np.inf, dtype=np.float32),
            )
            self._all_seats = np.ones((n, players), dtype=np.bool_)
            self._default_role = np.full(shape, WorkRole.ANY, dtype=np.int16)
            self._default_zone = np.full(shape, WorkZone.ANY, dtype=np.int16)
            self._default_reserved = np.zeros(
                (n, players, max(TaskKind) + 1),
                dtype=np.int16,
            )
            self._shape = shape

        assert self._assignments is not None
        assert self._all_seats is not None
        assert self._default_role is not None
        assert self._default_zone is not None
        assert self._default_reserved is not None

        if seat_mask is None:
            controlled_seats = self._all_seats
        else:
            if seat_mask.shape != (n, players):
                raise ValueError(f"seat mask must have shape {(n, players)}")
            if seat_mask.dtype != np.bool_ or not seat_mask.flags.c_contiguous:
                controlled_seats = np.ascontiguousarray(seat_mask, dtype=np.bool_)
            else:
                controlled_seats = seat_mask

        if workforce is None:
            unit_role = self._default_role
            unit_zone = self._default_zone
            reserved_by_kind = self._default_reserved
            role_bonus = 0.0
            zone_bonus = 0.0
        else:
            if workforce.role.shape != shape or workforce.zone.shape != shape:
                raise ValueError("workforce plan must match active unit shape")
            unit_role = np.ascontiguousarray(workforce.role, dtype=np.int16)
            unit_zone = np.ascontiguousarray(workforce.zone, dtype=np.int16)
            role_bonus = float(workforce.role_bonus)
            zone_bonus = float(workforce.zone_bonus)
            if workforce.reserved_by_kind is None:
                reserved_by_kind = self._default_reserved
            else:
                expected = (n, players, max(TaskKind) + 1)
                if workforce.reserved_by_kind.shape != expected:
                    raise ValueError(
                        f"capacity reservation must have shape {expected}"
                    )
                reserved_by_kind = np.ascontiguousarray(
                    workforce.reserved_by_kind,
                    dtype=np.int16,
                )

        return (
            controlled_seats,
            unit_role,
            unit_zone,
            reserved_by_kind,
            role_bonus,
            zone_bonus,
        )

    def assign(
        self,
        batch: Batch,
        tasks: TaskBatch,
        workforce: WorkforcePlan | None = None,
        seat_mask: NDArray[np.bool_] | None = None,
    ) -> TaskAssignments:
        (
            controlled_seats,
            unit_role,
            unit_zone,
            reserved_by_kind,
            role_bonus,
            zone_bonus,
        ) = self._prepare_inputs(batch, workforce, seat_mask)
        assert self._assignments is not None

        views = batch.observation_views
        self._native.assign(
            views.global_features,
            views.units,
            batch.active_units,
            tasks.active,
            tasks.kind,
            tasks.target_x,
            tasks.target_y,
            tasks.priority,
            tasks.deadline,
            tasks.required_item,
            tasks.required_count,
            tasks.exclusive,
            tasks.work_role,
            unit_role,
            unit_zone,
            reserved_by_kind,
            controlled_seats,
            self._assignments.task_index,
            self._assignments.score,
            role_bonus,
            zone_bonus,
        )
        return self._assignments

    def assign_and_execute(
        self,
        batch: Batch,
        tasks: TaskBatch,
        unit_actions: NDArray[np.int64],
        workforce: WorkforcePlan | None = None,
        seat_mask: NDArray[np.bool_] | None = None,
    ) -> TaskAssignments:
        """Assign tasks and emit legal unit actions in one native call."""

        (
            controlled_seats,
            unit_role,
            unit_zone,
            reserved_by_kind,
            role_bonus,
            zone_bonus,
        ) = self._prepare_inputs(batch, workforce, seat_mask)
        assert self._assignments is not None

        views = batch.observation_views
        masks = batch.mask_views
        self._native.assign_and_execute(
            views.global_features,
            views.units,
            masks.unit_ops,
            masks.unit_args,
            batch.active_units,
            tasks.active,
            tasks.kind,
            tasks.target_x,
            tasks.target_y,
            tasks.item,
            tasks.quantity,
            tasks.priority,
            tasks.deadline,
            tasks.required_item,
            tasks.required_count,
            tasks.exclusive,
            tasks.work_role,
            unit_role,
            unit_zone,
            reserved_by_kind,
            controlled_seats,
            self._assignments.task_index,
            self._assignments.score,
            unit_actions,
            role_bonus,
            zone_bonus,
        )
        return self._assignments


class TaskExecutor:
    """Convert assignments into legal primitive actions through Rust."""

    def __init__(self, board_size: int) -> None:
        if _execute_assignments is None:
            raise RuntimeError(
                "native task execution requires the bertani._rust extension"
            )
        self.board_size = board_size

    def execute(
        self,
        batch: Batch,
        tasks: TaskBatch,
        assignments: TaskAssignments,
        unit_actions: NDArray[np.int64],
    ) -> None:
        views = batch.observation_views
        masks = batch.mask_views
        _execute_assignments(
            views.units,
            masks.unit_ops,
            masks.unit_args,
            batch.active_units,
            tasks.kind,
            tasks.target_x,
            tasks.target_y,
            tasks.item,
            tasks.quantity,
            assignments.task_index,
            unit_actions,
            self.board_size,
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
    "propose_native_farm_tasks",
    "WorkforcePlan",
    "WorkforcePlanner",
    "WorkRole",
    "WorkZone",
]
