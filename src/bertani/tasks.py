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


def _active_unit_limit(active_units: NDArray[np.bool_]) -> int:
    """Return the smallest prefix containing every active unit slot.

    The vector environment reserves 231 unit slots under the default
    configuration, while normal rule-policy games use only a small prefix.
    Restricting temporary decoding/execution arrays to that live prefix avoids
    doing the same NumPy work over hundreds of guaranteed-inactive slots.
    """

    active_slots = np.flatnonzero(np.any(active_units, axis=(0, 1)))
    return int(active_slots[-1]) + 1 if active_slots.size else 0


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


@dataclass(slots=True)
class _RouteEvalCache:
    """Precomputed route state for O(1) insertion scoring.

    ``prev_x/y`` and ``elapsed_before`` describe the state immediately before
    each insertion position. ``suffix_new_misses[position, delta]`` stores how
    many currently-on-time tasks at/after ``position`` become late if insertion
    shifts that suffix by ``delta`` turns.
    """

    prev_x: NDArray[np.int16]
    prev_y: NDArray[np.int16]
    elapsed_before: NDArray[np.int16]
    suffix_new_misses: NDArray[np.int16]


class TaskScheduler:
    """Fast route-aware rolling-horizon assignment for farm tasks.

    This is behavior-compatible with the V2 scheduler's routing objective, but
    removes two major hot-path costs:

    1. Task ids are mapped to eligibility columns once per seat instead of
       repeatedly calling ``np.flatnonzero`` inside nested worker/task loops.
    2. A hypothetical insertion recomputes only the changed worker route.
       Metrics for every unchanged route are cached.

    The planning objective remains lexicographic:

        missed deadlines -> makespan -> total route length -> soft preference

    The scheduler still executes only the first task of each planned route and
    replans on the next observation.
    """

    def __init__(
        self,
        board_size: int,
        shed_capacity: int = 100,
        continuity_bonus: float = 1.0,
        episode_steps: int = 720,
        turns_per_day: int = 24,
        scheduler_mode: str = "route",
    ) -> None:
        if scheduler_mode not in {"route", "native"}:
            raise ValueError("scheduler_mode must be 'route' or 'native'")
        if scheduler_mode == "native" and _native_schedule_tasks is None:
            raise RuntimeError(
                "native scheduler requested but bertani._rust.schedule_tasks is unavailable"
            )
        self.scheduler_mode = scheduler_mode
        self.board_size = board_size
        self.shed_capacity = shed_capacity
        self.continuity_bonus = continuity_bonus
        self.last_step = max(1, episode_steps - 1)
        self.turns_per_day = turns_per_day
        self._shape: tuple[int, int, int] | None = None
        self._assignments: TaskAssignments | None = None
        self._previous_task: NDArray[np.int64] | None = None

        # Persistent per-seat route plans. Routes contain stable TaskBatch slot
        # ids, not primitive movement actions. Tile slots remain stable as a
        # tile transitions FEED -> CARE -> HARVEST, HARVEST -> PLANT, etc.
        self._route_cache: dict[tuple[int, int], dict[int, list[int]]] = {}
        self._route_cache_day: dict[tuple[int, int], int] = {}
        self._route_cache_units: dict[tuple[int, int], tuple[int, ...]] = {}
        self._force_replan: set[tuple[int, int]] = set()

        # Lightweight profiling counters.
        self.full_solves = 0
        self.cache_hits = 0

        # Reusable compact buffers for the opt-in native greedy scheduler.
        # Their unit axis is only the live prefix, never VecEnv's padded 231.
        self._native_shape: tuple[int, int, int] | None = None
        self._native_active: NDArray[np.bool_] | None = None
        self._native_role: NDArray[np.int16] | None = None
        self._native_zone: NDArray[np.int16] | None = None
        self._native_previous: NDArray[np.int64] | None = None
        self._native_task_index: NDArray[np.int64] | None = None
        self._native_scores: NDArray[np.float32] | None = None

    def assign(
        self,
        batch: Batch,
        tasks: TaskBatch,
        workforce: WorkforcePlan | None = None,
        seat_mask: NDArray[np.bool_] | None = None,
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
            self._route_cache.clear()
            self._route_cache_day.clear()
            self._route_cache_units.clear()
            self._force_replan.clear()

        assignments = self._assignments
        assert self._previous_task is not None
        assignments.task_index.fill(-1)
        assignments.score.fill(-np.inf)

        views = batch.observation_views
        step = np.rint(
            views.global_features[..., 0] * self.last_step
        ).astype(np.int64)
        hour = step % self.turns_per_day
        day = step // self.turns_per_day

        new_day = hour == 0
        self._previous_task[new_day] = -1
        self._previous_task[~batch.active_units] = -1

        active_limit = _active_unit_limit(batch.active_units)
        units = views.units[:, :, 0, :active_limit]
        scale = max(1, self.board_size - 1)
        unit_x = np.rint(units[..., 2] * scale).astype(np.int16)
        unit_y = np.rint(units[..., 3] * scale).astype(np.int16)
        inventories = np.rint(
            units[..., 5:17] * self.shed_capacity
        ).astype(np.int64)

        if seat_mask is None:
            controlled_seats = np.ones((n, players), dtype=np.bool_)
        else:
            if seat_mask.shape != (n, players):
                raise ValueError(
                    f"seat mask must have shape {(n, players)}"
                )
            controlled_seats = np.asarray(seat_mask, dtype=np.bool_)

        if workforce is None:
            unit_role = np.full(shape, WorkRole.ANY, dtype=np.int16)
            unit_zone = np.full(shape, WorkZone.ANY, dtype=np.int16)
            role_bonus = 0.0
            zone_bonus = 0.0
            reserved_by_kind = np.zeros(
                (*shape[:2], max(TaskKind) + 1),
                dtype=np.int16,
            )
        else:
            if workforce.role.shape != shape or workforce.zone.shape != shape:
                raise ValueError("workforce plan must match active unit shape")

            unit_role = np.ascontiguousarray(workforce.role, dtype=np.int16)
            unit_zone = np.ascontiguousarray(workforce.zone, dtype=np.int16)
            role_bonus = float(workforce.role_bonus)
            zone_bonus = float(workforce.zone_bonus)

            if workforce.reserved_by_kind is None:
                reserved_by_kind = np.zeros(
                    (*shape[:2], max(TaskKind) + 1),
                    dtype=np.int16,
                )
            else:
                expected = (*shape[:2], max(TaskKind) + 1)
                if workforce.reserved_by_kind.shape != expected:
                    raise ValueError(
                        f"capacity reservation must have shape {expected}"
                    )
                reserved_by_kind = np.ascontiguousarray(
                    workforce.reserved_by_kind,
                    dtype=np.int16,
                )

        half = self.board_size // 2
        task_zone = (
            (tasks.target_y >= half).astype(np.int16) * 2
            + (tasks.target_x >= half).astype(np.int16)
        )

        if self.scheduler_mode == "native":
            self._assign_native_greedy(
                batch=batch,
                tasks=tasks,
                assignments=assignments,
                active_limit=active_limit,
                unit_x=unit_x,
                unit_y=unit_y,
                inventories=inventories,
                unit_role=unit_role,
                unit_zone=unit_zone,
                task_zone=task_zone,
                role_bonus=role_bonus,
                zone_bonus=zone_bonus,
                reserved_by_kind=reserved_by_kind,
                controlled_seats=controlled_seats,
            )
            self._previous_task[...] = assignments.task_index
            self._previous_task[~batch.active_units] = -1
            return assignments

        for environment, player in np.argwhere(controlled_seats):
            self._assign_seat(
                environment=int(environment),
                player=int(player),
                batch=batch,
                tasks=tasks,
                assignments=assignments,
                unit_x=unit_x,
                unit_y=unit_y,
                inventories=inventories,
                unit_role=unit_role,
                unit_zone=unit_zone,
                task_zone=task_zone,
                role_bonus=role_bonus,
                zone_bonus=zone_bonus,
                reserved_by_kind=reserved_by_kind,
                hour=int(hour[environment, player]),
                day=int(day[environment, player]),
            )

        self._previous_task[...] = assignments.task_index
        self._previous_task[~batch.active_units] = -1
        return assignments

    def _assign_native_greedy(
        self,
        *,
        batch: Batch,
        tasks: TaskBatch,
        assignments: TaskAssignments,
        active_limit: int,
        unit_x: NDArray[np.int16],
        unit_y: NDArray[np.int16],
        inventories: NDArray[np.int64],
        unit_role: NDArray[np.int16],
        unit_zone: NDArray[np.int16],
        task_zone: NDArray[np.int16],
        role_bonus: float,
        zone_bonus: float,
        reserved_by_kind: NDArray[np.int16],
        controlled_seats: NDArray[np.bool_],
    ) -> None:
        """Run the existing Rust priority-tiered greedy scheduler.

        This is an opt-in experiment for rollout throughput.  It is *not*
        behavior-equivalent to the route-aware Python scheduler: it chooses one
        task per worker greedily inside urgency bands rather than constructing
        complete routes.  The pit harness exposes it behind ``--scheduler
        native`` so speed and strength can be measured independently.

        Only the live unit prefix is copied into contiguous arrays before
        crossing the PyO3 boundary.  Uncontrolled seats are marked inactive so
        the native scheduler does no useful work for outputs the pit will throw
        away.
        """

        assert _native_schedule_tasks is not None
        if active_limit <= 0:
            return
        assert self._previous_task is not None

        # Slicing the last axis of a [N, P, max_units] tensor leaves the parent
        # row stride in place, so explicitly compact the few live slots.  With
        # the current strategy this is normally ~5-14 units instead of 231.
        native_shape = (*batch.active_units.shape[:2], active_limit)
        if self._native_shape != native_shape:
            self._native_active = np.empty(native_shape, dtype=np.bool_)
            self._native_role = np.empty(native_shape, dtype=np.int16)
            self._native_zone = np.empty(native_shape, dtype=np.int16)
            self._native_previous = np.empty(native_shape, dtype=np.int64)
            self._native_task_index = np.empty(native_shape, dtype=np.int64)
            self._native_scores = np.empty(native_shape, dtype=np.float32)
            self._native_shape = native_shape

        assert self._native_active is not None
        assert self._native_role is not None
        assert self._native_zone is not None
        assert self._native_previous is not None
        assert self._native_task_index is not None
        assert self._native_scores is not None

        native_active = self._native_active
        native_role = self._native_role
        native_zone = self._native_zone
        native_previous = self._native_previous
        native_task_index = self._native_task_index
        native_scores = self._native_scores

        np.logical_and(
            batch.active_units[..., :active_limit],
            controlled_seats[..., None],
            out=native_active,
        )
        np.copyto(native_role, unit_role[..., :active_limit])
        np.copyto(native_zone, unit_zone[..., :active_limit])
        np.copyto(native_previous, self._previous_task[..., :active_limit])

        _native_schedule_tasks(
            np.ascontiguousarray(unit_x, dtype=np.int16),
            np.ascontiguousarray(unit_y, dtype=np.int16),
            np.ascontiguousarray(inventories, dtype=np.int64),
            np.ascontiguousarray(tasks.priority, dtype=np.float32),
            native_active,
            np.ascontiguousarray(tasks.active, dtype=np.bool_),
            np.ascontiguousarray(tasks.exclusive, dtype=np.bool_),
            np.ascontiguousarray(tasks.target_x, dtype=np.int16),
            np.ascontiguousarray(tasks.target_y, dtype=np.int16),
            np.ascontiguousarray(tasks.required_item, dtype=np.int16),
            np.ascontiguousarray(tasks.required_count, dtype=np.int64),
            np.ascontiguousarray(tasks.kind, dtype=np.int16),
            np.ascontiguousarray(tasks.work_role, dtype=np.int16),
            native_role,
            native_zone,
            np.ascontiguousarray(task_zone, dtype=np.int16),
            float(role_bonus),
            float(zone_bonus),
            np.ascontiguousarray(reserved_by_kind, dtype=np.int16),
            native_previous,
            float(self.continuity_bonus),
            int(self.board_size),
            native_task_index,
            native_scores,
        )

        assignments.task_index[..., :active_limit] = native_task_index
        assignments.score[..., :active_limit] = native_scores

        # Native mode intentionally has no route cache.  Keep the public
        # counters meaningful so the pit output immediately reveals which path
        # ran.
        self.full_solves += int(np.count_nonzero(controlled_seats))
        self._route_cache.clear()
        self._route_cache_day.clear()
        self._route_cache_units.clear()
        self._force_replan.clear()

    def _assign_seat(
        self,
        *,
        environment: int,
        player: int,
        batch: Batch,
        tasks: TaskBatch,
        assignments: TaskAssignments,
        unit_x: NDArray[np.int16],
        unit_y: NDArray[np.int16],
        inventories: NDArray[np.int64],
        unit_role: NDArray[np.int16],
        unit_zone: NDArray[np.int16],
        task_zone: NDArray[np.int16],
        role_bonus: float,
        zone_bonus: float,
        reserved_by_kind: NDArray[np.int16],
        hour: int,
        day: int,
    ) -> None:
        active_units = np.flatnonzero(batch.active_units[environment, player])
        active_tasks = np.flatnonzero(tasks.active[environment, player])

        if active_units.size == 0 or active_tasks.size == 0:
            key = (environment, player)
            self._route_cache.pop(key, None)
            self._route_cache_day.pop(key, None)
            self._route_cache_units.pop(key, None)
            self._force_replan.discard(key)
            return

        if self._serve_cached_routes(
            environment=environment,
            player=player,
            day=day,
            active_units=active_units,
            active_tasks=active_tasks,
            batch=batch,
            tasks=tasks,
            assignments=assignments,
            unit_x=unit_x,
            unit_y=unit_y,
            inventories=inventories,
            unit_role=unit_role,
            unit_zone=unit_zone,
            task_zone=task_zone,
            role_bonus=role_bonus,
            zone_bonus=zone_bonus,
        ):
            self.cache_hits += 1
            return

        self.full_solves += 1
        worker_count = int(active_units.size)
        starts_x = unit_x[environment, player, active_units].astype(
            np.int16, copy=False
        )
        starts_y = unit_y[environment, player, active_units].astype(
            np.int16, copy=False
        )

        # O(1) task-id -> active-task-column lookup. The old implementation did
        # np.flatnonzero(active_tasks == task) in several nested loops.
        task_column = np.full(tasks.capacity, -1, dtype=np.int16)
        task_column[active_tasks] = np.arange(
            active_tasks.size,
            dtype=np.int16,
        )

        routes: list[list[int]] = [[] for _ in range(worker_count)]
        prefix_length = np.zeros(worker_count, dtype=np.int16)
        # Rebuilt only for the worker whose route changed. The old scheduler
        # rescanned the complete route for every (task, worker, position)
        # candidate, which dominated full-solve CPU time.
        route_eval_cache: list[_RouteEvalCache | None] = [None] * worker_count

        eligible = self._eligibility_matrix_fast(
            environment,
            player,
            active_units,
            active_tasks,
            tasks,
            inventories,
        )

        exclusive_mask = tasks.exclusive[
            environment,
            player,
            active_tasks,
        ]
        exclusive_tasks = active_tasks[exclusive_mask]
        nonexclusive_tasks = active_tasks[~exclusive_mask]

        # Cached metrics for the current constructed routes. These are updated
        # only after accepting an insertion.
        route_missed = np.zeros(worker_count, dtype=np.int16)
        route_length = np.zeros(worker_count, dtype=np.int16)
        route_preference = np.zeros(worker_count, dtype=np.float32)
        total_missed = 0
        total_length = 0
        total_preference = 0.0

        if exclusive_tasks.size:
            priorities = tasks.priority[
                environment,
                player,
                exclusive_tasks,
            ]
            urgency = np.floor(priorities / 10.0)
            urgency_bands = np.unique(urgency)[::-1]

            for urgency_band in urgency_bands:
                tier_mask = urgency == urgency_band
                tier_tasks = exclusive_tasks[tier_mask]
                if tier_tasks.size == 0:
                    continue

                lower_tasks = exclusive_tasks[urgency < urgency_band]
                future_reserve = self._future_reserve(
                    environment,
                    player,
                    float(urgency_band),
                    lower_tasks,
                    tasks,
                    reserved_by_kind,
                )
                max_workers = max(
                    1,
                    worker_count - future_reserve,
                )

                candidate_locals = self._candidate_workers_for_tier_fast(
                    environment=environment,
                    player=player,
                    tier_tasks=tier_tasks,
                    starts_x=starts_x,
                    starts_y=starts_y,
                    routes=routes,
                    eligible=eligible,
                    task_column=task_column,
                    tasks=tasks,
                    max_workers=max_workers,
                )
                if not candidate_locals:
                    continue

                ordered_tasks = sorted(
                    (int(task) for task in tier_tasks),
                    key=lambda task: (
                        self._deadline_key(
                            int(tasks.deadline[environment, player, task]),
                            hour,
                        ),
                        -float(tasks.priority[environment, player, task]),
                        task,
                    ),
                )

                for task in ordered_tasks:
                    # Current top-two route lengths let us obtain the makespan
                    # after changing one route in O(1).
                    max1, max2, max1_count = self._top_two_lengths(route_length)

                    best: tuple[
                        tuple[int, int, int, float],
                        int,
                        int,
                        tuple[int, int, float],
                    ] | None = None

                    task_col = int(task_column[task])

                    for local in candidate_locals:
                        if not eligible[local, task_col]:
                            continue

                        worker = int(active_units[local])
                        first_position = int(prefix_length[local])
                        route = routes[local]

                        old_missed = int(route_missed[local])
                        old_length = int(route_length[local])
                        old_preference = float(route_preference[local])

                        if old_length == max1 and max1_count == 1:
                            other_max = max2
                        else:
                            other_max = max1

                        for position in range(
                            first_position,
                            len(route) + 1,
                        ):
                            route_cache = route_eval_cache[local]
                            if route_cache is None:
                                route_cache = self._build_route_eval_cache(
                                    environment=environment,
                                    player=player,
                                    start_x=int(starts_x[local]),
                                    start_y=int(starts_y[local]),
                                    route=route,
                                    tasks=tasks,
                                    hour=hour,
                                )
                                route_eval_cache[local] = route_cache

                            candidate_metrics = self._route_metrics_with_insertion_fast(
                                environment=environment,
                                player=player,
                                worker=worker,
                                route=route,
                                route_cache=route_cache,
                                old_missed=old_missed,
                                old_length=old_length,
                                old_preference=old_preference,
                                insert_task=task,
                                insert_position=position,
                                tasks=tasks,
                                unit_role=unit_role,
                                unit_zone=unit_zone,
                                task_zone=task_zone,
                                role_bonus=role_bonus,
                                zone_bonus=zone_bonus,
                                hour=hour,
                            )

                            cand_missed, cand_length, cand_preference = (
                                candidate_metrics
                            )

                            objective = (
                                total_missed - old_missed + cand_missed,
                                max(other_max, cand_length),
                                total_length - old_length + cand_length,
                                -(
                                    total_preference
                                    - old_preference
                                    + cand_preference
                                ),
                            )

                            candidate = (
                                objective,
                                local,
                                position,
                                candidate_metrics,
                            )
                            if best is None or candidate[:3] < best[:3]:
                                best = candidate

                    if best is None:
                        continue

                    _, local, position, new_metrics = best
                    old_missed = int(route_missed[local])
                    old_length = int(route_length[local])
                    old_preference = float(route_preference[local])

                    routes[local].insert(position, task)
                    route_eval_cache[local] = None

                    new_missed, new_length, new_preference = new_metrics
                    route_missed[local] = new_missed
                    route_length[local] = new_length
                    route_preference[local] = new_preference

                    total_missed += new_missed - old_missed
                    total_length += new_length - old_length
                    total_preference += new_preference - old_preference

                # Lower urgency bands may not jump ahead of tasks already placed
                # by higher urgency bands.
                for local in range(worker_count):
                    prefix_length[local] = len(routes[local])

        claimed: set[int] = set()

        for local, route in enumerate(routes):
            if not route:
                continue

            worker = int(active_units[local])
            task = int(route[0])
            assignments.task_index[environment, player, worker] = task
            assignments.score[
                environment,
                player,
                worker,
            ] = self._assignment_score(
                environment,
                player,
                worker,
                task,
                unit_x,
                unit_y,
                tasks,
                unit_role,
                unit_zone,
                task_zone,
                role_bonus,
                zone_bonus,
            )
            claimed.add(task)

        # Non-exclusive logistics tasks keep the V2 semantics, but avoid
        # allocating a temporary choices list.
        if nonexclusive_tasks.size:
            for local, worker_value in enumerate(active_units):
                worker = int(worker_value)
                if assignments.task_index[environment, player, worker] >= 0:
                    continue

                best_choice: tuple[float, int] | None = None
                for task_value in nonexclusive_tasks:
                    task = int(task_value)
                    col = int(task_column[task])
                    if not eligible[local, col]:
                        continue

                    score = self._assignment_score(
                        environment,
                        player,
                        worker,
                        task,
                        unit_x,
                        unit_y,
                        tasks,
                        unit_role,
                        unit_zone,
                        task_zone,
                        role_bonus,
                        zone_bonus,
                    )
                    choice = (score, task)
                    if best_choice is None or choice > best_choice:
                        best_choice = choice

                if best_choice is not None:
                    score, task = best_choice
                    assignments.task_index[
                        environment,
                        player,
                        worker,
                    ] = task
                    assignments.score[
                        environment,
                        player,
                        worker,
                    ] = score

        self._prefer_local_task(
            environment,
            player,
            active_units,
            claimed,
            assignments,
            tasks,
            unit_x,
            unit_y,
            inventories,
        )

        key = (environment, player)
        self._route_cache[key] = {
            int(active_units[local]): list(route)
            for local, route in enumerate(routes)
        }
        self._route_cache_day[key] = day
        self._route_cache_units[key] = tuple(
            int(worker) for worker in active_units
        )

        # FETCH_ITEM is intentionally a one-turn logistics task in V2. Once a
        # pickup executes, worker eligibility changes, so solve again next turn.
        assigned_indices = assignments.task_index[
            environment,
            player,
            active_units,
        ]
        assigned_mask = assigned_indices >= 0
        if np.any(assigned_mask):
            assigned_kinds = tasks.kind[
                environment,
                player,
                assigned_indices[assigned_mask],
            ]
            if np.any(assigned_kinds == int(TaskKind.FETCH_ITEM)):
                self._force_replan.add(key)

        # The underfoot override may intentionally deviate from the route's
        # first planned task. Replan next turn instead of mutating the cached
        # route in a potentially inconsistent way.
        for local, worker_value in enumerate(active_units):
            worker = int(worker_value)
            route = routes[local]
            assigned_task = int(
                assignments.task_index[
                    environment,
                    player,
                    worker,
                ]
            )
            planned_task = route[0] if route else -1
            if assigned_task >= 0 and assigned_task != planned_task:
                self._force_replan.add(key)
                break

    def _serve_cached_routes(
        self,
        *,
        environment: int,
        player: int,
        day: int,
        active_units: NDArray[np.int64],
        active_tasks: NDArray[np.int64],
        batch: Batch,
        tasks: TaskBatch,
        assignments: TaskAssignments,
        unit_x: NDArray[np.int16],
        unit_y: NDArray[np.int16],
        inventories: NDArray[np.int64],
        unit_role: NDArray[np.int16],
        unit_zone: NDArray[np.int16],
        task_zone: NDArray[np.int16],
        role_bonus: float,
        zone_bonus: float,
    ) -> bool:
        """Serve a still-valid day plan without running route construction.

        A full replan is required only when:
        - the day changed,
        - the active worker set changed (hire/end-of-day),
        - a previous FETCH_ITEM changed eligibility,
        - a genuinely new exclusive task appears outside cached routes, or
        - a cached route's next task is no longer executable by its worker.

        Completed/inactive tasks are simply removed from cached routes.
        """

        key = (environment, player)
        routes = self._route_cache.get(key)
        if routes is None:
            return False
        if self._route_cache_day.get(key) != day:
            return False

        active_unit_tuple = tuple(int(worker) for worker in active_units)
        if self._route_cache_units.get(key) != active_unit_tuple:
            return False

        if key in self._force_replan:
            self._force_replan.discard(key)
            return False

        active_mask = tasks.active[environment, player]

        # Drop jobs that another worker already completed or that disappeared.
        # If the same tile changes from FEED to CARE/HARVEST, its slot remains
        # active and therefore stays at the same place in the cached route.
        planned: set[int] = set()
        for worker in active_unit_tuple:
            route = routes.get(worker)
            if route is None:
                return False
            if route:
                route[:] = [
                    task for task in route
                    if bool(active_mask[task])
                ]
                planned.update(route)

        # A new exclusive task that was not visible during the last solve
        # indicates a material state change: land/seed changes, a newly created
        # production job, etc. Recompute to place it globally.
        exclusive_active = active_tasks[
            tasks.exclusive[
                environment,
                player,
                active_tasks,
            ]
        ]
        for task_value in exclusive_active:
            if int(task_value) not in planned:
                return False

        claimed: set[int] = set()

        for worker_value in active_units:
            worker = int(worker_value)
            route = routes[worker]
            if not route:
                continue

            task = int(route[0])
            required = int(
                tasks.required_item[
                    environment,
                    player,
                    task,
                ]
            )
            if (
                required >= 0
                and inventories[
                    environment,
                    player,
                    worker,
                    required,
                ]
                < tasks.required_count[
                    environment,
                    player,
                    task,
                ]
            ):
                return False

            # Deposit tasks require the worker to still be carrying something.
            if (
                int(
                    tasks.kind[
                        environment,
                        player,
                        task,
                    ]
                )
                == int(TaskKind.DEPOSIT_INVENTORY)
                and inventories[
                    environment,
                    player,
                    worker,
                ].sum()
                <= 0
            ):
                return False

            assignments.task_index[
                environment,
                player,
                worker,
            ] = task
            assignments.score[
                environment,
                player,
                worker,
            ] = self._assignment_score(
                environment,
                player,
                worker,
                task,
                unit_x,
                unit_y,
                tasks,
                unit_role,
                unit_zone,
                task_zone,
                role_bonus,
                zone_bonus,
            )
            claimed.add(task)

        # Preserve V2's cheap non-exclusive logistics behavior for otherwise
        # idle workers. These are intentionally not persisted.
        nonexclusive_tasks = active_tasks[
            ~tasks.exclusive[
                environment,
                player,
                active_tasks,
            ]
        ]
        if nonexclusive_tasks.size:
            for worker_value in active_units:
                worker = int(worker_value)
                if assignments.task_index[
                    environment,
                    player,
                    worker,
                ] >= 0:
                    continue

                best_choice: tuple[float, int] | None = None
                for task_value in nonexclusive_tasks:
                    task = int(task_value)

                    required = int(
                        tasks.required_item[
                            environment,
                            player,
                            task,
                        ]
                    )
                    if (
                        required >= 0
                        and inventories[
                            environment,
                            player,
                            worker,
                            required,
                        ]
                        < tasks.required_count[
                            environment,
                            player,
                            task,
                        ]
                    ):
                        continue

                    if (
                        int(
                            tasks.kind[
                                environment,
                                player,
                                task,
                            ]
                        )
                        == int(TaskKind.DEPOSIT_INVENTORY)
                        and inventories[
                            environment,
                            player,
                            worker,
                        ].sum()
                        <= 0
                    ):
                        continue

                    score = self._assignment_score(
                        environment,
                        player,
                        worker,
                        task,
                        unit_x,
                        unit_y,
                        tasks,
                        unit_role,
                        unit_zone,
                        task_zone,
                        role_bonus,
                        zone_bonus,
                    )
                    choice = (score, task)
                    if best_choice is None or choice > best_choice:
                        best_choice = choice

                if best_choice is not None:
                    score, task = best_choice
                    assignments.task_index[
                        environment,
                        player,
                        worker,
                    ] = task
                    assignments.score[
                        environment,
                        player,
                        worker,
                    ] = score
                    if (
                        int(
                            tasks.kind[
                                environment,
                                player,
                                task,
                            ]
                        )
                        == int(TaskKind.FETCH_ITEM)
                    ):
                        self._force_replan.add(key)

        before_local = assignments.task_index[
            environment,
            player,
            active_units,
        ].copy()

        self._prefer_local_task(
            environment,
            player,
            active_units,
            claimed,
            assignments,
            tasks,
            unit_x,
            unit_y,
            inventories,
        )

        after_local = assignments.task_index[
            environment,
            player,
            active_units,
        ]
        if np.any(before_local != after_local):
            self._force_replan.add(key)

        return True

    @staticmethod
    def _top_two_lengths(
        lengths: NDArray[np.int16],
    ) -> tuple[int, int, int]:
        """Return largest, second-largest, and multiplicity of largest."""

        max1 = 0
        max2 = 0
        count = 0

        for value_raw in lengths:
            value = int(value_raw)
            if value > max1:
                max2 = max1
                max1 = value
                count = 1
            elif value == max1:
                count += 1
            elif value > max2:
                max2 = value

        return max1, max2, count

    def _eligibility_matrix_fast(
        self,
        environment: int,
        player: int,
        active_units: NDArray[np.int64],
        active_tasks: NDArray[np.int64],
        tasks: TaskBatch,
        inventories: NDArray[np.int64],
    ) -> NDArray[np.bool_]:
        """Vectorized V2 eligibility calculation."""

        required_item = tasks.required_item[
            environment,
            player,
            active_tasks,
        ].astype(np.int64, copy=False)

        required_count = tasks.required_count[
            environment,
            player,
            active_tasks,
        ].astype(np.int64, copy=False)

        safe_item = np.maximum(required_item, 0)

        carried = inventories[
            environment,
            player,
            active_units[:, None],
            safe_item[None, :],
        ]

        eligible = (
            (required_item[None, :] < 0)
            | (carried >= required_count[None, :])
        )

        deposit_columns = (
            tasks.kind[
                environment,
                player,
                active_tasks,
            ]
            == int(TaskKind.DEPOSIT_INVENTORY)
        )

        if np.any(deposit_columns):
            carrying = (
                inventories[
                    environment,
                    player,
                    active_units,
                ].sum(axis=-1)
                > 0
            )
            eligible[:, deposit_columns] &= carrying[:, None]

        return np.ascontiguousarray(eligible, dtype=np.bool_)

    @staticmethod
    def _deadline_key(deadline: int, hour: int) -> int:
        if deadline < 0:
            return 1_000_000
        return max(0, deadline - hour)

    @staticmethod
    def _future_reserve(
        environment: int,
        player: int,
        urgency_band: float,
        lower_tasks: NDArray[np.int64],
        tasks: TaskBatch,
        reserved_by_kind: NDArray[np.int16],
    ) -> int:
        if urgency_band >= 12 or lower_tasks.size == 0:
            return 0

        future_reserve = 0
        lower_kinds = tasks.kind[
            environment,
            player,
            lower_tasks,
        ]

        for kind in range(reserved_by_kind.shape[-1]):
            requested = int(
                reserved_by_kind[
                    environment,
                    player,
                    kind,
                ]
            )
            if requested <= 0:
                continue

            available = int(
                np.count_nonzero(lower_kinds == kind)
            )
            future_reserve += min(requested, available)

        return future_reserve

    def _candidate_workers_for_tier_fast(
        self,
        *,
        environment: int,
        player: int,
        tier_tasks: NDArray[np.int64],
        starts_x: NDArray[np.int16],
        starts_y: NDArray[np.int16],
        routes: list[list[int]],
        eligible: NDArray[np.bool_],
        task_column: NDArray[np.int16],
        tasks: TaskBatch,
        max_workers: int,
    ) -> list[int]:
        """Equivalent V2 worker ranking without repeated task-column searches."""

        ranking: list[tuple[int, int, int]] = []

        tier_x = tasks.target_x[
            environment,
            player,
            tier_tasks,
        ]
        tier_y = tasks.target_y[
            environment,
            player,
            tier_tasks,
        ]

        for local, route in enumerate(routes):
            if route:
                endpoint_task = route[-1]
                start_x = int(
                    tasks.target_x[
                        environment,
                        player,
                        endpoint_task,
                    ]
                )
                start_y = int(
                    tasks.target_y[
                        environment,
                        player,
                        endpoint_task,
                    ]
                )
            else:
                start_x = int(starts_x[local])
                start_y = int(starts_y[local])

            columns = task_column[tier_tasks].astype(np.int64, copy=False)
            allowed = eligible[local, columns]
            if not np.any(allowed):
                continue

            distances = (
                np.abs(tier_x[allowed] - start_x)
                + np.abs(tier_y[allowed] - start_y)
            )
            best_distance = int(distances.min())
            ranking.append(
                (
                    best_distance,
                    len(route),
                    local,
                )
            )

        ranking.sort()
        return [
            local
            for _, _, local in ranking[:max_workers]
        ]

    def _build_route_eval_cache(
        self,
        *,
        environment: int,
        player: int,
        start_x: int,
        start_y: int,
        route: list[int],
        tasks: TaskBatch,
        hour: int,
    ) -> _RouteEvalCache:
        """Build route statistics reused by every hypothetical insertion.

        A 10x10 board bounds the extra distance introduced by inserting one
        task between two route points to 37 turns.  We keep a tiny suffix table
        over that bounded delta so deadline effects are an O(1) lookup.
        """

        route_len = len(route)
        prev_x = np.empty(route_len + 1, dtype=np.int16)
        prev_y = np.empty(route_len + 1, dtype=np.int16)
        elapsed_before = np.empty(route_len + 1, dtype=np.int16)
        completion = np.empty(route_len, dtype=np.int16)

        x = start_x
        y = start_y
        elapsed = 0
        for index, task in enumerate(route):
            prev_x[index] = x
            prev_y[index] = y
            elapsed_before[index] = elapsed

            tx = int(tasks.target_x[environment, player, task])
            ty = int(tasks.target_y[environment, player, task])
            elapsed += abs(x - tx) + abs(y - ty) + 1
            completion[index] = elapsed
            x = tx
            y = ty

        prev_x[route_len] = x
        prev_y[route_len] = y
        elapsed_before[route_len] = elapsed

        # d(prev,new)+d(new,next)-d(prev,next)+1.  Manhattan distance on a
        # BxB board is <= 2(B-1), so 4(B-1)+1 is an exact upper bound.
        max_delta = 4 * (self.board_size - 1) + 1
        suffix_hist = np.zeros(
            (route_len + 1, max_delta),
            dtype=np.int16,
        )

        for index in range(route_len - 1, -1, -1):
            suffix_hist[index] = suffix_hist[index + 1]
            task = route[index]
            deadline = int(tasks.deadline[environment, player, task])
            if deadline < 0:
                continue

            completion_hour = hour + int(completion[index]) - 1
            slack = deadline - completion_hour
            # Negative slack means this task is already late and therefore
            # remains counted in old_missed.  Very large slack cannot be
            # crossed by a single insertion on this board.
            if 0 <= slack < max_delta:
                suffix_hist[index, slack] += 1

        suffix_new_misses = np.zeros(
            (route_len + 1, max_delta + 1),
            dtype=np.int16,
        )
        if max_delta:
            # Column d is the number of suffix tasks with slack < d.
            suffix_new_misses[:, 1:] = np.cumsum(
                suffix_hist,
                axis=1,
                dtype=np.int16,
            )

        return _RouteEvalCache(
            prev_x=prev_x,
            prev_y=prev_y,
            elapsed_before=elapsed_before,
            suffix_new_misses=suffix_new_misses,
        )

    def _route_metrics_with_insertion_fast(
        self,
        *,
        environment: int,
        player: int,
        worker: int,
        route: list[int],
        route_cache: _RouteEvalCache,
        old_missed: int,
        old_length: int,
        old_preference: float,
        insert_task: int,
        insert_position: int,
        tasks: TaskBatch,
        unit_role: NDArray[np.int16],
        unit_zone: NDArray[np.int16],
        task_zone: NDArray[np.int16],
        role_bonus: float,
        zone_bonus: float,
        hour: int,
    ) -> tuple[int, int, float]:
        """Exact V2 insertion objective using cached route statistics."""

        px = int(route_cache.prev_x[insert_position])
        py = int(route_cache.prev_y[insert_position])
        elapsed_before = int(route_cache.elapsed_before[insert_position])

        tx = int(tasks.target_x[environment, player, insert_task])
        ty = int(tasks.target_y[environment, player, insert_task])
        to_insert = abs(px - tx) + abs(py - ty)

        if insert_position < len(route):
            next_task = route[insert_position]
            nx = int(tasks.target_x[environment, player, next_task])
            ny = int(tasks.target_y[environment, player, next_task])
            delta = (
                to_insert
                + abs(tx - nx)
                + abs(ty - ny)
                - abs(px - nx)
                - abs(py - ny)
                + 1
            )
        else:
            delta = to_insert + 1

        new_length = old_length + delta
        inserted_completion = elapsed_before + to_insert + 1

        remaining_turns = self.turns_per_day - hour
        old_overflow = max(0, old_length - remaining_turns)
        deadline_misses = old_missed - old_overflow

        deadline = int(tasks.deadline[environment, player, insert_task])
        completion_hour = hour + inserted_completion - 1
        if deadline >= 0 and completion_hour > deadline:
            deadline_misses += 1

        deadline_misses += int(
            route_cache.suffix_new_misses[insert_position, delta]
        )
        new_missed = deadline_misses + max(
            0,
            new_length - remaining_turns,
        )

        preference = old_preference
        worker_role = int(unit_role[environment, player, worker])
        task_role = int(tasks.work_role[environment, player, insert_task])
        if worker_role != int(WorkRole.ANY) and worker_role == task_role:
            preference += role_bonus

        worker_zone = int(unit_zone[environment, player, worker])
        if (
            worker_zone != int(WorkZone.ANY)
            and worker_zone
            == int(task_zone[environment, player, insert_task])
        ):
            preference += zone_bonus

        if (
            self._previous_task is not None
            and insert_task
            == int(self._previous_task[environment, player, worker])
        ):
            preference += self.continuity_bonus

        return new_missed, new_length, preference

    def _route_metrics_with_insertion_reference(
        self,
        *,
        environment: int,
        player: int,
        worker: int,
        start_x: int,
        start_y: int,
        route: list[int],
        insert_task: int,
        insert_position: int,
        tasks: TaskBatch,
        unit_role: NDArray[np.int16],
        unit_zone: NDArray[np.int16],
        task_zone: NDArray[np.int16],
        role_bonus: float,
        zone_bonus: float,
        hour: int,
    ) -> tuple[int, int, float]:
        """Evaluate one hypothetical insertion without copying the route."""

        x = start_x
        y = start_y
        elapsed = 0
        missed_deadlines = 0
        preference = 0.0

        previous = (
            -1
            if self._previous_task is None
            else int(
                self._previous_task[
                    environment,
                    player,
                    worker,
                ]
            )
        )

        worker_role = int(
            unit_role[
                environment,
                player,
                worker,
            ]
        )
        worker_zone = int(
            unit_zone[
                environment,
                player,
                worker,
            ]
        )

        route_len = len(route)

        for index in range(route_len + 1):
            if index == insert_position:
                task = insert_task
            else:
                route_index = index if index < insert_position else index - 1
                if route_index >= route_len:
                    continue
                task = route[route_index]

            tx = int(
                tasks.target_x[
                    environment,
                    player,
                    task,
                ]
            )
            ty = int(
                tasks.target_y[
                    environment,
                    player,
                    task,
                ]
            )

            elapsed += abs(x - tx) + abs(y - ty) + 1
            x = tx
            y = ty

            deadline = int(
                tasks.deadline[
                    environment,
                    player,
                    task,
                ]
            )
            completion_hour = hour + elapsed - 1
            if deadline >= 0 and completion_hour > deadline:
                missed_deadlines += 1

            task_role = int(
                tasks.work_role[
                    environment,
                    player,
                    task,
                ]
            )
            if (
                worker_role != int(WorkRole.ANY)
                and worker_role == task_role
            ):
                preference += role_bonus

            if (
                worker_zone != int(WorkZone.ANY)
                and worker_zone
                == int(
                    task_zone[
                        environment,
                        player,
                        task,
                    ]
                )
            ):
                preference += zone_bonus

            if task == previous:
                preference += self.continuity_bonus

        remaining_turns = self.turns_per_day - hour
        missed_deadlines += max(
            0,
            elapsed - remaining_turns,
        )

        return (
            missed_deadlines,
            elapsed,
            preference,
        )

    def _assignment_score(
        self,
        environment: int,
        player: int,
        worker: int,
        task: int,
        unit_x: NDArray[np.int16],
        unit_y: NDArray[np.int16],
        tasks: TaskBatch,
        unit_role: NDArray[np.int16],
        unit_zone: NDArray[np.int16],
        task_zone: NDArray[np.int16],
        role_bonus: float,
        zone_bonus: float,
    ) -> float:
        distance = abs(
            int(unit_x[environment, player, worker])
            - int(tasks.target_x[environment, player, task])
        ) + abs(
            int(unit_y[environment, player, worker])
            - int(tasks.target_y[environment, player, task])
        )

        score = float(
            tasks.priority[
                environment,
                player,
                task,
            ]
        ) - distance

        task_role = int(
            tasks.work_role[
                environment,
                player,
                task,
            ]
        )
        worker_role = int(
            unit_role[
                environment,
                player,
                worker,
            ]
        )
        if (
            worker_role != int(WorkRole.ANY)
            and worker_role == task_role
        ):
            score += role_bonus

        worker_zone = int(
            unit_zone[
                environment,
                player,
                worker,
            ]
        )
        if (
            worker_zone != int(WorkZone.ANY)
            and worker_zone
            == int(
                task_zone[
                    environment,
                    player,
                    task,
                ]
            )
        ):
            score += zone_bonus

        if (
            self._previous_task is not None
            and int(
                self._previous_task[
                    environment,
                    player,
                    worker,
                ]
            )
            == task
        ):
            score += self.continuity_bonus

        return score

    @staticmethod
    def _prefer_local_task(
        environment: int,
        player: int,
        active_units: NDArray[np.int64],
        claimed: set[int],
        assignments: TaskAssignments,
        tasks: TaskBatch,
        unit_x: NDArray[np.int16],
        unit_y: NDArray[np.int16],
        inventories: NDArray[np.int64],
        priority_slack: float = 5.0,
    ) -> None:
        for worker_value in active_units:
            worker = int(worker_value)
            local = (
                int(unit_y[environment, player, worker]) * tasks.board_size
                + int(unit_x[environment, player, worker])
            )

            if (
                local >= tasks.tile_slots
                or not tasks.active[environment, player, local]
                or local in claimed
            ):
                continue

            required = int(
                tasks.required_item[
                    environment,
                    player,
                    local,
                ]
            )

            if (
                required >= 0
                and inventories[
                    environment,
                    player,
                    worker,
                    required,
                ]
                < tasks.required_count[
                    environment,
                    player,
                    local,
                ]
            ):
                continue

            current = int(
                assignments.task_index[
                    environment,
                    player,
                    worker,
                ]
            )

            current_priority = (
                float(
                    tasks.priority[
                        environment,
                        player,
                        current,
                    ]
                )
                if current >= 0
                else -np.inf
            )
            local_priority = float(
                tasks.priority[
                    environment,
                    player,
                    local,
                ]
            )

            if local_priority + priority_slack < current_priority:
                continue

            if current >= 0:
                claimed.discard(current)

            assignments.task_index[
                environment,
                player,
                worker,
            ] = local
            assignments.score[
                environment,
                player,
                worker,
            ] = local_priority * 1_000.0
            claimed.add(local)

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
        active_limit = _active_unit_limit(batch.active_units)
        if active_limit == 0:
            return

        units = batch.observation_views.units[:, :, 0, :active_limit]
        scale = max(1, self.board_size - 1)
        unit_x = np.rint(units[..., 2] * scale).astype(np.int16)
        unit_y = np.rint(units[..., 3] * scale).astype(np.int16)
        task_index = assignments.task_index[..., :active_limit]
        assigned = task_index >= 0
        safe_task = np.maximum(task_index, 0)

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

        active_actions = unit_actions[..., :active_limit, :]
        active_actions[..., 0][legal] = operation[legal]
        write_arguments = legal & interaction
        active_actions[..., 1][write_arguments] = safe_item[write_arguments]
        active_actions[..., 2][write_arguments] = count[write_arguments]

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
