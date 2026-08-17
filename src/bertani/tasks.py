"""Task-level action abstractions for extensible rule-based policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

try:
    from ._rust import schedule_routes as _schedule_routes
except (ImportError, ModuleNotFoundError):
    _schedule_routes = None

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

class TaskScheduler:
    """Route-aware rolling-horizon task assignment backed by Rust.

    Python owns the route cache, non-exclusive logistics, underfoot overrides,
    assignment scores, and replan lifecycle. The expensive exclusive-task route
    construction has a single implementation in ``bertani._rust.schedule_routes``.

    The route objective is lexicographic:

        missed deadlines -> makespan -> total route length -> soft preference
    """

    def __init__(
        self,
        board_size: int,
        shed_capacity: int = 100,
        continuity_bonus: float = 1.0,
        episode_steps: int = 720,
        turns_per_day: int = 24,
    ) -> None:
        if _schedule_routes is None:
            raise RuntimeError(
                "bertani._rust.schedule_routes is unavailable; rebuild the "
                "extension with `uv run maturin develop --release`"
            )
        self.board_size = board_size
        self.shed_capacity = shed_capacity
        self.continuity_bonus = continuity_bonus
        self.last_step = max(1, episode_steps - 1)
        self.turns_per_day = turns_per_day
        self._shape: tuple[int, int, int] | None = None
        self._assignments: TaskAssignments | None = None
        self._previous_task: NDArray[np.int64] | None = None

        # Routes contain stable TaskBatch slot ids rather than primitive
        # movement actions. Stable tile slots let FEED -> CARE -> HARVEST and
        # HARVEST -> PLANT workflows reuse a day plan when it remains valid.
        self._route_cache: dict[tuple[int, int], dict[int, list[int]]] = {}
        self._route_cache_day: dict[tuple[int, int], int] = {}
        self._route_cache_units: dict[tuple[int, int], tuple[int, ...]] = {}
        self._force_replan: set[tuple[int, int]] = set()

        # Lightweight diagnostics used by the native pit harness.
        self.full_solves = 0
        self.cache_hits = 0
        self.cache_miss_reasons: dict[str, int] = {
            "no_route": 0,
            "day_changed": 0,
            "unit_set_changed": 0,
            "forced": 0,
            "missing_worker_route": 0,
            "new_exclusive": 0,
            "required_item": 0,
            "empty_deposit": 0,
        }
        self.force_replan_reasons: dict[str, int] = {
            "fetch_item": 0,
            "local_override": 0,
        }

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
                raise ValueError(f"seat mask must have shape {(n, players)}")
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

    def _solve_routes(
        self,
        *,
        environment: int,
        player: int,
        active_units: NDArray[np.int64],
        starts_x: NDArray[np.int16],
        starts_y: NDArray[np.int16],
        inventories: NDArray[np.int64],
        tasks: TaskBatch,
        unit_role: NDArray[np.int16],
        unit_zone: NDArray[np.int16],
        task_zone: NDArray[np.int16],
        role_bonus: float,
        zone_bonus: float,
        reserved_by_kind: NDArray[np.int16],
        hour: int,
    ) -> list[list[int]]:
        """Construct the exclusive-task route plan in the native solver."""

        assert _schedule_routes is not None
        assert self._previous_task is not None

        seat_inventory = np.ascontiguousarray(
            inventories[environment, player, active_units],
            dtype=np.int64,
        )
        seat_roles = np.ascontiguousarray(
            unit_role[environment, player, active_units],
            dtype=np.int16,
        )
        seat_zones = np.ascontiguousarray(
            unit_zone[environment, player, active_units],
            dtype=np.int16,
        )
        previous = np.ascontiguousarray(
            self._previous_task[environment, player, active_units],
            dtype=np.int64,
        )

        raw_routes = _schedule_routes(
            np.ascontiguousarray(starts_x, dtype=np.int16),
            np.ascontiguousarray(starts_y, dtype=np.int16),
            seat_inventory,
            np.ascontiguousarray(tasks.active[environment, player], dtype=np.bool_),
            np.ascontiguousarray(tasks.exclusive[environment, player], dtype=np.bool_),
            np.ascontiguousarray(tasks.priority[environment, player], dtype=np.float32),
            np.ascontiguousarray(tasks.target_x[environment, player], dtype=np.int16),
            np.ascontiguousarray(tasks.target_y[environment, player], dtype=np.int16),
            np.ascontiguousarray(tasks.deadline[environment, player], dtype=np.int16),
            np.ascontiguousarray(tasks.required_item[environment, player], dtype=np.int16),
            np.ascontiguousarray(tasks.required_count[environment, player], dtype=np.int64),
            np.ascontiguousarray(tasks.kind[environment, player], dtype=np.int16),
            np.ascontiguousarray(tasks.work_role[environment, player], dtype=np.int16),
            seat_roles,
            seat_zones,
            np.ascontiguousarray(task_zone[environment, player], dtype=np.int16),
            np.ascontiguousarray(reserved_by_kind[environment, player], dtype=np.int16),
            previous,
            float(role_bonus),
            float(zone_bonus),
            float(self.continuity_bonus),
            int(self.board_size),
            int(hour),
            int(self.turns_per_day),
        )

        routes = [[int(task) for task in route] for route in raw_routes]
        if len(routes) != active_units.size:
            raise RuntimeError(
                "native route solver returned the wrong number of worker routes"
            )
        return routes

    def _count_cache_miss(self, reason: str) -> None:
        self.cache_miss_reasons[reason] = self.cache_miss_reasons.get(reason, 0) + 1

    def _count_force_replan(self, reason: str) -> None:
        self.force_replan_reasons[reason] = self.force_replan_reasons.get(reason, 0) + 1

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
        starts_x = unit_x[environment, player, active_units].astype(
            np.int16, copy=False
        )
        starts_y = unit_y[environment, player, active_units].astype(
            np.int16, copy=False
        )

        # O(1) task-id -> active-task-column lookup for non-exclusive logistics.
        task_column = np.full(tasks.capacity, -1, dtype=np.int16)
        task_column[active_tasks] = np.arange(
            active_tasks.size,
            dtype=np.int16,
        )

        eligible = self._eligibility_matrix(
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

        routes: list[list[int]] = [[] for _ in range(active_units.size)]
        if exclusive_tasks.size:
            routes = self._solve_routes(
                environment=environment,
                player=player,
                active_units=active_units,
                starts_x=starts_x,
                starts_y=starts_y,
                inventories=inventories,
                tasks=tasks,
                unit_role=unit_role,
                unit_zone=unit_zone,
                task_zone=task_zone,
                role_bonus=role_bonus,
                zone_bonus=zone_bonus,
                reserved_by_kind=reserved_by_kind,
                hour=hour,
            )

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

        # Non-exclusive logistics tasks keep the existing semantics.
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

        # FETCH_ITEM is a one-turn logistics task. Once a pickup executes,
        # worker eligibility changes, so solve again next turn.
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
                self._count_force_replan("fetch_item")

        # An underfoot override may intentionally deviate from the route's
        # first planned task. Replan next turn rather than mutating the route.
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
                self._count_force_replan("local_override")
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
                self._count_cache_miss("no_route")
                return False
            if self._route_cache_day.get(key) != day:
                self._count_cache_miss("day_changed")
                return False

            active_unit_tuple = tuple(int(worker) for worker in active_units)
            if self._route_cache_units.get(key) != active_unit_tuple:
                self._count_cache_miss("unit_set_changed")
                return False

            if key in self._force_replan:
                self._force_replan.discard(key)
                self._count_cache_miss("forced")
                return False

            active_mask = tasks.active[environment, player]

            # Drop jobs that another worker already completed or that disappeared.
            # If the same tile changes from FEED to CARE/HARVEST, its slot remains
            # active and therefore stays at the same place in the cached route.
            planned: set[int] = set()
            for worker in active_unit_tuple:
                route = routes.get(worker)
                if route is None:
                    self._count_cache_miss("missing_worker_route")
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
                    self._count_cache_miss("new_exclusive")
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
                    self._count_cache_miss("required_item")
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
                    self._count_cache_miss("empty_deposit")
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
                            self._count_force_replan("fetch_item")

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
                self._count_force_replan("local_override")

            return True

    def _eligibility_matrix(
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
