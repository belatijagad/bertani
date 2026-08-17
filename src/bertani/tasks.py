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
    """Route-aware rolling-horizon assignment for farm tasks.

    The scheduler still returns exactly one task per active unit because the
    executor/environment accept one primitive unit action per turn.  Internally,
    however, it constructs a short multi-task route for every worker and assigns
    the first task on that route.  Replanning on every observation gives us a
    model-predictive-control style loop without persisting brittle imperative
    scripts.

    Important design constraints:

    * Priority *bands* remain hard ordering boundaries.  A lower-band task is
      never inserted ahead of a higher-band task already planned for a worker.
    * Required-item checks are conservative.  A FEED/FERTILIZE/PLACE task is
      routeable only by a worker that already carries the required item.  The
      existing FETCH_ITEM task rules therefore remain the explicit prerequisite;
      after a pickup succeeds, the next turn's replan can route the dependent
      task to that worker.
    * Exclusive tasks appear in at most one route.  Non-exclusive logistics jobs
      (currently chiefly deposit) are assigned afterwards to otherwise-idle
      eligible workers.
    * The objective is lexicographic: missed deadlines, longest route, total
      route length, then negative soft preference.  This strongly favors a plan
      that fits inside the day before shaving movement from an already-late plan.
    """

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

        views = batch.observation_views
        step = np.rint(
            views.global_features[..., 0] * self.last_step
        ).astype(np.int64)
        hour = step % self.turns_per_day
        new_day = hour == 0
        self._previous_task[new_day] = -1
        self._previous_task[~batch.active_units] = -1

        units = views.units[:, :, 0]
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
            reserved_by_kind = np.zeros(
                (*shape[:2], max(TaskKind) + 1), dtype=np.int16
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

        for environment in range(n):
            for player in range(players):
                self._assign_seat(
                    environment=environment,
                    player=player,
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
                )

        self._previous_task[...] = assignments.task_index
        self._previous_task[~batch.active_units] = -1
        return assignments

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
    ) -> None:
        active_units = np.flatnonzero(batch.active_units[environment, player])
        active_tasks = np.flatnonzero(tasks.active[environment, player])
        if active_units.size == 0 or active_tasks.size == 0:
            return

        starts = [
            (
                int(unit_x[environment, player, unit]),
                int(unit_y[environment, player, unit]),
            )
            for unit in active_units
        ]
        worker_to_local = {
            int(worker): local for local, worker in enumerate(active_units)
        }
        routes: list[list[int]] = [[] for _ in active_units]
        prefix_length = np.zeros(active_units.size, dtype=np.int16)

        eligible = self._eligibility_matrix(
            environment,
            player,
            active_units,
            active_tasks,
            batch,
            tasks,
            inventories,
        )

        exclusive_tasks = active_tasks[
            tasks.exclusive[environment, player, active_tasks]
        ]
        nonexclusive_tasks = active_tasks[
            ~tasks.exclusive[environment, player, active_tasks]
        ]

        if exclusive_tasks.size:
            priorities = tasks.priority[environment, player, exclusive_tasks]
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
                max_workers = max(1, active_units.size - future_reserve)
                candidate_locals = self._candidate_workers_for_tier(
                    environment,
                    player,
                    tier_tasks,
                    active_units,
                    starts,
                    routes,
                    eligible,
                    active_tasks,
                    tasks,
                    max_workers,
                )
                if not candidate_locals:
                    continue

                # Higher priority/deadline-sensitive jobs are inserted first.
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
                    best: tuple[tuple[int, int, int, float], int, int] | None = None
                    for local in candidate_locals:
                        worker = int(active_units[local])
                        if not self._is_eligible_task(
                            task,
                            worker,
                            active_tasks,
                            eligible,
                            worker_to_local,
                        ):
                            continue

                        # Lower urgency bands may not jump ahead of the prefix
                        # already planned by higher bands for this worker.
                        first_position = int(prefix_length[local])
                        for position in range(first_position, len(routes[local]) + 1):
                            trial_route = list(routes[local])
                            trial_route.insert(position, task)
                            objective = self._global_objective_with_trial(
                                environment,
                                player,
                                routes,
                                local,
                                trial_route,
                                starts,
                                tasks,
                                active_units,
                                unit_role,
                                unit_zone,
                                task_zone,
                                role_bonus,
                                zone_bonus,
                                hour,
                            )
                            candidate = (objective, local, position)
                            if best is None or candidate < best:
                                best = candidate

                    if best is None:
                        continue
                    _, local, position = best
                    routes[local].insert(position, task)

                # Everything placed through this band becomes an immutable
                # prefix for all lower urgency bands.
                for local in range(len(routes)):
                    prefix_length[local] = len(routes[local])

        # Assign the first task of every constructed route.  Replanning next
        # turn advances the worker naturally when that task disappears/changes.
        claimed: set[int] = set()
        for local, route in enumerate(routes):
            if not route:
                continue
            worker = int(active_units[local])
            task = int(route[0])
            assignments.task_index[environment, player, worker] = task
            assignments.score[environment, player, worker] = self._assignment_score(
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

        # Non-exclusive jobs are per-worker logistics objectives.  They are
        # allowed to be selected by multiple units, but only use workers that
        # did not receive a higher-priority exclusive route head.
        if nonexclusive_tasks.size:
            for worker in active_units:
                worker = int(worker)
                if assignments.task_index[environment, player, worker] >= 0:
                    continue
                local = worker_to_local[worker]
                choices: list[tuple[float, int]] = []
                for task in nonexclusive_tasks:
                    task = int(task)
                    if not eligible[local, self._task_column(task, active_tasks)]:
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
                    choices.append((score, task))
                if choices:
                    score, task = max(choices)
                    assignments.task_index[environment, player, worker] = task
                    assignments.score[environment, player, worker] = score

        # A useful action underfoot should beat needless travel when it is in
        # roughly the same priority range and not already owned.  This keeps
        # harvest->plant workflows responsive without restoring hard ownership.
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

    def _eligibility_matrix(
        self,
        environment: int,
        player: int,
        active_units: NDArray[np.int64],
        active_tasks: NDArray[np.int64],
        batch: Batch,
        tasks: TaskBatch,
        inventories: NDArray[np.int64],
    ) -> NDArray[np.bool_]:
        eligible = np.ones((active_units.size, active_tasks.size), dtype=np.bool_)

        for task_column, task in enumerate(active_tasks):
            task = int(task)
            required_item = int(tasks.required_item[environment, player, task])
            required_count = int(tasks.required_count[environment, player, task])
            if required_item >= 0:
                eligible[:, task_column] &= (
                    inventories[
                        environment,
                        player,
                        active_units,
                        required_item,
                    ]
                    >= required_count
                )

            if int(tasks.kind[environment, player, task]) == int(
                TaskKind.DEPOSIT_INVENTORY
            ):
                carrying = inventories[
                    environment, player, active_units
                ].sum(axis=-1) > 0
                eligible[:, task_column] &= carrying

        return eligible

    @staticmethod
    def _task_column(task: int, active_tasks: NDArray[np.int64]) -> int:
        matches = np.flatnonzero(active_tasks == task)
        if matches.size == 0:
            raise KeyError(f"task {task} is not active")
        return int(matches[0])

    def _is_eligible_task(
        self,
        task: int,
        worker: int,
        active_tasks: NDArray[np.int64],
        eligible: NDArray[np.bool_],
        worker_to_local: dict[int, int],
    ) -> bool:
        return bool(
            eligible[
                worker_to_local[worker],
                self._task_column(task, active_tasks),
            ]
        )

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
        for kind in range(reserved_by_kind.shape[-1]):
            requested = int(reserved_by_kind[environment, player, kind])
            if requested <= 0:
                continue
            available = int(
                np.count_nonzero(
                    tasks.kind[environment, player, lower_tasks] == kind
                )
            )
            future_reserve += min(requested, available)
        return future_reserve

    def _candidate_workers_for_tier(
        self,
        environment: int,
        player: int,
        tier_tasks: NDArray[np.int64],
        active_units: NDArray[np.int64],
        starts: list[tuple[int, int]],
        routes: list[list[int]],
        eligible: NDArray[np.bool_],
        active_tasks: NDArray[np.int64],
        tasks: TaskBatch,
        max_workers: int,
    ) -> list[int]:
        """Pick a compact worker subset when capacity is reserved downstream."""

        ranking: list[tuple[int, int, int]] = []
        for local, worker in enumerate(active_units):
            best_distance = 1_000_000
            for task in tier_tasks:
                task = int(task)
                column = self._task_column(task, active_tasks)
                if not eligible[local, column]:
                    continue
                start = self._route_endpoint(
                    environment,
                    player,
                    starts[local],
                    routes[local],
                    tasks,
                )
                distance = abs(start[0] - int(tasks.target_x[environment, player, task]))
                distance += abs(start[1] - int(tasks.target_y[environment, player, task]))
                best_distance = min(best_distance, distance)
            if best_distance < 1_000_000:
                ranking.append((best_distance, len(routes[local]), local))

        ranking.sort()
        return [local for _, _, local in ranking[:max_workers]]

    @staticmethod
    def _route_endpoint(
        environment: int,
        player: int,
        start: tuple[int, int],
        route: list[int],
        tasks: TaskBatch,
    ) -> tuple[int, int]:
        if not route:
            return start
        task = route[-1]
        return (
            int(tasks.target_x[environment, player, task]),
            int(tasks.target_y[environment, player, task]),
        )

    def _global_objective_with_trial(
        self,
        environment: int,
        player: int,
        routes: list[list[int]],
        changed_local: int,
        trial_route: list[int],
        starts: list[tuple[int, int]],
        tasks: TaskBatch,
        active_units: NDArray[np.int64],
        unit_role: NDArray[np.int16],
        unit_zone: NDArray[np.int16],
        task_zone: NDArray[np.int16],
        role_bonus: float,
        zone_bonus: float,
        hour: int,
    ) -> tuple[int, int, int, float]:
        missed = 0
        makespan = 0
        total = 0
        preference = 0.0

        for local, route in enumerate(routes):
            candidate_route = trial_route if local == changed_local else route
            worker = int(active_units[local])
            route_missed, route_length, route_preference = self._route_metrics(
                environment,
                player,
                worker,
                starts[local],
                candidate_route,
                tasks,
                unit_role,
                unit_zone,
                task_zone,
                role_bonus,
                zone_bonus,
                hour,
            )
            missed += route_missed
            makespan = max(makespan, route_length)
            total += route_length
            preference += route_preference

        return missed, makespan, total, -preference

    def _route_metrics(
        self,
        environment: int,
        player: int,
        worker: int,
        start: tuple[int, int],
        route: list[int],
        tasks: TaskBatch,
        unit_role: NDArray[np.int16],
        unit_zone: NDArray[np.int16],
        task_zone: NDArray[np.int16],
        role_bonus: float,
        zone_bonus: float,
        hour: int,
    ) -> tuple[int, int, float]:
        x, y = start
        elapsed = 0
        missed_deadlines = 0
        preference = 0.0
        previous = -1 if self._previous_task is None else int(
            self._previous_task[environment, player, worker]
        )

        for task in route:
            tx = int(tasks.target_x[environment, player, task])
            ty = int(tasks.target_y[environment, player, task])
            elapsed += abs(x - tx) + abs(y - ty) + 1
            x, y = tx, ty

            deadline = int(tasks.deadline[environment, player, task])
            # elapsed=1 means the task is completed on the current hour.
            completion_hour = hour + elapsed - 1
            if deadline >= 0 and completion_hour > deadline:
                missed_deadlines += 1

            task_role = int(tasks.work_role[environment, player, task])
            worker_role = int(unit_role[environment, player, worker])
            if worker_role != int(WorkRole.ANY) and worker_role == task_role:
                preference += role_bonus

            worker_zone = int(unit_zone[environment, player, worker])
            if worker_zone != int(WorkZone.ANY) and worker_zone == int(
                task_zone[environment, player, task]
            ):
                preference += zone_bonus

            if task == previous:
                preference += self.continuity_bonus

        # A worker disappears at the day boundary, so a route longer than the
        # remaining action budget is intrinsically infeasible even when its
        # individual tasks have no explicit deadline. Count overflow actions
        # as deadline misses so feasibility dominates distance optimization.
        remaining_turns = self.turns_per_day - hour
        missed_deadlines += max(0, elapsed - remaining_turns)
        return missed_deadlines, elapsed, preference

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
        score = float(tasks.priority[environment, player, task]) - distance

        task_role = int(tasks.work_role[environment, player, task])
        worker_role = int(unit_role[environment, player, worker])
        if worker_role != int(WorkRole.ANY) and worker_role == task_role:
            score += role_bonus

        worker_zone = int(unit_zone[environment, player, worker])
        if worker_zone != int(WorkZone.ANY) and worker_zone == int(
            task_zone[environment, player, task]
        ):
            score += zone_bonus

        if self._previous_task is not None and int(
            self._previous_task[environment, player, worker]
        ) == task:
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

            required = int(tasks.required_item[environment, player, local])
            if required >= 0 and inventories[
                environment, player, worker, required
            ] < tasks.required_count[environment, player, local]:
                continue

            current = int(assignments.task_index[environment, player, worker])
            current_priority = (
                float(tasks.priority[environment, player, current])
                if current >= 0
                else -np.inf
            )
            local_priority = float(tasks.priority[environment, player, local])
            if local_priority + priority_slack < current_priority:
                continue

            if current >= 0:
                claimed.discard(current)
            assignments.task_index[environment, player, worker] = local
            assignments.score[environment, player, worker] = local_priority * 1_000.0
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
