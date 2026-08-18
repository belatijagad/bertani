//! Persistent native controller for the rule policy's rolling-horizon scheduler.
//!
//! The route objective itself lives in `route_scheduler`. This module ports the
//! Python orchestration around it: route-cache validation, cached-route serving,
//! non-exclusive logistics, underfoot overrides, assignment scores, and replan
//! invalidation. The semantics intentionally mirror the validated Python V10
//! controller; only the implementation boundary changes.

#![allow(clippy::all, clippy::pedantic)]

use numpy::{
    ndarray::ArrayView5, PyArray2, PyArray3, PyArray4, PyArray5, PyArrayMethods,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::executor::execute_assignments;
use crate::route_scheduler::solve_routes_core;

const INVENTORY_ITEMS: usize = 12;
const TASK_KIND_COUNT: usize = 14;
const TASK_FETCH_ITEM: i16 = 12;
const TASK_DEPOSIT_INVENTORY: i16 = 13;
const ROLE_ANY: i16 = 0;
const ZONE_ANY: i16 = -1;
const LOCAL_PRIORITY_SLACK: f32 = 5.0;

const MISS_NO_ROUTE: usize = 0;
const MISS_DAY_CHANGED: usize = 1;
const MISS_UNIT_SET_CHANGED: usize = 2;
const MISS_FORCED: usize = 3;
const MISS_MISSING_WORKER_ROUTE: usize = 4;
const MISS_NEW_EXCLUSIVE: usize = 5;
const MISS_REQUIRED_ITEM: usize = 6;
const MISS_EMPTY_DEPOSIT: usize = 7;
const MISS_COUNT: usize = 8;

const FORCE_FETCH_ITEM: usize = 0;
const FORCE_LOCAL_OVERRIDE: usize = 1;
const FORCE_COUNT: usize = 2;

#[derive(Clone, Copy, Debug)]
struct StealCandidate {
    makespan: i32,
    total_delta: i32,
    preference_loss: f64,
    idle_distance: i32,
    idle_local: usize,
    donor_local: usize,
    donor_position: usize,
    task: usize,
}

#[inline]
fn steal_candidate_is_better(candidate: StealCandidate, best: StealCandidate) -> bool {
    candidate
        .makespan
        .cmp(&best.makespan)
        .then_with(|| candidate.total_delta.cmp(&best.total_delta))
        .then_with(|| {
            candidate
                .preference_loss
                .partial_cmp(&best.preference_loss)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| candidate.idle_distance.cmp(&best.idle_distance))
        .then_with(|| candidate.idle_local.cmp(&best.idle_local))
        .then_with(|| candidate.donor_local.cmp(&best.donor_local))
        .then_with(|| candidate.donor_position.cmp(&best.donor_position))
        .then_with(|| candidate.task.cmp(&best.task))
        == std::cmp::Ordering::Less
}

#[derive(Clone, Debug)]
struct SeatCache {
    day: i32,
    units: Vec<usize>,
    routes: Vec<Vec<usize>>,
}

#[pyclass]
pub(crate) struct NativeTaskScheduler {
    board_size: usize,
    shed_capacity: i64,
    continuity_bonus: f64,
    last_step: i64,
    turns_per_day: i64,
    shape: Option<(usize, usize, usize)>,
    previous_task: Vec<i64>,
    caches: Vec<Option<SeatCache>>,
    force_replan: Vec<bool>,
    full_solves: u64,
    cache_hits: u64,
    idle_worker_steals: u64,
    cache_miss_counts: [u64; MISS_COUNT],
    force_replan_counts: [u64; FORCE_COUNT],
}

#[pymethods]
impl NativeTaskScheduler {
    #[new]
    fn new(
        board_size: usize,
        shed_capacity: i64,
        continuity_bonus: f64,
        episode_steps: i64,
        turns_per_day: i64,
    ) -> PyResult<Self> {
        if board_size == 0 || shed_capacity <= 0 || episode_steps <= 0 || turns_per_day <= 0 {
            return Err(PyValueError::new_err("scheduler configuration must be positive"));
        }
        Ok(Self {
            board_size,
            shed_capacity,
            continuity_bonus,
            last_step: (episode_steps - 1).max(1),
            turns_per_day,
            shape: None,
            previous_task: Vec::new(),
            caches: Vec::new(),
            force_replan: Vec::new(),
            full_solves: 0,
            cache_hits: 0,
            idle_worker_steals: 0,
            cache_miss_counts: [0; MISS_COUNT],
            force_replan_counts: [0; FORCE_COUNT],
        })
    }

    #[getter]
    fn full_solves(&self) -> u64 {
        self.full_solves
    }

    #[getter]
    fn cache_hits(&self) -> u64 {
        self.cache_hits
    }

    #[getter]
    fn idle_worker_steals(&self) -> u64 {
        self.idle_worker_steals
    }

    fn cache_miss_counts(&self) -> Vec<u64> {
        self.cache_miss_counts.to_vec()
    }

    fn force_replan_counts(&self) -> Vec<u64> {
        self.force_replan_counts.to_vec()
    }

    #[allow(clippy::too_many_arguments)]
    fn assign<'py>(
        &mut self,
        global_features: Bound<'py, PyArray3<f32>>,
        units: Bound<'py, PyArray5<f32>>,
        active_units: Bound<'py, PyArray3<bool>>,
        task_active: Bound<'py, PyArray3<bool>>,
        task_kind: Bound<'py, PyArray3<i16>>,
        task_target_x: Bound<'py, PyArray3<i16>>,
        task_target_y: Bound<'py, PyArray3<i16>>,
        task_priority: Bound<'py, PyArray3<f32>>,
        task_deadline: Bound<'py, PyArray3<i16>>,
        task_required_item: Bound<'py, PyArray3<i16>>,
        task_required_count: Bound<'py, PyArray3<i64>>,
        task_exclusive: Bound<'py, PyArray3<bool>>,
        task_work_role: Bound<'py, PyArray3<i16>>,
        unit_role: Bound<'py, PyArray3<i16>>,
        unit_zone: Bound<'py, PyArray3<i16>>,
        reserved_by_kind: Bound<'py, PyArray3<i16>>,
        seat_mask: Bound<'py, PyArray2<bool>>,
        out_task_index: Bound<'py, PyArray3<i64>>,
        out_score: Bound<'py, PyArray3<f32>>,
        role_bonus: f64,
        zone_bonus: f64,
    ) -> PyResult<()> {
        for (name, contiguous) in [
            ("active_units", active_units.is_c_contiguous()),
            ("task_active", task_active.is_c_contiguous()),
            ("task_kind", task_kind.is_c_contiguous()),
            ("task_target_x", task_target_x.is_c_contiguous()),
            ("task_target_y", task_target_y.is_c_contiguous()),
            ("task_priority", task_priority.is_c_contiguous()),
            ("task_deadline", task_deadline.is_c_contiguous()),
            ("task_required_item", task_required_item.is_c_contiguous()),
            ("task_required_count", task_required_count.is_c_contiguous()),
            ("task_exclusive", task_exclusive.is_c_contiguous()),
            ("task_work_role", task_work_role.is_c_contiguous()),
            ("unit_role", unit_role.is_c_contiguous()),
            ("unit_zone", unit_zone.is_c_contiguous()),
            ("reserved_by_kind", reserved_by_kind.is_c_contiguous()),
            ("seat_mask", seat_mask.is_c_contiguous()),
            ("out_task_index", out_task_index.is_c_contiguous()),
            ("out_score", out_score.is_c_contiguous()),
        ] {
            if !contiguous {
                return Err(PyValueError::new_err(format!("{name} must be C-contiguous")));
            }
        }

        let global_shape = global_features.shape();
        if global_shape.len() != 3 || global_shape[2] < 1 {
            return Err(PyValueError::new_err("global_features has incompatible shape"));
        }
        let num_envs = global_shape[0];
        let players = global_shape[1];

        let active_shape = active_units.shape();
        if active_shape.len() != 3 || active_shape[0] != num_envs || active_shape[1] != players {
            return Err(PyValueError::new_err("active_units has incompatible shape"));
        }
        let max_units = active_shape[2];

        let unit_shape = units.shape();
        if unit_shape.len() != 5
            || unit_shape[0] != num_envs
            || unit_shape[1] != players
            || unit_shape[2] < 1
            || unit_shape[3] != max_units
            || unit_shape[4] < 5 + INVENTORY_ITEMS
        {
            return Err(PyValueError::new_err("units has incompatible shape"));
        }

        let task_shape = task_active.shape();
        if task_shape.len() != 3 || task_shape[0] != num_envs || task_shape[1] != players {
            return Err(PyValueError::new_err("task arrays have incompatible shape"));
        }
        let task_count = task_shape[2];
        if self.board_size.saturating_mul(self.board_size) > task_count {
            return Err(PyValueError::new_err("task capacity is smaller than board tile slots"));
        }
        for (name, shape) in [
            ("task_kind", task_kind.shape()),
            ("task_target_x", task_target_x.shape()),
            ("task_target_y", task_target_y.shape()),
            ("task_priority", task_priority.shape()),
            ("task_deadline", task_deadline.shape()),
            ("task_required_item", task_required_item.shape()),
            ("task_required_count", task_required_count.shape()),
            ("task_exclusive", task_exclusive.shape()),
            ("task_work_role", task_work_role.shape()),
        ] {
            if shape != task_shape {
                return Err(PyValueError::new_err(format!("{name} shape does not match task_active")));
            }
        }
        if unit_role.shape() != active_shape || unit_zone.shape() != active_shape {
            return Err(PyValueError::new_err("workforce unit arrays must match active_units"));
        }
        if reserved_by_kind.shape() != [num_envs, players, TASK_KIND_COUNT] {
            return Err(PyValueError::new_err("reserved_by_kind has incompatible shape"));
        }
        if seat_mask.shape() != [num_envs, players] {
            return Err(PyValueError::new_err("seat_mask has incompatible shape"));
        }
        if out_task_index.shape() != active_shape || out_score.shape() != active_shape {
            return Err(PyValueError::new_err("assignment outputs must match active_units"));
        }

        let shape = (num_envs, players, max_units);
        if self.shape != Some(shape) {
            self.shape = Some(shape);
            self.previous_task = vec![-1; num_envs * players * max_units];
            self.caches = vec![None; num_envs * players];
            self.force_replan = vec![false; num_envs * players];
        }

        let global_guard = global_features.try_readonly()?;
        let units_guard = units.try_readonly()?;
        let active_guard = active_units.try_readonly()?;
        let task_active_guard = task_active.try_readonly()?;
        let task_kind_guard = task_kind.try_readonly()?;
        let task_target_x_guard = task_target_x.try_readonly()?;
        let task_target_y_guard = task_target_y.try_readonly()?;
        let task_priority_guard = task_priority.try_readonly()?;
        let task_deadline_guard = task_deadline.try_readonly()?;
        let task_required_item_guard = task_required_item.try_readonly()?;
        let task_required_count_guard = task_required_count.try_readonly()?;
        let task_exclusive_guard = task_exclusive.try_readonly()?;
        let task_work_role_guard = task_work_role.try_readonly()?;
        let unit_role_guard = unit_role.try_readonly()?;
        let unit_zone_guard = unit_zone.try_readonly()?;
        let reserved_guard = reserved_by_kind.try_readonly()?;
        let seat_mask_guard = seat_mask.try_readonly()?;

        // ObservationViews are zero-copy slices of the flattened observation
        // tensor, so their leading strides are intentionally larger than their
        // logical row width. `as_array()` preserves those NumPy strides without
        // allocating; the task/workforce buffers below remain contiguous.
        let global = global_guard.as_array();
        let units = units_guard.as_array();
        let active_units = active_guard.as_slice()?;
        let task_active = task_active_guard.as_slice()?;
        let task_kind = task_kind_guard.as_slice()?;
        let task_target_x = task_target_x_guard.as_slice()?;
        let task_target_y = task_target_y_guard.as_slice()?;
        let task_priority = task_priority_guard.as_slice()?;
        let task_deadline = task_deadline_guard.as_slice()?;
        let task_required_item = task_required_item_guard.as_slice()?;
        let task_required_count = task_required_count_guard.as_slice()?;
        let task_exclusive = task_exclusive_guard.as_slice()?;
        let task_work_role = task_work_role_guard.as_slice()?;
        let unit_role = unit_role_guard.as_slice()?;
        let unit_zone = unit_zone_guard.as_slice()?;
        let reserved_by_kind = reserved_guard.as_slice()?;
        let seat_mask = seat_mask_guard.as_slice()?;

        let mut task_index_guard = out_task_index.try_readwrite()?;
        let mut score_guard = out_score.try_readwrite()?;
        let out_task_index = task_index_guard.as_slice_mut()?;
        let out_score = score_guard.as_slice_mut()?;
        out_task_index.fill(-1);
        out_score.fill(f32::NEG_INFINITY);

        // Match Python's pre-pass: new-day and inactive workers lose affinity
        // even on seats not selected for scheduling in this call.
        for environment in 0..num_envs {
            for player in 0..players {
                let seat = environment * players + player;
                let step = round_i64(
                    global[[environment, player, 0]] * self.last_step as f32,
                );
                let hour = step.rem_euclid(self.turns_per_day);
                let new_day = hour == 0;
                let unit_offset = seat * max_units;
                for worker in 0..max_units {
                    let index = unit_offset + worker;
                    if new_day || !active_units[index] {
                        self.previous_task[index] = -1;
                    }
                }
            }
        }

        for environment in 0..num_envs {
            for player in 0..players {
                let seat = environment * players + player;
                if !seat_mask[seat] {
                    continue;
                }
                let step = round_i64(
                    global[[environment, player, 0]] * self.last_step as f32,
                );
                let hour_i64 = step.rem_euclid(self.turns_per_day);
                let day_i64 = step.div_euclid(self.turns_per_day);
                let hour = i32::try_from(hour_i64)
                    .map_err(|_| PyValueError::new_err("hour exceeds i32"))?;
                let day = i32::try_from(day_i64)
                    .map_err(|_| PyValueError::new_err("day exceeds i32"))?;

                self.assign_seat(
                    seat,
                    day,
                    hour,
                    players,
                    max_units,
                    task_count,
                    active_units,
                    &units,
                    task_active,
                    task_kind,
                    task_target_x,
                    task_target_y,
                    task_priority,
                    task_deadline,
                    task_required_item,
                    task_required_count,
                    task_exclusive,
                    task_work_role,
                    unit_role,
                    unit_zone,
                    reserved_by_kind,
                    out_task_index,
                    out_score,
                    role_bonus,
                    zone_bonus,
                )?;
            }
        }

        // Match Python exactly: previous_task becomes this call's assignment
        // buffer, so uncontrolled seats are reset to -1.
        self.previous_task.copy_from_slice(out_task_index);
        for (index, active) in active_units.iter().copied().enumerate() {
            if !active {
                self.previous_task[index] = -1;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn assign_and_execute<'py>(
        &mut self,
        global_features: Bound<'py, PyArray3<f32>>,
        units: Bound<'py, PyArray5<f32>>,
        unit_ops: Bound<'py, PyArray4<bool>>,
        unit_args: Bound<'py, PyArray5<bool>>,
        active_units: Bound<'py, PyArray3<bool>>,
        task_active: Bound<'py, PyArray3<bool>>,
        task_kind: Bound<'py, PyArray3<i16>>,
        task_target_x: Bound<'py, PyArray3<i16>>,
        task_target_y: Bound<'py, PyArray3<i16>>,
        task_item: Bound<'py, PyArray3<i16>>,
        task_quantity: Bound<'py, PyArray3<i64>>,
        task_priority: Bound<'py, PyArray3<f32>>,
        task_deadline: Bound<'py, PyArray3<i16>>,
        task_required_item: Bound<'py, PyArray3<i16>>,
        task_required_count: Bound<'py, PyArray3<i64>>,
        task_exclusive: Bound<'py, PyArray3<bool>>,
        task_work_role: Bound<'py, PyArray3<i16>>,
        unit_role: Bound<'py, PyArray3<i16>>,
        unit_zone: Bound<'py, PyArray3<i16>>,
        reserved_by_kind: Bound<'py, PyArray3<i16>>,
        seat_mask: Bound<'py, PyArray2<bool>>,
        out_task_index: Bound<'py, PyArray3<i64>>,
        out_score: Bound<'py, PyArray3<f32>>,
        out_unit_actions: Bound<'py, PyArray4<i64>>,
        role_bonus: f64,
        zone_bonus: f64,
    ) -> PyResult<()> {
        let units_for_execute = units.clone();
        let active_for_execute = active_units.clone();
        let kind_for_execute = task_kind.clone();
        let target_x_for_execute = task_target_x.clone();
        let target_y_for_execute = task_target_y.clone();
        let task_index_for_execute = out_task_index.clone();

        self.assign(
            global_features,
            units,
            active_units,
            task_active,
            task_kind,
            task_target_x,
            task_target_y,
            task_priority,
            task_deadline,
            task_required_item,
            task_required_count,
            task_exclusive,
            task_work_role,
            unit_role,
            unit_zone,
            reserved_by_kind,
            seat_mask,
            out_task_index,
            out_score,
            role_bonus,
            zone_bonus,
        )?;

        execute_assignments(
            units_for_execute,
            unit_ops,
            unit_args,
            active_for_execute,
            kind_for_execute,
            target_x_for_execute,
            target_y_for_execute,
            task_item,
            task_quantity,
            task_index_for_execute,
            out_unit_actions,
            self.board_size,
        )
    }
}

impl NativeTaskScheduler {
    #[allow(clippy::too_many_arguments)]
    fn assign_seat(
        &mut self,
        seat: usize,
        day: i32,
        hour: i32,
        players: usize,
        max_units: usize,
        task_count: usize,
        active_units_all: &[bool],
        units_all: &ArrayView5<'_, f32>,
        task_active_all: &[bool],
        task_kind_all: &[i16],
        task_target_x_all: &[i16],
        task_target_y_all: &[i16],
        task_priority_all: &[f32],
        task_deadline_all: &[i16],
        task_required_item_all: &[i16],
        task_required_count_all: &[i64],
        task_exclusive_all: &[bool],
        task_work_role_all: &[i16],
        unit_role_all: &[i16],
        unit_zone_all: &[i16],
        reserved_by_kind_all: &[i16],
        out_task_index: &mut [i64],
        out_score: &mut [f32],
        role_bonus: f64,
        zone_bonus: f64,
    ) -> PyResult<()> {
        let unit_offset = seat * max_units;
        let task_offset = seat * task_count;
        let reserve_offset = seat * TASK_KIND_COUNT;

        let active_units = (0..max_units)
            .filter(|&worker| active_units_all[unit_offset + worker])
            .collect::<Vec<_>>();
        let active_tasks = (0..task_count)
            .filter(|&task| task_active_all[task_offset + task])
            .collect::<Vec<_>>();

        if active_units.is_empty() || active_tasks.is_empty() {
            self.caches[seat] = None;
            self.force_replan[seat] = false;
            return Ok(());
        }

        let scale = self.board_size.saturating_sub(1) as f32;
        let shed_scale = self.shed_capacity as f32;
        let environment = seat / players;
        let player = seat % players;
        let mut unit_x = vec![0_i16; max_units];
        let mut unit_y = vec![0_i16; max_units];
        let mut inventories = vec![0_i64; max_units * INVENTORY_ITEMS];
        for &worker in &active_units {
            unit_x[worker] = round_i16(units_all[[environment, player, 0, worker, 2]] * scale);
            unit_y[worker] = round_i16(units_all[[environment, player, 0, worker, 3]] * scale);
            for item in 0..INVENTORY_ITEMS {
                inventories[worker * INVENTORY_ITEMS + item] = round_i64(
                    units_all[[environment, player, 0, worker, 5 + item]] * shed_scale,
                );
            }
        }

        let task_active = &task_active_all[task_offset..task_offset + task_count];
        let task_kind = &task_kind_all[task_offset..task_offset + task_count];
        let task_target_x = &task_target_x_all[task_offset..task_offset + task_count];
        let task_target_y = &task_target_y_all[task_offset..task_offset + task_count];
        let task_priority = &task_priority_all[task_offset..task_offset + task_count];
        let task_deadline = &task_deadline_all[task_offset..task_offset + task_count];
        let task_required_item =
            &task_required_item_all[task_offset..task_offset + task_count];
        let task_required_count =
            &task_required_count_all[task_offset..task_offset + task_count];
        let task_exclusive = &task_exclusive_all[task_offset..task_offset + task_count];
        let task_work_role = &task_work_role_all[task_offset..task_offset + task_count];
        let reserved_by_kind =
            &reserved_by_kind_all[reserve_offset..reserve_offset + TASK_KIND_COUNT];

        let unit_role = &unit_role_all[unit_offset..unit_offset + max_units];
        let unit_zone = &unit_zone_all[unit_offset..unit_offset + max_units];
        let half = self.board_size / 2;
        let task_zone = (0..task_count)
            .map(|task| {
                i16::from(u8::from(task_target_y[task] >= half as i16)) * 2
                    + i16::from(u8::from(task_target_x[task] >= half as i16))
            })
            .collect::<Vec<_>>();

        if self.serve_cached(
            seat,
            day,
            hour,
            &active_units,
            &active_tasks,
            task_active,
            task_kind,
            task_target_x,
            task_target_y,
            task_priority,
            task_deadline,
            task_required_item,
            task_required_count,
            task_exclusive,
            task_work_role,
            unit_role,
            unit_zone,
            &task_zone,
            &unit_x,
            &unit_y,
            &inventories,
            max_units,
            task_count,
            out_task_index,
            out_score,
            role_bonus,
            zone_bonus,
        )? {
            self.cache_hits += 1;
            return Ok(());
        }

        self.full_solves += 1;
        let starts_x = active_units.iter().map(|&worker| unit_x[worker]).collect::<Vec<_>>();
        let starts_y = active_units.iter().map(|&worker| unit_y[worker]).collect::<Vec<_>>();
        let mut seat_inventories = Vec::with_capacity(active_units.len() * INVENTORY_ITEMS);
        let mut seat_roles = Vec::with_capacity(active_units.len());
        let mut seat_zones = Vec::with_capacity(active_units.len());
        let mut previous = Vec::with_capacity(active_units.len());
        for &worker in &active_units {
            seat_inventories.extend_from_slice(
                &inventories[worker * INVENTORY_ITEMS..(worker + 1) * INVENTORY_ITEMS],
            );
            seat_roles.push(unit_role[worker]);
            seat_zones.push(unit_zone[worker]);
            previous.push(self.previous_task[unit_offset + worker]);
        }

        let routes = solve_routes_core(
            &starts_x,
            &starts_y,
            &seat_inventories,
            task_active,
            task_exclusive,
            task_priority,
            task_target_x,
            task_target_y,
            task_deadline,
            task_required_item,
            task_required_count,
            task_kind,
            task_work_role,
            &seat_roles,
            &seat_zones,
            &task_zone,
            reserved_by_kind,
            &previous,
            role_bonus,
            zone_bonus,
            self.continuity_bonus,
            self.board_size,
            hour,
            i32::try_from(self.turns_per_day)
                .map_err(|_| PyValueError::new_err("turns_per_day exceeds i32"))?,
        )?;

        // FULL_SOLVE_IDLE_REPAIR: the cache-hit path already repairs empty
        // worker routes by stealing safe future work. Apply the same repair
        // immediately after a fresh solve instead of waiting for a later cache
        // hit. Store `routes` first so the existing repair implementation can
        // operate on the exact same representation.
        self.caches[seat] = Some(SeatCache {
            day,
            units: active_units.clone(),
            routes,
        });
        let full_solve_steals = self.repair_idle_cached_routes(
            seat,
            hour,
            &active_units,
            task_kind,
            task_target_x,
            task_target_y,
            task_deadline,
            task_required_item,
            task_required_count,
            task_exclusive,
            task_work_role,
            unit_role,
            unit_zone,
            &task_zone,
            &unit_x,
            &unit_y,
            &inventories,
            role_bonus,
            zone_bonus,
        )?;
        self.idle_worker_steals += full_solve_steals as u64;

        let first_tasks = self.caches[seat]
            .as_ref()
            .expect("fresh full-solve cache exists")
            .routes
            .iter()
            .map(|route| route.first().copied())
            .collect::<Vec<_>>();

        let mut claimed = vec![false; task_count];
        for (local, &worker) in active_units.iter().enumerate() {
            let Some(task) = first_tasks[local] else {
                continue;
            };
            out_task_index[unit_offset + worker] = task as i64;
            out_score[unit_offset + worker] = self.assignment_score(
                unit_offset,
                worker,
                task,
                task_target_x,
                task_target_y,
                task_priority,
                task_work_role,
                unit_role,
                unit_zone,
                &task_zone,
                &unit_x,
                &unit_y,
                role_bonus,
                zone_bonus,
            ) as f32;
            claimed[task] = true;
        }

        self.assign_nonexclusive(
            seat,
            &active_units,
            &active_tasks,
            task_kind,
            task_target_x,
            task_target_y,
            task_priority,
            task_required_item,
            task_required_count,
            task_exclusive,
            task_work_role,
            unit_role,
            unit_zone,
            &task_zone,
            &unit_x,
            &unit_y,
            &inventories,
            max_units,
            task_count,
            out_task_index,
            out_score,
            role_bonus,
            zone_bonus,
            false,
        )?;

        self.prefer_local_task(
            unit_offset,
            &active_units,
            &mut claimed,
            task_active,
            task_priority,
            task_required_item,
            task_required_count,
            &unit_x,
            &unit_y,
            &inventories,
            task_count,
            out_task_index,
            out_score,
        )?;

        let any_fetch = active_units.iter().any(|&worker| {
            let assigned = out_task_index[unit_offset + worker];
            assigned >= 0 && task_kind[assigned as usize] == TASK_FETCH_ITEM
        });
        if any_fetch {
            self.force_replan[seat] = true;
            self.force_replan_counts[FORCE_FETCH_ITEM] += 1;
        }

        for (local, &worker) in active_units.iter().enumerate() {
            let assigned = out_task_index[unit_offset + worker];
            let planned = first_tasks[local].map_or(-1, |task| task as i64);
            if assigned >= 0 && assigned != planned {
                self.force_replan[seat] = true;
                self.force_replan_counts[FORCE_LOCAL_OVERRIDE] += 1;
                break;
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn serve_cached(
        &mut self,
        seat: usize,
        day: i32,
        hour: i32,
        active_units: &[usize],
        active_tasks: &[usize],
        task_active: &[bool],
        task_kind: &[i16],
        task_target_x: &[i16],
        task_target_y: &[i16],
        task_priority: &[f32],
        task_deadline: &[i16],
        task_required_item: &[i16],
        task_required_count: &[i64],
        task_exclusive: &[bool],
        task_work_role: &[i16],
        unit_role: &[i16],
        unit_zone: &[i16],
        task_zone: &[i16],
        unit_x: &[i16],
        unit_y: &[i16],
        inventories: &[i64],
        max_units: usize,
        task_count: usize,
        out_task_index: &mut [i64],
        out_score: &mut [f32],
        role_bonus: f64,
        zone_bonus: f64,
    ) -> PyResult<bool> {
        let Some(cache) = self.caches[seat].as_ref() else {
            self.cache_miss_counts[MISS_NO_ROUTE] += 1;
            return Ok(false);
        };
        if cache.day != day {
            self.cache_miss_counts[MISS_DAY_CHANGED] += 1;
            return Ok(false);
        }
        if cache.units != active_units {
            self.cache_miss_counts[MISS_UNIT_SET_CHANGED] += 1;
            return Ok(false);
        }
        if self.force_replan[seat] {
            self.force_replan[seat] = false;
            self.cache_miss_counts[MISS_FORCED] += 1;
            return Ok(false);
        }
        if cache.routes.len() != active_units.len() {
            self.cache_miss_counts[MISS_MISSING_WORKER_ROUTE] += 1;
            return Ok(false);
        }

        let mut planned = vec![false; task_count];
        {
            let cache = self.caches[seat]
                .as_mut()
                .expect("cache existence checked above");
            for route in &mut cache.routes {
                route.retain(|&task| task_active[task]);
                for &task in route.iter() {
                    planned[task] = true;
                }
            }
        }
        for &task in active_tasks {
            if task_exclusive[task] && !planned[task] {
                self.cache_miss_counts[MISS_NEW_EXCLUSIVE] += 1;
                return Ok(false);
            }
        }

        let steals = self.repair_idle_cached_routes(
            seat,
            hour,
            active_units,
            task_kind,
            task_target_x,
            task_target_y,
            task_deadline,
            task_required_item,
            task_required_count,
            task_exclusive,
            task_work_role,
            unit_role,
            unit_zone,
            task_zone,
            unit_x,
            unit_y,
            inventories,
            role_bonus,
            zone_bonus,
        )?;
        self.idle_worker_steals += steals as u64;

        let first_tasks = self.caches[seat]
            .as_ref()
            .expect("cache existence checked above")
            .routes
            .iter()
            .map(|route| route.first().copied())
            .collect::<Vec<_>>();

        let unit_offset = seat * max_units;
        let mut claimed = vec![false; task_count];
        for (local, &worker) in active_units.iter().enumerate() {
            let Some(task) = first_tasks[local] else {
                continue;
            };
            let required = task_required_item[task];
            if required >= 0 {
                let item = usize::try_from(required)
                    .map_err(|_| PyValueError::new_err("required item is invalid"))?;
                if item >= INVENTORY_ITEMS
                    || inventories[worker * INVENTORY_ITEMS + item] < task_required_count[task]
                {
                    self.cache_miss_counts[MISS_REQUIRED_ITEM] += 1;
                    return Ok(false);
                }
            }
            if task_kind[task] == TASK_DEPOSIT_INVENTORY {
                let start = worker * INVENTORY_ITEMS;
                if inventories[start..start + INVENTORY_ITEMS]
                    .iter()
                    .copied()
                    .sum::<i64>()
                    <= 0
                {
                    self.cache_miss_counts[MISS_EMPTY_DEPOSIT] += 1;
                    return Ok(false);
                }
            }

            out_task_index[unit_offset + worker] = task as i64;
            out_score[unit_offset + worker] = self.assignment_score(
                unit_offset,
                worker,
                task,
                task_target_x,
                task_target_y,
                task_priority,
                task_work_role,
                unit_role,
                unit_zone,
                task_zone,
                unit_x,
                unit_y,
                role_bonus,
                zone_bonus,
            ) as f32;
            claimed[task] = true;
        }

        self.assign_nonexclusive(
            seat,
            active_units,
            active_tasks,
            task_kind,
            task_target_x,
            task_target_y,
            task_priority,
            task_required_item,
            task_required_count,
            task_exclusive,
            task_work_role,
            unit_role,
            unit_zone,
            task_zone,
            unit_x,
            unit_y,
            inventories,
            max_units,
            task_count,
            out_task_index,
            out_score,
            role_bonus,
            zone_bonus,
            true,
        )?;

        let before = active_units
            .iter()
            .map(|&worker| out_task_index[unit_offset + worker])
            .collect::<Vec<_>>();
        self.prefer_local_task(
            unit_offset,
            active_units,
            &mut claimed,
            task_active,
            task_priority,
            task_required_item,
            task_required_count,
            unit_x,
            unit_y,
            inventories,
            task_count,
            out_task_index,
            out_score,
        )?;
        if active_units
            .iter()
            .enumerate()
            .any(|(local, &worker)| before[local] != out_task_index[unit_offset + worker])
        {
            self.force_replan[seat] = true;
            self.force_replan_counts[FORCE_LOCAL_OVERRIDE] += 1;
        }
        Ok(true)
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn repair_idle_cached_routes(
        &mut self,
        seat: usize,
        hour: i32,
        active_units: &[usize],
        task_kind: &[i16],
        task_target_x: &[i16],
        task_target_y: &[i16],
        task_deadline: &[i16],
        task_required_item: &[i16],
        task_required_count: &[i64],
        task_exclusive: &[bool],
        task_work_role: &[i16],
        unit_role: &[i16],
        unit_zone: &[i16],
        task_zone: &[i16],
        unit_x: &[i16],
        unit_y: &[i16],
        inventories: &[i64],
        role_bonus: f64,
        zone_bonus: f64,
    ) -> PyResult<usize> {
        let turns_per_day = i32::try_from(self.turns_per_day)
            .map_err(|_| PyValueError::new_err("turns_per_day exceeds i32"))?;
        let remaining_turns = turns_per_day - hour;
        if remaining_turns <= 0 {
            return Ok(0);
        }

        let mut steals = 0_usize;

        loop {
            let best = {
                let cache = self.caches[seat]
                    .as_ref()
                    .expect("cache existence checked above");
                let mut best: Option<StealCandidate> = None;

                let route_lengths = cache
                    .routes
                    .iter()
                    .enumerate()
                    .map(|(local, route)| {
                        let worker = active_units[local];
                        cached_route_length(
                            unit_x[worker],
                            unit_y[worker],
                            route,
                            task_target_x,
                            task_target_y,
                        )
                    })
                    .collect::<Vec<_>>();

                for (idle_local, idle_route) in cache.routes.iter().enumerate() {
                    if !idle_route.is_empty() {
                        continue;
                    }
                    let idle_worker = active_units[idle_local];

                    for (donor_local, donor_route) in cache.routes.iter().enumerate() {
                        // Never steal the donor's head. A route of length 1 has no
                        // safely stealable future work.
                        if donor_local == idle_local || donor_route.len() <= 1 {
                            continue;
                        }
                        let donor_worker = active_units[donor_local];
                        let donor_old_length = route_lengths[donor_local];
                        let other_max = route_lengths
                            .iter()
                            .enumerate()
                            .filter(|(local, _)| *local != idle_local && *local != donor_local)
                            .map(|(_, &length)| length)
                            .max()
                            .unwrap_or(0);

                        for position in 1..donor_route.len() {
                            let task = donor_route[position];
                            if !task_exclusive[task]
                                || !eligible(
                                    idle_worker,
                                    task,
                                    inventories,
                                    task_required_item,
                                    task_required_count,
                                    task_kind,
                                )?
                            {
                                continue;
                            }

                            let tx = task_target_x[task];
                            let ty = task_target_y[task];
                            let idle_distance = manhattan(
                                unit_x[idle_worker],
                                unit_y[idle_worker],
                                tx,
                                ty,
                            );
                            let idle_length = idle_distance + 1;

                            // The stolen task becomes the idle worker's head.
                            if idle_length > remaining_turns {
                                continue;
                            }
                            let deadline = task_deadline[task];
                            if deadline >= 0 && hour + idle_distance > i32::from(deadline) {
                                continue;
                            }

                            let prev_task = donor_route[position - 1];
                            let px = task_target_x[prev_task];
                            let py = task_target_y[prev_task];
                            let to_task = manhattan(px, py, tx, ty);
                            let removal_savings = if position + 1 < donor_route.len() {
                                let next_task = donor_route[position + 1];
                                let nx = task_target_x[next_task];
                                let ny = task_target_y[next_task];
                                to_task + manhattan(tx, ty, nx, ny)
                                    - manhattan(px, py, nx, ny)
                                    + 1
                            } else {
                                to_task + 1
                            };
                            let donor_new_length = donor_old_length - removal_savings;
                            let makespan = other_max.max(donor_new_length).max(idle_length);
                            let total_delta = donor_new_length + idle_length - donor_old_length;

                            let donor_preference = worker_task_preference(
                                donor_worker,
                                task,
                                task_work_role,
                                unit_role,
                                unit_zone,
                                task_zone,
                                role_bonus,
                                zone_bonus,
                            );
                            let idle_preference = worker_task_preference(
                                idle_worker,
                                task,
                                task_work_role,
                                unit_role,
                                unit_zone,
                                task_zone,
                                role_bonus,
                                zone_bonus,
                            );

                            let candidate = StealCandidate {
                                makespan,
                                total_delta,
                                preference_loss: donor_preference - idle_preference,
                                idle_distance,
                                idle_local,
                                donor_local,
                                donor_position: position,
                                task,
                            };
                            if best.is_none_or(|current| {
                                steal_candidate_is_better(candidate, current)
                            }) {
                                best = Some(candidate);
                            }
                        }
                    }
                }

                best
            };

            let Some(best) = best else {
                break;
            };

            let cache = self.caches[seat]
                .as_mut()
                .expect("cache existence checked above");
            let removed = cache.routes[best.donor_local].remove(best.donor_position);
            debug_assert_eq!(removed, best.task);
            debug_assert!(cache.routes[best.idle_local].is_empty());
            cache.routes[best.idle_local].push(best.task);
            steals += 1;
        }

        Ok(steals)
    }

    #[allow(clippy::too_many_arguments)]
    fn assign_nonexclusive(
        &mut self,
        seat: usize,
        active_units: &[usize],
        active_tasks: &[usize],
        task_kind: &[i16],
        task_target_x: &[i16],
        task_target_y: &[i16],
        task_priority: &[f32],
        task_required_item: &[i16],
        task_required_count: &[i64],
        task_exclusive: &[bool],
        task_work_role: &[i16],
        unit_role: &[i16],
        unit_zone: &[i16],
        task_zone: &[i16],
        unit_x: &[i16],
        unit_y: &[i16],
        inventories: &[i64],
        max_units: usize,
        _task_count: usize,
        out_task_index: &mut [i64],
        out_score: &mut [f32],
        role_bonus: f64,
        zone_bonus: f64,
        mark_fetch_replan: bool,
    ) -> PyResult<()> {
        let unit_offset = seat * max_units;
        for &worker in active_units {
            if out_task_index[unit_offset + worker] >= 0 {
                continue;
            }
            let mut best: Option<(f64, usize)> = None;
            for &task in active_tasks {
                if task_exclusive[task]
                    || !eligible(
                        worker,
                        task,
                        inventories,
                        task_required_item,
                        task_required_count,
                        task_kind,
                    )?
                {
                    continue;
                }
                let score = self.assignment_score(
                    unit_offset,
                    worker,
                    task,
                    task_target_x,
                    task_target_y,
                    task_priority,
                    task_work_role,
                    unit_role,
                    unit_zone,
                    task_zone,
                    unit_x,
                    unit_y,
                    role_bonus,
                    zone_bonus,
                );
                if best.is_none_or(|(best_score, best_task)| {
                    score > best_score || (score == best_score && task > best_task)
                }) {
                    best = Some((score, task));
                }
            }
            if let Some((score, task)) = best {
                out_task_index[unit_offset + worker] = task as i64;
                out_score[unit_offset + worker] = score as f32;
                if mark_fetch_replan && task_kind[task] == TASK_FETCH_ITEM {
                    self.force_replan[seat] = true;
                    self.force_replan_counts[FORCE_FETCH_ITEM] += 1;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn prefer_local_task(
        &self,
        unit_offset: usize,
        active_units: &[usize],
        claimed: &mut [bool],
        task_active: &[bool],
        task_priority: &[f32],
        task_required_item: &[i16],
        task_required_count: &[i64],
        unit_x: &[i16],
        unit_y: &[i16],
        inventories: &[i64],
        task_count: usize,
        out_task_index: &mut [i64],
        out_score: &mut [f32],
    ) -> PyResult<()> {
        let tile_slots = self.board_size * self.board_size;
        for &worker in active_units {
            let local = usize::try_from(unit_y[worker])
                .ok()
                .and_then(|y| usize::try_from(unit_x[worker]).ok().map(|x| y * self.board_size + x));
            let Some(local) = local else {
                continue;
            };
            if local >= tile_slots || local >= task_count || !task_active[local] || claimed[local] {
                continue;
            }
            let required = task_required_item[local];
            if required >= 0 {
                let item = usize::try_from(required)
                    .map_err(|_| PyValueError::new_err("required item is invalid"))?;
                if item >= INVENTORY_ITEMS
                    || inventories[worker * INVENTORY_ITEMS + item] < task_required_count[local]
                {
                    continue;
                }
            }
            let current = out_task_index[unit_offset + worker];
            let current_priority = if current >= 0 {
                task_priority[current as usize]
            } else {
                f32::NEG_INFINITY
            };
            let local_priority = task_priority[local];
            if local_priority + LOCAL_PRIORITY_SLACK < current_priority {
                continue;
            }
            if current >= 0 {
                claimed[current as usize] = false;
            }
            out_task_index[unit_offset + worker] = local as i64;
            out_score[unit_offset + worker] = (f64::from(local_priority) * 1_000.0) as f32;
            claimed[local] = true;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn assignment_score(
        &self,
        unit_offset: usize,
        worker: usize,
        task: usize,
        task_target_x: &[i16],
        task_target_y: &[i16],
        task_priority: &[f32],
        task_work_role: &[i16],
        unit_role: &[i16],
        unit_zone: &[i16],
        task_zone: &[i16],
        unit_x: &[i16],
        unit_y: &[i16],
        role_bonus: f64,
        zone_bonus: f64,
    ) -> f64 {
        let distance = i32::from((unit_x[worker] - task_target_x[task]).abs())
            + i32::from((unit_y[worker] - task_target_y[task]).abs());
        let mut score = f64::from(task_priority[task]) - f64::from(distance);
        let worker_role = unit_role[worker];
        if worker_role != ROLE_ANY && worker_role == task_work_role[task] {
            score += role_bonus;
        }
        let worker_zone = unit_zone[worker];
        if worker_zone != ZONE_ANY && worker_zone == task_zone[task] {
            score += zone_bonus;
        }
        if self.previous_task[unit_offset + worker] == task as i64 {
            score += self.continuity_bonus;
        }
        score
    }
}

#[inline]
fn manhattan(x0: i16, y0: i16, x1: i16, y1: i16) -> i32 {
    i32::from((x0 - x1).abs()) + i32::from((y0 - y1).abs())
}

fn cached_route_length(
    start_x: i16,
    start_y: i16,
    route: &[usize],
    task_target_x: &[i16],
    task_target_y: &[i16],
) -> i32 {
    let mut x = start_x;
    let mut y = start_y;
    let mut length = 0_i32;
    for &task in route {
        let tx = task_target_x[task];
        let ty = task_target_y[task];
        length += manhattan(x, y, tx, ty) + 1;
        x = tx;
        y = ty;
    }
    length
}

#[allow(clippy::too_many_arguments)]
fn worker_task_preference(
    worker: usize,
    task: usize,
    task_work_role: &[i16],
    unit_role: &[i16],
    unit_zone: &[i16],
    task_zone: &[i16],
    role_bonus: f64,
    zone_bonus: f64,
) -> f64 {
    let mut preference = 0.0_f64;
    let worker_role = unit_role[worker];
    if worker_role != ROLE_ANY && worker_role == task_work_role[task] {
        preference += role_bonus;
    }
    let worker_zone = unit_zone[worker];
    if worker_zone != ZONE_ANY && worker_zone == task_zone[task] {
        preference += zone_bonus;
    }
    preference
}

#[inline]
fn eligible(
    worker: usize,
    task: usize,
    inventories: &[i64],
    required_item: &[i16],
    required_count: &[i64],
    task_kind: &[i16],
) -> PyResult<bool> {
    let required = required_item[task];
    if required >= 0 {
        let item = usize::try_from(required)
            .map_err(|_| PyValueError::new_err("required item is invalid"))?;
        if item >= INVENTORY_ITEMS {
            return Err(PyValueError::new_err("required item is outside inventory"));
        }
        if inventories[worker * INVENTORY_ITEMS + item] < required_count[task] {
            return Ok(false);
        }
    }
    if task_kind[task] == TASK_DEPOSIT_INVENTORY {
        let start = worker * INVENTORY_ITEMS;
        if inventories[start..start + INVENTORY_ITEMS]
            .iter()
            .copied()
            .sum::<i64>()
            <= 0
        {
            return Ok(false);
        }
    }
    Ok(true)
}

#[inline]
fn round_i64(value: f32) -> i64 {
    value.round_ties_even() as i64
}

#[inline]
fn round_i16(value: f32) -> i16 {
    value.round_ties_even() as i16
}
