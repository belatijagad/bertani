//! Native implementation of the rule policy's route-aware full solver.
//!
//! This mirrors the Python `TaskScheduler` route-construction objective.  It
//! intentionally does *not* own cache validation, non-exclusive logistics,
//! local underfoot overrides, or replan invalidation; those remain in Python.

use std::cmp::Ordering;

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const INVENTORY_ITEMS: usize = 12;
const TASK_KIND_DEPOSIT_INVENTORY: i16 = 13;
const ROLE_ANY: i16 = 0;
const ZONE_ANY: i16 = -1;

#[derive(Debug)]
struct RouteEvalCache {
    prev_x: Vec<i16>,
    prev_y: Vec<i16>,
    elapsed_before: Vec<i32>,
    suffix_new_misses: Vec<i16>,
    suffix_width: usize,
}

impl RouteEvalCache {
    #[inline]
    fn suffix_misses(&self, position: usize, delta: usize) -> i32 {
        i32::from(self.suffix_new_misses[position * self.suffix_width + delta])
    }
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    missed: i32,
    makespan: i32,
    total_length: i32,
    preference_objective: f64,
    local: usize,
    position: usize,
    new_missed: i32,
    new_length: i32,
    new_preference: f64,
}

#[inline]
fn candidate_is_better(candidate: Candidate, best: Candidate) -> bool {
    candidate
        .missed
        .cmp(&best.missed)
        .then_with(|| candidate.makespan.cmp(&best.makespan))
        .then_with(|| candidate.total_length.cmp(&best.total_length))
        .then_with(|| {
            candidate
                .preference_objective
                .partial_cmp(&best.preference_objective)
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| candidate.local.cmp(&best.local))
        .then_with(|| candidate.position.cmp(&best.position))
        == Ordering::Less
}

#[inline]
fn deadline_key(deadline: i16, hour: i32) -> i32 {
    if deadline < 0 {
        1_000_000
    } else {
        (i32::from(deadline) - hour).max(0)
    }
}

#[inline]
fn urgency(priority: f32) -> f32 {
    (priority / 10.0).floor()
}

#[inline]
fn is_eligible(
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
            .map_err(|_| PyValueError::new_err("required item is outside inventory"))?;
        if item >= INVENTORY_ITEMS {
            return Err(PyValueError::new_err("required item is outside inventory"));
        }
        if inventories[worker * INVENTORY_ITEMS + item] < required_count[task] {
            return Ok(false);
        }
    }

    if task_kind[task] == TASK_KIND_DEPOSIT_INVENTORY {
        let start = worker * INVENTORY_ITEMS;
        let carrying = inventories[start..start + INVENTORY_ITEMS]
            .iter()
            .copied()
            .sum::<i64>();
        if carrying <= 0 {
            return Ok(false);
        }
    }

    Ok(true)
}

fn top_two_lengths(lengths: &[i32]) -> (i32, i32, usize) {
    let mut max1 = 0;
    let mut max2 = 0;
    let mut count = 0;

    for &value in lengths {
        if value > max1 {
            max2 = max1;
            max1 = value;
            count = 1;
        } else if value == max1 {
            count += 1;
        } else if value > max2 {
            max2 = value;
        }
    }

    (max1, max2, count)
}

#[allow(clippy::too_many_arguments)]
fn build_route_eval_cache(
    start_x: i16,
    start_y: i16,
    route: &[usize],
    target_x: &[i16],
    target_y: &[i16],
    deadlines: &[i16],
    hour: i32,
    board_size: usize,
) -> RouteEvalCache {
    let route_len = route.len();
    let mut prev_x = vec![0_i16; route_len + 1];
    let mut prev_y = vec![0_i16; route_len + 1];
    let mut elapsed_before = vec![0_i32; route_len + 1];
    let mut completion = vec![0_i32; route_len];

    let mut x = start_x;
    let mut y = start_y;
    let mut elapsed = 0_i32;

    for (index, &task) in route.iter().enumerate() {
        prev_x[index] = x;
        prev_y[index] = y;
        elapsed_before[index] = elapsed;

        let tx = target_x[task];
        let ty = target_y[task];
        elapsed += i32::from((x - tx).abs()) + i32::from((y - ty).abs()) + 1;
        completion[index] = elapsed;
        x = tx;
        y = ty;
    }

    prev_x[route_len] = x;
    prev_y[route_len] = y;
    elapsed_before[route_len] = elapsed;

    // Exact bound for d(prev,new)+d(new,next)-d(prev,next)+1 on a square grid.
    let max_delta = 4 * board_size.saturating_sub(1) + 1;
    let mut suffix_hist = vec![0_i16; (route_len + 1) * max_delta];

    for index in (0..route_len).rev() {
        let row = index * max_delta;
        let next_row = (index + 1) * max_delta;
        for delta in 0..max_delta {
            suffix_hist[row + delta] = suffix_hist[next_row + delta];
        }

        let task = route[index];
        let deadline = deadlines[task];
        if deadline < 0 {
            continue;
        }

        let completion_hour = hour + completion[index] - 1;
        let slack = i32::from(deadline) - completion_hour;
        if slack >= 0 {
            if let Ok(slack) = usize::try_from(slack) {
                if slack < max_delta {
                    suffix_hist[row + slack] += 1;
                }
            }
        }
    }

    let suffix_width = max_delta + 1;
    let mut suffix_new_misses = vec![0_i16; (route_len + 1) * suffix_width];
    for position in 0..=route_len {
        let hist_row = position * max_delta;
        let out_row = position * suffix_width;
        let mut cumulative = 0_i16;
        for delta in 0..max_delta {
            cumulative += suffix_hist[hist_row + delta];
            suffix_new_misses[out_row + delta + 1] = cumulative;
        }
    }

    RouteEvalCache {
        prev_x,
        prev_y,
        elapsed_before,
        suffix_new_misses,
        suffix_width,
    }
}

#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn best_insertion_for_worker(
    local: usize,
    route: &[usize],
    route_cache: &RouteEvalCache,
    first_position: usize,
    old_missed: i32,
    old_length: i32,
    old_preference: f64,
    insert_task: usize,
    other_max: i32,
    total_missed: i32,
    total_length: i32,
    total_preference: f64,
    target_x: &[i16],
    target_y: &[i16],
    deadlines: &[i16],
    task_role: &[i16],
    unit_role: &[i16],
    unit_zone: &[i16],
    task_zone: &[i16],
    previous_task: &[i64],
    role_bonus: f64,
    zone_bonus: f64,
    continuity_bonus: f64,
    hour: i32,
    turns_per_day: i32,
) -> Option<Candidate> {
    let route_len = route.len();
    if first_position > route_len {
        return None;
    }

    let tx = target_x[insert_task];
    let ty = target_y[insert_task];
    let deadline = deadlines[insert_task];

    let mut preference_add = 0.0_f64;
    let worker_role = unit_role[local];
    let assigned_role = task_role[insert_task];
    if worker_role != ROLE_ANY && worker_role == assigned_role {
        preference_add += role_bonus;
    }

    let worker_zone = unit_zone[local];
    if worker_zone != ZONE_ANY && worker_zone == task_zone[insert_task] {
        preference_add += zone_bonus;
    }

    if previous_task[local] == i64::try_from(insert_task).unwrap_or(-1) {
        preference_add += continuity_bonus;
    }

    let new_preference = old_preference + preference_add;
    let objective_preference = -(total_preference - old_preference + new_preference);

    let remaining_turns = turns_per_day - hour;
    let old_overflow = (old_length - remaining_turns).max(0);
    let base_deadline_misses = old_missed - old_overflow;
    let base_total_missed = total_missed - old_missed;
    let base_total_length = total_length - old_length;

    let mut best: Option<Candidate> = None;

    for position in first_position..=route_len {
        let px = route_cache.prev_x[position];
        let py = route_cache.prev_y[position];
        let to_insert = i32::from((px - tx).abs()) + i32::from((py - ty).abs());

        let delta = if position < route_len {
            let next_task = route[position];
            let nx = target_x[next_task];
            let ny = target_y[next_task];
            to_insert
                + i32::from((tx - nx).abs())
                + i32::from((ty - ny).abs())
                - i32::from((px - nx).abs())
                - i32::from((py - ny).abs())
                + 1
        } else {
            to_insert + 1
        };

        let new_length = old_length + delta;
        let inserted_completion = route_cache.elapsed_before[position] + to_insert + 1;

        let mut deadline_misses = base_deadline_misses;
        let completion_hour = hour + inserted_completion - 1;
        if deadline >= 0 && completion_hour > i32::from(deadline) {
            deadline_misses += 1;
        }

        let Ok(delta_index) = usize::try_from(delta) else {
            continue;
        };
        if delta_index >= route_cache.suffix_width {
            continue;
        }
        deadline_misses += route_cache.suffix_misses(position, delta_index);

        let new_missed = deadline_misses + (new_length - remaining_turns).max(0);
        let candidate = Candidate {
            missed: base_total_missed + new_missed,
            makespan: other_max.max(new_length),
            total_length: base_total_length + new_length,
            preference_objective: objective_preference,
            local,
            position,
            new_missed,
            new_length,
            new_preference,
        };

        if best.is_none_or(|current| candidate_is_better(candidate, current)) {
            best = Some(candidate);
        }
    }

    best
}

#[pyfunction]
#[allow(
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::needless_pass_by_value,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
pub(crate) fn schedule_routes<'py>(
    starts_x: Bound<'py, PyArray1<i16>>,
    starts_y: Bound<'py, PyArray1<i16>>,
    inventories: Bound<'py, PyArray2<i64>>,
    active_tasks: Bound<'py, PyArray1<bool>>,
    exclusive: Bound<'py, PyArray1<bool>>,
    priorities: Bound<'py, PyArray1<f32>>,
    target_x: Bound<'py, PyArray1<i16>>,
    target_y: Bound<'py, PyArray1<i16>>,
    deadlines: Bound<'py, PyArray1<i16>>,
    required_item: Bound<'py, PyArray1<i16>>,
    required_count: Bound<'py, PyArray1<i64>>,
    task_kind: Bound<'py, PyArray1<i16>>,
    task_role: Bound<'py, PyArray1<i16>>,
    unit_role: Bound<'py, PyArray1<i16>>,
    unit_zone: Bound<'py, PyArray1<i16>>,
    task_zone: Bound<'py, PyArray1<i16>>,
    reserved_by_kind: Bound<'py, PyArray1<i16>>,
    previous_task: Bound<'py, PyArray1<i64>>,
    role_bonus: f64,
    zone_bonus: f64,
    continuity_bonus: f64,
    board_size: usize,
    hour: i32,
    turns_per_day: i32,
) -> PyResult<Vec<Vec<i64>>> {
    let workers = starts_x.shape();
    if workers.len() != 1 {
        return Err(PyValueError::new_err("starts_x must be one-dimensional"));
    }
    let worker_count = workers[0];
    if starts_y.shape() != [worker_count]
        || unit_role.shape() != [worker_count]
        || unit_zone.shape() != [worker_count]
        || previous_task.shape() != [worker_count]
    {
        return Err(PyValueError::new_err(
            "worker arrays must have the same one-dimensional shape",
        ));
    }
    if inventories.shape() != [worker_count, INVENTORY_ITEMS] {
        return Err(PyValueError::new_err(format!(
            "inventories has shape {:?}, expected {:?}",
            inventories.shape(),
            [worker_count, INVENTORY_ITEMS]
        )));
    }

    let task_shape = active_tasks.shape();
    if task_shape.len() != 1 {
        return Err(PyValueError::new_err("active_tasks must be one-dimensional"));
    }
    let task_count = task_shape[0];
    for (name, shape) in [
        ("exclusive", exclusive.shape()),
        ("priorities", priorities.shape()),
        ("target_x", target_x.shape()),
        ("target_y", target_y.shape()),
        ("deadlines", deadlines.shape()),
        ("required_item", required_item.shape()),
        ("required_count", required_count.shape()),
        ("task_kind", task_kind.shape()),
        ("task_role", task_role.shape()),
        ("task_zone", task_zone.shape()),
    ] {
        if shape != [task_count] {
            return Err(PyValueError::new_err(format!(
                "{name} has shape {shape:?}, expected {:?}",
                [task_count]
            )));
        }
    }
    if reserved_by_kind.shape() != [14] {
        return Err(PyValueError::new_err(format!(
            "reserved_by_kind has shape {:?}, expected [14]",
            reserved_by_kind.shape()
        )));
    }
    if board_size == 0 || board_size.saturating_mul(board_size) > task_count {
        return Err(PyValueError::new_err(
            "board size does not fit inside the task slots",
        ));
    }
    if turns_per_day <= 0 || hour < 0 || hour >= turns_per_day {
        return Err(PyValueError::new_err("hour/turns_per_day are invalid"));
    }

    for (name, contiguous) in [
        ("starts_x", starts_x.is_c_contiguous()),
        ("starts_y", starts_y.is_c_contiguous()),
        ("inventories", inventories.is_c_contiguous()),
        ("active_tasks", active_tasks.is_c_contiguous()),
        ("exclusive", exclusive.is_c_contiguous()),
        ("priorities", priorities.is_c_contiguous()),
        ("target_x", target_x.is_c_contiguous()),
        ("target_y", target_y.is_c_contiguous()),
        ("deadlines", deadlines.is_c_contiguous()),
        ("required_item", required_item.is_c_contiguous()),
        ("required_count", required_count.is_c_contiguous()),
        ("task_kind", task_kind.is_c_contiguous()),
        ("task_role", task_role.is_c_contiguous()),
        ("unit_role", unit_role.is_c_contiguous()),
        ("unit_zone", unit_zone.is_c_contiguous()),
        ("task_zone", task_zone.is_c_contiguous()),
        ("reserved_by_kind", reserved_by_kind.is_c_contiguous()),
        ("previous_task", previous_task.is_c_contiguous()),
    ] {
        if !contiguous {
            return Err(PyValueError::new_err(format!("{name} must be C-contiguous")));
        }
    }

    let starts_x_guard = starts_x.try_readonly()?;
    let starts_y_guard = starts_y.try_readonly()?;
    let inventories_guard = inventories.try_readonly()?;
    let active_tasks_guard = active_tasks.try_readonly()?;
    let exclusive_guard = exclusive.try_readonly()?;
    let priorities_guard = priorities.try_readonly()?;
    let target_x_guard = target_x.try_readonly()?;
    let target_y_guard = target_y.try_readonly()?;
    let deadlines_guard = deadlines.try_readonly()?;
    let required_item_guard = required_item.try_readonly()?;
    let required_count_guard = required_count.try_readonly()?;
    let task_kind_guard = task_kind.try_readonly()?;
    let task_role_guard = task_role.try_readonly()?;
    let unit_role_guard = unit_role.try_readonly()?;
    let unit_zone_guard = unit_zone.try_readonly()?;
    let task_zone_guard = task_zone.try_readonly()?;
    let reserved_by_kind_guard = reserved_by_kind.try_readonly()?;
    let previous_task_guard = previous_task.try_readonly()?;

    let starts_x = starts_x_guard.as_slice()?;
    let starts_y = starts_y_guard.as_slice()?;
    let inventories = inventories_guard.as_slice()?;
    let active_tasks = active_tasks_guard.as_slice()?;
    let exclusive = exclusive_guard.as_slice()?;
    let priorities = priorities_guard.as_slice()?;
    let target_x = target_x_guard.as_slice()?;
    let target_y = target_y_guard.as_slice()?;
    let deadlines = deadlines_guard.as_slice()?;
    let required_item = required_item_guard.as_slice()?;
    let required_count = required_count_guard.as_slice()?;
    let task_kind = task_kind_guard.as_slice()?;
    let task_role = task_role_guard.as_slice()?;
    let unit_role = unit_role_guard.as_slice()?;
    let unit_zone = unit_zone_guard.as_slice()?;
    let task_zone = task_zone_guard.as_slice()?;
    let reserved_by_kind = reserved_by_kind_guard.as_slice()?;
    let previous_task = previous_task_guard.as_slice()?;

    let mut routes = vec![Vec::<usize>::new(); worker_count];
    if worker_count == 0 {
        return Ok(Vec::new());
    }

    let exclusive_tasks = (0..task_count)
        .filter(|&task| active_tasks[task] && exclusive[task])
        .collect::<Vec<_>>();
    if exclusive_tasks.is_empty() {
        return Ok(vec![Vec::new(); worker_count]);
    }

    let mut urgency_bands = exclusive_tasks
        .iter()
        .map(|&task| urgency(priorities[task]))
        .collect::<Vec<_>>();
    urgency_bands.sort_by(|left, right| {
        right.partial_cmp(left).unwrap_or(Ordering::Equal)
    });
    urgency_bands.dedup_by(|left, right| *left == *right);

    let mut prefix_length = vec![0_usize; worker_count];
    let mut route_eval_cache = (0..worker_count)
        .map(|_| None::<RouteEvalCache>)
        .collect::<Vec<_>>();
    let mut route_missed = vec![0_i32; worker_count];
    let mut route_length = vec![0_i32; worker_count];
    let mut route_preference = vec![0_f32; worker_count];
    let mut total_missed = 0_i32;
    let mut total_length = 0_i32;
    let mut total_preference = 0.0_f64;

    for urgency_band in urgency_bands {
        let tier_tasks = exclusive_tasks
            .iter()
            .copied()
            .filter(|&task| urgency(priorities[task]) == urgency_band)
            .collect::<Vec<_>>();
        if tier_tasks.is_empty() {
            continue;
        }

        let future_reserve = if urgency_band >= 12.0 {
            0_usize
        } else {
            let mut lower_kind_count = [0_usize; 14];
            for &task in &exclusive_tasks {
                if urgency(priorities[task]) < urgency_band {
                    if let Ok(kind) = usize::try_from(task_kind[task]) {
                        if kind < lower_kind_count.len() {
                            lower_kind_count[kind] += 1;
                        }
                    }
                }
            }
            lower_kind_count
                .iter()
                .enumerate()
                .map(|(kind, &available)| {
                    let requested = usize::try_from(reserved_by_kind[kind].max(0)).unwrap_or(0);
                    requested.min(available)
                })
                .sum::<usize>()
        };
        let max_workers = worker_count.saturating_sub(future_reserve).max(1);

        let mut ranking = Vec::<(i32, usize, usize)>::new();
        for local in 0..worker_count {
            let (start_x, start_y) = if let Some(&endpoint_task) = routes[local].last() {
                (target_x[endpoint_task], target_y[endpoint_task])
            } else {
                (starts_x[local], starts_y[local])
            };

            let mut best_distance: Option<i32> = None;
            for &task in &tier_tasks {
                if !is_eligible(
                    local,
                    task,
                    inventories,
                    required_item,
                    required_count,
                    task_kind,
                )? {
                    continue;
                }
                let distance = i32::from((target_x[task] - start_x).abs())
                    + i32::from((target_y[task] - start_y).abs());
                best_distance = Some(best_distance.map_or(distance, |current| current.min(distance)));
            }

            if let Some(distance) = best_distance {
                ranking.push((distance, routes[local].len(), local));
            }
        }
        ranking.sort_unstable();
        let candidate_locals = ranking
            .into_iter()
            .take(max_workers)
            .map(|(_, _, local)| local)
            .collect::<Vec<_>>();
        if candidate_locals.is_empty() {
            continue;
        }

        let mut ordered_tasks = tier_tasks;
        ordered_tasks.sort_by(|&left, &right| {
            deadline_key(deadlines[left], hour)
                .cmp(&deadline_key(deadlines[right], hour))
                .then_with(|| {
                    priorities[right]
                        .partial_cmp(&priorities[left])
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| left.cmp(&right))
        });

        for task in ordered_tasks {
            let (max1, max2, max1_count) = top_two_lengths(&route_length);
            let mut best: Option<Candidate> = None;

            for &local in &candidate_locals {
                if !is_eligible(
                    local,
                    task,
                    inventories,
                    required_item,
                    required_count,
                    task_kind,
                )? {
                    continue;
                }

                let first_position = prefix_length[local];
                let old_missed = route_missed[local];
                let old_length = route_length[local];
                let old_preference = f64::from(route_preference[local]);
                let other_max = if old_length == max1 && max1_count == 1 {
                    max2
                } else {
                    max1
                };

                if route_eval_cache[local].is_none() {
                    route_eval_cache[local] = Some(build_route_eval_cache(
                        starts_x[local],
                        starts_y[local],
                        &routes[local],
                        target_x,
                        target_y,
                        deadlines,
                        hour,
                        board_size,
                    ));
                }
                let Some(cache) = route_eval_cache[local].as_ref() else {
                    continue;
                };

                if let Some(candidate) = best_insertion_for_worker(
                    local,
                    &routes[local],
                    cache,
                    first_position,
                    old_missed,
                    old_length,
                    old_preference,
                    task,
                    other_max,
                    total_missed,
                    total_length,
                    total_preference,
                    target_x,
                    target_y,
                    deadlines,
                    task_role,
                    unit_role,
                    unit_zone,
                    task_zone,
                    previous_task,
                    role_bonus,
                    zone_bonus,
                    continuity_bonus,
                    hour,
                    turns_per_day,
                ) {
                    if best.is_none_or(|current| candidate_is_better(candidate, current)) {
                        best = Some(candidate);
                    }
                }
            }

            let Some(best) = best else {
                continue;
            };

            let local = best.local;
            let old_missed = route_missed[local];
            let old_length = route_length[local];
            let old_preference = f64::from(route_preference[local]);

            routes[local].insert(best.position, task);
            route_eval_cache[local] = None;
            route_missed[local] = best.new_missed;
            route_length[local] = best.new_length;
            route_preference[local] = best.new_preference as f32;

            total_missed += best.new_missed - old_missed;
            total_length += best.new_length - old_length;
            total_preference += best.new_preference - old_preference;
        }

        for local in 0..worker_count {
            prefix_length[local] = routes[local].len();
        }
    }

    routes
        .into_iter()
        .map(|route| {
            route
                .into_iter()
                .map(|task| {
                    i64::try_from(task)
                        .map_err(|_| PyValueError::new_err("task index exceeds i64"))
                })
                .collect::<PyResult<Vec<_>>>()
        })
        .collect::<PyResult<Vec<_>>>()
}
