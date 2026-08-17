//! Native hot loop for the rule policy's priority-tiered greedy matcher.

use std::cmp::Ordering;

use numpy::{PyArray3, PyArray4, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[allow(
    clippy::float_cmp,
    clippy::needless_pass_by_value,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
pub(crate) fn schedule_tasks<'py>(
    unit_x: Bound<'py, PyArray3<i16>>,
    unit_y: Bound<'py, PyArray3<i16>>,
    inventories: Bound<'py, PyArray4<i64>>,
    priorities: Bound<'py, PyArray3<f32>>,
    active_units: Bound<'py, PyArray3<bool>>,
    active_tasks: Bound<'py, PyArray3<bool>>,
    exclusive: Bound<'py, PyArray3<bool>>,
    target_x: Bound<'py, PyArray3<i16>>,
    target_y: Bound<'py, PyArray3<i16>>,
    required_item: Bound<'py, PyArray3<i16>>,
    required_count: Bound<'py, PyArray3<i64>>,
    task_kind: Bound<'py, PyArray3<i16>>,
    task_role: Bound<'py, PyArray3<i16>>,
    unit_role: Bound<'py, PyArray3<i16>>,
    unit_zone: Bound<'py, PyArray3<i16>>,
    task_zone: Bound<'py, PyArray3<i16>>,
    role_bonus: f32,
    zone_bonus: f32,
    reserved_by_kind: Bound<'py, PyArray3<i16>>,
    previous_task: Bound<'py, PyArray3<i64>>,
    continuity_bonus: f32,
    board_size: usize,
    task_index: Bound<'py, PyArray3<i64>>,
    output_scores: Bound<'py, PyArray3<f32>>,
) -> PyResult<()> {
    let unit_shape = active_units.shape();
    let task_shape = active_tasks.shape();
    if unit_shape.len() != 3 || task_shape.len() != 3 || unit_shape[..2] != task_shape[..2] {
        return Err(PyValueError::new_err(
            "unit and task arrays must be three-dimensional with matching farm axes",
        ));
    }
    let (environments, players, units, tasks) =
        (unit_shape[0], unit_shape[1], unit_shape[2], task_shape[2]);
    let farm_shape = [environments, players];
    let unit_shape = [environments, players, units];
    let task_shape = [environments, players, tasks];
    for (name, actual, expected) in [
        ("unit_x", unit_x.shape(), unit_shape.as_slice()),
        ("unit_y", unit_y.shape(), unit_shape.as_slice()),
        ("priorities", priorities.shape(), task_shape.as_slice()),
        ("active_units", active_units.shape(), unit_shape.as_slice()),
        ("active_tasks", active_tasks.shape(), task_shape.as_slice()),
        ("exclusive", exclusive.shape(), task_shape.as_slice()),
        ("target_x", target_x.shape(), task_shape.as_slice()),
        ("target_y", target_y.shape(), task_shape.as_slice()),
        (
            "required_item",
            required_item.shape(),
            task_shape.as_slice(),
        ),
        (
            "required_count",
            required_count.shape(),
            task_shape.as_slice(),
        ),
        ("task_kind", task_kind.shape(), task_shape.as_slice()),
        ("task_role", task_role.shape(), task_shape.as_slice()),
        ("unit_role", unit_role.shape(), unit_shape.as_slice()),
        ("unit_zone", unit_zone.shape(), unit_shape.as_slice()),
        ("task_zone", task_zone.shape(), task_shape.as_slice()),
        (
            "reserved_by_kind",
            reserved_by_kind.shape(),
            [environments, players, 14].as_slice(),
        ),
        (
            "previous_task",
            previous_task.shape(),
            unit_shape.as_slice(),
        ),
        ("task_index", task_index.shape(), unit_shape.as_slice()),
        (
            "output_scores",
            output_scores.shape(),
            unit_shape.as_slice(),
        ),
    ] {
        if actual != expected {
            return Err(PyValueError::new_err(format!(
                "{name} has shape {actual:?}, expected {expected:?}"
            )));
        }
    }
    if inventories.shape() != [environments, players, units, 12] {
        return Err(PyValueError::new_err(format!(
            "inventories has shape {:?}, expected {:?}",
            inventories.shape(),
            [environments, players, units, 12]
        )));
    }
    if board_size == 0 || board_size.saturating_mul(board_size) > tasks {
        return Err(PyValueError::new_err(
            "board size does not fit inside the task slots",
        ));
    }
    for (name, contiguous) in [
        ("unit_x", unit_x.is_c_contiguous()),
        ("unit_y", unit_y.is_c_contiguous()),
        ("inventories", inventories.is_c_contiguous()),
        ("priorities", priorities.is_c_contiguous()),
        ("active_units", active_units.is_c_contiguous()),
        ("active_tasks", active_tasks.is_c_contiguous()),
        ("exclusive", exclusive.is_c_contiguous()),
        ("target_x", target_x.is_c_contiguous()),
        ("target_y", target_y.is_c_contiguous()),
        ("required_item", required_item.is_c_contiguous()),
        ("required_count", required_count.is_c_contiguous()),
        ("task_kind", task_kind.is_c_contiguous()),
        ("task_role", task_role.is_c_contiguous()),
        ("unit_role", unit_role.is_c_contiguous()),
        ("unit_zone", unit_zone.is_c_contiguous()),
        ("task_zone", task_zone.is_c_contiguous()),
        ("reserved_by_kind", reserved_by_kind.is_c_contiguous()),
        ("previous_task", previous_task.is_c_contiguous()),
        ("task_index", task_index.is_c_contiguous()),
        ("output_scores", output_scores.is_c_contiguous()),
    ] {
        if !contiguous {
            return Err(PyValueError::new_err(format!(
                "{name} must be C-contiguous"
            )));
        }
    }

    let unit_x_guard = unit_x.try_readonly()?;
    let unit_y_guard = unit_y.try_readonly()?;
    let inventories_guard = inventories.try_readonly()?;
    let priorities_guard = priorities.try_readonly()?;
    let active_units_guard = active_units.try_readonly()?;
    let active_tasks_guard = active_tasks.try_readonly()?;
    let exclusive_guard = exclusive.try_readonly()?;
    let target_x_guard = target_x.try_readonly()?;
    let target_y_guard = target_y.try_readonly()?;
    let required_item_guard = required_item.try_readonly()?;
    let required_count_guard = required_count.try_readonly()?;
    let task_kind_guard = task_kind.try_readonly()?;
    let task_role_guard = task_role.try_readonly()?;
    let unit_role_guard = unit_role.try_readonly()?;
    let unit_zone_guard = unit_zone.try_readonly()?;
    let task_zone_guard = task_zone.try_readonly()?;
    let reserved_by_kind_guard = reserved_by_kind.try_readonly()?;
    let previous_task_guard = previous_task.try_readonly()?;
    let unit_x = unit_x_guard.as_slice()?;
    let unit_y = unit_y_guard.as_slice()?;
    let inventories = inventories_guard.as_slice()?;
    let priorities = priorities_guard.as_slice()?;
    let active_units = active_units_guard.as_slice()?;
    let active_tasks = active_tasks_guard.as_slice()?;
    let exclusive = exclusive_guard.as_slice()?;
    let target_x = target_x_guard.as_slice()?;
    let target_y = target_y_guard.as_slice()?;
    let required_item = required_item_guard.as_slice()?;
    let required_count = required_count_guard.as_slice()?;
    let task_kind = task_kind_guard.as_slice()?;
    let task_role = task_role_guard.as_slice()?;
    let unit_role = unit_role_guard.as_slice()?;
    let unit_zone = unit_zone_guard.as_slice()?;
    let task_zone = task_zone_guard.as_slice()?;
    let reserved_by_kind = reserved_by_kind_guard.as_slice()?;
    let previous_task = previous_task_guard.as_slice()?;
    let mut task_index = task_index.try_readwrite()?;
    let mut output_scores = output_scores.try_readwrite()?;
    let task_index = task_index.as_slice_mut()?;
    let output_scores = output_scores.as_slice_mut()?;
    task_index.fill(-1);
    output_scores.fill(f32::NEG_INFINITY);

    let farm_count = farm_shape.iter().product::<usize>();
    for farm in 0..farm_count {
        let unit_start = farm * units;
        let task_start = farm * tasks;
        let inventory_start = farm * units * 12;
        let mut available = active_units[unit_start..unit_start + units].to_vec();
        let mut ordered_tasks = (0..tasks)
            .filter(|&task| active_tasks[task_start + task])
            .collect::<Vec<_>>();
        let urgency_band = |task: usize| (priorities[task_start + task] / 10.0).floor();
        ordered_tasks.sort_by(|&left, &right| {
            urgency_band(right)
                .partial_cmp(&urgency_band(left))
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.cmp(&right))
        });

        let mut tier_start = 0;
        while tier_start < ordered_tasks.len() && available.iter().any(|&value| value) {
            let priority = urgency_band(ordered_tasks[tier_start]);
            let mut tier_end = tier_start + 1;
            while tier_end < ordered_tasks.len()
                && urgency_band(ordered_tasks[tier_end]) == priority
            {
                tier_end += 1;
            }
            let tier = &ordered_tasks[tier_start..tier_end];
            let mut task_available = vec![true; tier.len()];
            let mut lower_kind_count = [0usize; 14];
            for &task in &ordered_tasks[tier_end..] {
                if let Ok(kind) = usize::try_from(task_kind[task_start + task]) {
                    if kind < lower_kind_count.len() {
                        lower_kind_count[kind] += 1;
                    }
                }
            }
            let reserve_start = farm * 14;
            let future_reserve = if priority >= 12.0 {
                0
            } else {
                lower_kind_count
                    .iter()
                    .enumerate()
                    .map(|(kind, &count)| {
                        count.min(
                            usize::try_from(reserved_by_kind[reserve_start + kind].max(0))
                                .unwrap_or(0),
                        )
                    })
                    .sum::<usize>()
            };
            while available.iter().filter(|&&value| value).count() > future_reserve {
                let mut best: Option<(usize, usize, f32)> = None;
                for unit in 0..units {
                    if !available[unit] {
                        continue;
                    }
                    for (offset, &task) in tier.iter().enumerate() {
                        if !task_available[offset] {
                            continue;
                        }
                        let required = required_item[task_start + task];
                        if required >= 0 {
                            let item = usize::try_from(required).map_err(|_| {
                                PyValueError::new_err("required item is outside inventory")
                            })?;
                            if item >= 12
                                || inventories[inventory_start + unit * 12 + item]
                                    < required_count[task_start + task]
                            {
                                continue;
                            }
                        }
                        // DEPOSIT_INVENTORY is task kind 13. It is eligible only
                        // when this particular unit has something to deposit.
                        if task_kind[task_start + task] == 13
                            && inventories
                                [inventory_start + unit * 12..inventory_start + (unit + 1) * 12]
                                .iter()
                                .all(|&count| count == 0)
                        {
                            continue;
                        }
                        let distance = (unit_x[unit_start + unit] - target_x[task_start + task])
                            .abs()
                            + (unit_y[unit_start + unit] - target_y[task_start + task]).abs();
                        let preferred_role = unit_role[unit_start + unit];
                        let assigned_role = task_role[task_start + task];
                        let role_affinity =
                            if preferred_role != 0 && preferred_role == assigned_role {
                                role_bonus
                            } else {
                                0.0
                            };
                        // Territorial affinity applies to board work, but not
                        // global shed/logistics slots.
                        let preferred_zone = unit_zone[unit_start + unit];
                        let flexible_field_work = task_kind[task_start + task] == 6;
                        let zone_affinity = if task < board_size * board_size
                            && flexible_field_work
                            && preferred_zone >= 0
                            && preferred_zone == task_zone[task_start + task]
                        {
                            zone_bonus
                        } else {
                            0.0
                        };
                        let continuity_affinity = if previous_task[unit_start + unit]
                            == i64::try_from(task).unwrap_or(-1)
                        {
                            continuity_bonus
                        } else {
                            0.0
                        };
                        let score = priorities[task_start + task] - f32::from(distance)
                            + role_affinity
                            + zone_affinity
                            + continuity_affinity;
                        if score.is_finite()
                            && best.is_none_or(|(_, _, best_score)| score > best_score)
                        {
                            best = Some((unit, offset, score));
                        }
                    }
                }
                let Some((unit, offset, score)) = best else {
                    break;
                };
                let task = tier[offset];
                task_index[unit_start + unit] = i64::try_from(task)
                    .map_err(|_| PyValueError::new_err("task index exceeds i64"))?;
                output_scores[unit_start + unit] = score;
                available[unit] = false;
                if exclusive[task_start + task] {
                    task_available[offset] = false;
                }
                if !available.iter().any(|&value| value) {
                    break;
                }
            }
            tier_start = tier_end;
        }

        // A worker already standing on a nearly-as-urgent tile should finish
        // it before crossing the board, provided no other worker owns it.
        let mut claimed = vec![false; tasks];
        for unit in 0..units {
            if let Ok(task) = usize::try_from(task_index[unit_start + unit]) {
                if task < tasks {
                    claimed[task] = true;
                }
            }
        }
        for unit in 0..units {
            if !active_units[unit_start + unit] {
                continue;
            }
            let Ok(x) = usize::try_from(unit_x[unit_start + unit]) else {
                continue;
            };
            let Ok(y) = usize::try_from(unit_y[unit_start + unit]) else {
                continue;
            };
            let Some(local) = y.checked_mul(board_size).and_then(|row| row.checked_add(x)) else {
                continue;
            };
            if local >= board_size * board_size
                || !active_tasks[task_start + local]
                || claimed[local]
            {
                continue;
            }
            let required = required_item[task_start + local];
            if required >= 0 {
                let item = usize::try_from(required)
                    .map_err(|_| PyValueError::new_err("required item is outside inventory"))?;
                if item >= 12
                    || inventories[inventory_start + unit * 12 + item]
                        < required_count[task_start + local]
                {
                    continue;
                }
            }
            let current = usize::try_from(task_index[unit_start + unit]).ok();
            let current_priority = current
                .filter(|&task| task < tasks)
                .map_or(f32::NEG_INFINITY, |task| priorities[task_start + task]);
            let local_priority = priorities[task_start + local];
            if local_priority + 5.0 < current_priority {
                continue;
            }
            if let Some(current) = current.filter(|&task| task < tasks) {
                claimed[current] = false;
            }
            task_index[unit_start + unit] = i64::try_from(local)
                .map_err(|_| PyValueError::new_err("task index exceeds i64"))?;
            output_scores[unit_start + unit] = local_priority * 1_000.0;
            claimed[local] = true;
        }
    }
    Ok(())
}
