//! Native task-assignment executor.
//!
//! This mirrors `TaskExecutor.execute` exactly while avoiding the NumPy
//! temporary arrays used by the Python implementation. Observation and mask
//! tensors are zero-copy strided views, so they are read through ndarray
//! ArrayView objects rather than requiring C-contiguity.

#![allow(clippy::all, clippy::pedantic)]

use numpy::{PyArray3, PyArray4, PyArray5, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const TASK_WATER: i16 = 1;
const TASK_FEED: i16 = 2;
const TASK_CARE: i16 = 3;
const TASK_HARVEST: i16 = 4;
const TASK_COLLECT_FERTILIZER: i16 = 5;
const TASK_CLEAR_WEED: i16 = 6;
const TASK_PLANT: i16 = 7;
const TASK_FERTILIZE: i16 = 8;
const TASK_BUILD_COOP: i16 = 9;
const TASK_BUILD_PASTURE: i16 = 10;
const TASK_PLACE_ANIMAL: i16 = 11;
const TASK_FETCH_ITEM: i16 = 12;
const TASK_DEPOSIT_INVENTORY: i16 = 13;

const UNIT_PASS: i64 = 0;
const UNIT_NORTH: i64 = 1;
const UNIT_SOUTH: i64 = 2;
const UNIT_EAST: i64 = 3;
const UNIT_WEST: i64 = 4;
const UNIT_PICKUP: i64 = 5;
const UNIT_DROP: i64 = 6;
const UNIT_PLACE: i64 = 7;
const UNIT_PLANT: i64 = 8;
const UNIT_WATER: i64 = 9;
const UNIT_HARVEST: i64 = 10;
const UNIT_FERTILIZE: i64 = 11;
const UNIT_DIG: i64 = 12;
const UNIT_BUILD_COOP: i64 = 13;
const UNIT_BUILD_PASTURE: i64 = 14;
const UNIT_FEED: i64 = 15;
const UNIT_COLLECT_FERTILIZER: i64 = 16;
const UNIT_CARE: i64 = 17;

#[inline]
fn rounded_i16(value: f32, scale: f32) -> i16 {
    (value * scale).round() as i16
}

#[inline]
fn operation_for_task(kind: i16) -> i64 {
    match kind {
        TASK_WATER => UNIT_WATER,
        TASK_FEED => UNIT_FEED,
        TASK_CARE => UNIT_CARE,
        TASK_HARVEST => UNIT_HARVEST,
        TASK_COLLECT_FERTILIZER => UNIT_COLLECT_FERTILIZER,
        TASK_CLEAR_WEED => UNIT_DIG,
        TASK_PLANT => UNIT_PLANT,
        TASK_FERTILIZE => UNIT_FERTILIZE,
        TASK_BUILD_COOP => UNIT_BUILD_COOP,
        TASK_BUILD_PASTURE => UNIT_BUILD_PASTURE,
        TASK_PLACE_ANIMAL => UNIT_PLACE,
        TASK_FETCH_ITEM => UNIT_PICKUP,
        TASK_DEPOSIT_INVENTORY => UNIT_DROP,
        _ => UNIT_PASS,
    }
}

#[inline]
fn needs_argument(operation: i64) -> bool {
    matches!(operation, UNIT_PICKUP | UNIT_PLACE | UNIT_PLANT)
}

#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
pub(crate) fn execute_assignments<'py>(
    units: Bound<'py, PyArray5<f32>>,
    unit_ops: Bound<'py, PyArray4<bool>>,
    unit_args: Bound<'py, PyArray5<bool>>,
    active_units: Bound<'py, PyArray3<bool>>,
    task_kind: Bound<'py, PyArray3<i16>>,
    task_target_x: Bound<'py, PyArray3<i16>>,
    task_target_y: Bound<'py, PyArray3<i16>>,
    task_item: Bound<'py, PyArray3<i16>>,
    task_quantity: Bound<'py, PyArray3<i64>>,
    task_index: Bound<'py, PyArray3<i64>>,
    out_unit_actions: Bound<'py, PyArray4<i64>>,
    board_size: usize,
) -> PyResult<()> {
    if board_size == 0 {
        return Err(PyValueError::new_err("board_size must be positive"));
    }

    for (name, contiguous) in [
        ("active_units", active_units.is_c_contiguous()),
        ("task_kind", task_kind.is_c_contiguous()),
        ("task_target_x", task_target_x.is_c_contiguous()),
        ("task_target_y", task_target_y.is_c_contiguous()),
        ("task_item", task_item.is_c_contiguous()),
        ("task_quantity", task_quantity.is_c_contiguous()),
        ("task_index", task_index.is_c_contiguous()),
        ("out_unit_actions", out_unit_actions.is_c_contiguous()),
    ] {
        if !contiguous {
            return Err(PyValueError::new_err(format!(
                "{name} must be C-contiguous"
            )));
        }
    }

    let unit_shape = units.shape();
    if unit_shape.len() != 5 || unit_shape[2] < 1 || unit_shape[4] < 4 {
        return Err(PyValueError::new_err("units has incompatible shape"));
    }
    let num_envs = unit_shape[0];
    let players = unit_shape[1];
    let max_units = unit_shape[3];

    if active_units.shape() != [num_envs, players, max_units] {
        return Err(PyValueError::new_err(
            "active_units shape does not match units",
        ));
    }
    if task_index.shape() != [num_envs, players, max_units] {
        return Err(PyValueError::new_err(
            "task_index shape does not match units",
        ));
    }
    if out_unit_actions.shape() != [num_envs, players, max_units, 3] {
        return Err(PyValueError::new_err(
            "unit action output has incompatible shape",
        ));
    }

    let task_shape = task_kind.shape();
    if task_shape.len() != 3 || task_shape[0] != num_envs || task_shape[1] != players {
        return Err(PyValueError::new_err("task arrays have incompatible shape"));
    }
    let task_count = task_shape[2];
    for (name, shape) in [
        ("task_target_x", task_target_x.shape()),
        ("task_target_y", task_target_y.shape()),
        ("task_item", task_item.shape()),
        ("task_quantity", task_quantity.shape()),
    ] {
        if shape != task_shape {
            return Err(PyValueError::new_err(format!(
                "{name} shape does not match task_kind"
            )));
        }
    }

    let ops_shape = unit_ops.shape();
    if ops_shape.len() != 4
        || ops_shape[0] != num_envs
        || ops_shape[1] != players
        || ops_shape[2] != max_units
        || ops_shape[3] <= UNIT_CARE as usize
    {
        return Err(PyValueError::new_err("unit_ops has incompatible shape"));
    }
    let args_shape = unit_args.shape();
    if args_shape.len() != 5
        || args_shape[0] != num_envs
        || args_shape[1] != players
        || args_shape[2] != max_units
        || args_shape[3] != ops_shape[3]
        || args_shape[4] == 0
    {
        return Err(PyValueError::new_err("unit_args has incompatible shape"));
    }

    let units_guard = units.try_readonly()?;
    let ops_guard = unit_ops.try_readonly()?;
    let args_guard = unit_args.try_readonly()?;
    let active_guard = active_units.try_readonly()?;
    let kind_guard = task_kind.try_readonly()?;
    let target_x_guard = task_target_x.try_readonly()?;
    let target_y_guard = task_target_y.try_readonly()?;
    let item_guard = task_item.try_readonly()?;
    let quantity_guard = task_quantity.try_readonly()?;
    let task_index_guard = task_index.try_readonly()?;

    let units = units_guard.as_array();
    let unit_ops = ops_guard.as_array();
    let unit_args = args_guard.as_array();
    let active_units = active_guard.as_slice()?;
    let task_kind = kind_guard.as_slice()?;
    let task_target_x = target_x_guard.as_slice()?;
    let task_target_y = target_y_guard.as_slice()?;
    let task_item = item_guard.as_slice()?;
    let task_quantity = quantity_guard.as_slice()?;
    let task_index = task_index_guard.as_slice()?;

    let mut actions_guard = out_unit_actions.try_readwrite()?;
    let actions = actions_guard.as_slice_mut()?;
    actions.fill(0);

    let mut active_limit = 0_usize;
    for worker in 0..max_units {
        let mut any = false;
        for environment in 0..num_envs {
            for player in 0..players {
                let index = (environment * players + player) * max_units + worker;
                if active_units[index] {
                    any = true;
                    break;
                }
            }
            if any {
                break;
            }
        }
        if any {
            active_limit = worker + 1;
        }
    }
    if active_limit == 0 {
        return Ok(());
    }

    let scale = board_size.saturating_sub(1) as f32;
    let half = board_size / 2;
    let low_center = half.saturating_sub(1) as i16;
    let high_center = half as i16;

    for environment in 0..num_envs {
        for player in 0..players {
            let seat = environment * players + player;
            let task_base = seat * task_count;
            let unit_base = seat * max_units;
            for worker in 0..active_limit {
                let unit_index = unit_base + worker;
                let assigned = task_index[unit_index];
                if assigned < 0 {
                    continue;
                }
                let task = usize::try_from(assigned)
                    .map_err(|_| PyValueError::new_err("task index is invalid"))?;
                if task >= task_count {
                    return Err(PyValueError::new_err("task index is outside task capacity"));
                }
                let task_offset = task_base + task;
                let kind = task_kind[task_offset];
                let item = task_item[task_offset];
                let count = task_quantity[task_offset];

                let unit_x = rounded_i16(units[[environment, player, 0, worker, 2]], scale);
                let unit_y = rounded_i16(units[[environment, player, 0, worker, 3]], scale);
                let mut target_x = task_target_x[task_offset];
                let mut target_y = task_target_y[task_offset];

                let deposit = kind == TASK_DEPOSIT_INVENTORY;
                let at_shed = deposit
                    && (unit_x == low_center || unit_x == high_center)
                    && (unit_y == low_center || unit_y == high_center);
                if deposit {
                    target_x = if unit_x <= low_center {
                        low_center
                    } else {
                        high_center
                    };
                    target_y = if unit_y <= low_center {
                        low_center
                    } else {
                        high_center
                    };
                }

                let moving = !at_shed && (unit_x != target_x || unit_y != target_y);
                let mut operation = if moving {
                    if unit_x < target_x {
                        UNIT_EAST
                    } else if unit_x > target_x {
                        UNIT_WEST
                    } else if unit_y < target_y {
                        UNIT_SOUTH
                    } else {
                        UNIT_NORTH
                    }
                } else {
                    operation_for_task(kind)
                };
                if at_shed {
                    operation = UNIT_DROP;
                }

                let op = usize::try_from(operation)
                    .map_err(|_| PyValueError::new_err("unit operation is invalid"))?;
                if op >= ops_shape[3] || !unit_ops[[environment, player, worker, op]] {
                    continue;
                }

                let interaction = !moving && !at_shed;
                let safe_item = item.max(0) as usize;
                if interaction && needs_argument(operation) {
                    if item < 0
                        || safe_item >= args_shape[4]
                        || !unit_args[[environment, player, worker, op, safe_item]]
                    {
                        continue;
                    }
                }

                let action_offset = unit_index * 3;
                actions[action_offset] = operation;
                // Only operations with an argument may populate the action's
                // arg/count fields.  Routing-only STAGE tasks intentionally
                // map to PASS at their target while retaining task.item as
                // metadata for the future PLANT task.
                if interaction && needs_argument(operation) {
                    actions[action_offset + 1] = safe_item as i64;
                    actions[action_offset + 2] = count;
                }
            }
        }
    }
    Ok(())
}
