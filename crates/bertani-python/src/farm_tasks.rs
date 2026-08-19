//! Fused native farm-task generation.
//!
//! Maintenance and production remain independently testable kernels, but the
//! production policy calls them through one PyO3 boundary. This keeps the
//! strategy boundary in Python while avoiding a second Python/Rust dispatch.

#![allow(clippy::all, clippy::pedantic)]

use numpy::{PyArray2, PyArray3, PyArray4, PyArray5, PyArray6};
use pyo3::prelude::*;

use crate::maintenance_tasks::propose_maintenance_tasks;
use crate::production_tasks::propose_production_tasks;

#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
pub(crate) fn propose_farm_tasks<'py>(
    tiles: Bound<'py, PyArray6<f32>>,
    global_features: Bound<'py, PyArray3<f32>>,
    units: Bound<'py, PyArray5<f32>>,
    private: Bound<'py, PyArray3<f32>>,
    active_units: Bound<'py, PyArray3<bool>>,
    seat_mask: Bound<'py, PyArray2<bool>>,
    target_crop_counts: Bound<'py, PyArray3<i64>>,
    target_animal_counts: Bound<'py, PyArray3<i64>>,
    liquidate: Bound<'py, PyArray2<bool>>,
    market_actions: Bound<'py, PyArray4<i64>>,
    market_lengths: Bound<'py, PyArray2<i64>>,
    task_active: Bound<'py, PyArray3<bool>>,
    task_kind: Bound<'py, PyArray3<i16>>,
    task_target_x: Bound<'py, PyArray3<i16>>,
    task_target_y: Bound<'py, PyArray3<i16>>,
    task_item: Bound<'py, PyArray3<i16>>,
    task_quantity: Bound<'py, PyArray3<i64>>,
    task_priority: Bound<'py, PyArray3<f32>>,
    task_deadline: Bound<'py, PyArray3<i16>>,
    task_estimated_value: Bound<'py, PyArray3<f32>>,
    task_required_item: Bound<'py, PyArray3<i16>>,
    task_required_count: Bound<'py, PyArray3<i64>>,
    task_exclusive: Bound<'py, PyArray3<bool>>,
    task_work_role: Bound<'py, PyArray3<i16>>,
    board_size: usize,
    tile_slots: usize,
    turns_per_day: i32,
    shed_capacity: i64,
    episode_steps: i64,
) -> PyResult<()> {
    propose_maintenance_tasks(
        tiles.clone(),
        global_features.clone(),
        units.clone(),
        private.clone(),
        active_units.clone(),
        seat_mask.clone(),
        task_active.clone(),
        task_kind.clone(),
        task_target_x.clone(),
        task_target_y.clone(),
        task_item.clone(),
        task_quantity.clone(),
        task_priority.clone(),
        task_deadline.clone(),
        task_estimated_value.clone(),
        task_required_item.clone(),
        task_required_count.clone(),
        task_exclusive.clone(),
        task_work_role.clone(),
        board_size,
        tile_slots,
        turns_per_day,
        shed_capacity,
        episode_steps,
    )?;

    propose_production_tasks(
        tiles,
        global_features,
        units,
        private,
        active_units,
        seat_mask,
        target_crop_counts,
        target_animal_counts,
        liquidate,
        market_actions,
        market_lengths,
        task_active,
        task_kind,
        task_target_x,
        task_target_y,
        task_item,
        task_quantity,
        task_priority,
        task_deadline,
        task_estimated_value,
        task_required_item,
        task_required_count,
        task_exclusive,
        task_work_role,
        board_size,
        tile_slots,
        turns_per_day,
        shed_capacity,
        episode_steps,
    )
}
