//! Native production-task generation for the rule policy.
//!
//! Python keeps the strategic targets (`StrategicIntent`). This module only
//! translates those targets plus the current encoded farm state into concrete
//! deterministic board tasks: clearing exhausted tiles, building pasture,
//! choosing planting locations, placing livestock, fetching animals, and
//! depositing carried inventory.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::needless_pass_by_value,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

use numpy::{
    PyArray2, PyArray3, PyArray5, PyArray6, PyArrayMethods, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const ITEM_COUNT: usize = 12;
const CROP_COUNT: usize = 5;
const ANIMAL_COUNT: usize = 3;

const ITEM_WHEAT: i16 = 0;
const ITEM_CARROT: i16 = 1;
const ITEM_TOMATO: i16 = 2;
const ITEM_STRAWBERRY: i16 = 3;
const ITEM_MELON: i16 = 4;
const ITEM_MILK: usize = 6;
const ITEM_WOOL: usize = 7;
const ITEM_COW: i16 = 10;
const ITEM_SHEEP: i16 = 11;

const TASK_NONE: i16 = 0;
const TASK_CLEAR_WEED: i16 = 6;
const TASK_PLANT: i16 = 7;
const TASK_BUILD_PASTURE: i16 = 10;
const TASK_PLACE_ANIMAL: i16 = 11;
const TASK_FETCH_ITEM: i16 = 12;
const TASK_DEPOSIT_INVENTORY: i16 = 13;

const ROLE_ANY: i16 = 0;
const ROLE_LOGISTICS: i16 = 1;
const ROLE_LIVESTOCK: i16 = 2;
const ROLE_FIELD: i16 = 3;

const TILE_EMPTY: usize = 0;
const TILE_WEED: usize = 2;
const TILE_EXISTING_PRODUCTION_START: usize = 3;
const TILE_EXISTING_PRODUCTION_END: usize = 9;
const TILE_EMPTY_PASTURE: usize = 5;
const TILE_ANIMAL_START: usize = 6;
const TILE_CROP_START: usize = 9;
const TILE_CROP_AGE: usize = 14;
const TILE_HARVESTABLE: usize = 23;

const UNIT_INVENTORY_START: usize = 5;
const PRIVATE_SEED_START: usize = 12;
const SEED_SCALE: f32 = 10.0;

const PASTURE_SLOTS: [(usize, usize); 18] = [
    (3, 3),
    (4, 3),
    (3, 4),
    (4, 4),
    (2, 4),
    (4, 2),
    (5, 3),
    (5, 4),
    (6, 4),
    (6, 3),
    (5, 2),
    (7, 4),
    (3, 5),
    (4, 5),
    (6, 5),
    (6, 6),
    (5, 6),
    (5, 7),
];

struct TaskOutput<'a> {
    active: &'a mut [bool],
    kind: &'a mut [i16],
    target_x: &'a mut [i16],
    target_y: &'a mut [i16],
    item: &'a mut [i16],
    quantity: &'a mut [i64],
    priority: &'a mut [f32],
    deadline: &'a mut [i16],
    estimated_value: &'a mut [f32],
    required_item: &'a mut [i16],
    required_count: &'a mut [i64],
    exclusive: &'a mut [bool],
    work_role: &'a mut [i16],
}

#[inline]
fn propose_tile(
    out: &mut TaskOutput<'_>,
    index: usize,
    kind: i16,
    priority: f32,
    item: i16,
    required_item: i16,
    required_count: i64,
    work_role: i16,
) {
    // Match TaskBatch.propose_tiles exactly: ties do not replace the incumbent.
    if priority <= out.priority[index] {
        return;
    }
    out.active[index] = true;
    out.kind[index] = kind;
    out.item[index] = item;
    out.quantity[index] = 1;
    out.priority[index] = priority;
    out.deadline[index] = -1;
    out.estimated_value[index] = 0.0;
    out.required_item[index] = required_item;
    out.required_count[index] = required_count;
    out.exclusive[index] = true;
    out.work_role[index] = work_role;
}

#[inline]
fn set_global(
    out: &mut TaskOutput<'_>,
    index: usize,
    active: bool,
    kind: i16,
    target_x: i16,
    target_y: i16,
    priority: f32,
    item: i16,
    quantity: i64,
    exclusive: bool,
    work_role: i16,
) {
    out.active[index] = active;
    out.kind[index] = if active { kind } else { TASK_NONE };
    out.target_x[index] = target_x;
    out.target_y[index] = target_y;
    out.item[index] = item;
    out.quantity[index] = quantity;
    out.priority[index] = if active { priority } else { f32::NEG_INFINITY };
    out.deadline[index] = -1;
    out.estimated_value[index] = 0.0;
    out.required_item[index] = -1;
    out.required_count[index] = 0;
    out.exclusive[index] = exclusive;
    out.work_role[index] = if active { work_role } else { ROLE_ANY };
}

#[inline]
fn rounded_i64(value: f32, scale: f32) -> i64 {
    // Encoded discrete values are exact integer/scale ratios in this interface.
    (value * scale).round() as i64
}

struct Geometry {
    center_distance: Vec<i64>,
    pasture_rank: Vec<i64>,
    quadrant: Vec<usize>,
    tie_y: Vec<i64>,
    tie_x: Vec<i64>,
}

fn build_geometry(board_size: usize) -> Geometry {
    let tile_slots = board_size * board_size;
    let half = board_size / 2;
    let low_center = half.saturating_sub(1);
    let mut center_distance = vec![0_i64; tile_slots];
    let mut pasture_rank = vec![-1_i64; tile_slots];
    let mut quadrant = vec![0_usize; tile_slots];
    let mut tie_y = vec![0_i64; tile_slots];
    let mut tie_x = vec![0_i64; tile_slots];

    for y in 0..board_size {
        for x in 0..board_size {
            let index = y * board_size + x;
            let dx = x.abs_diff(low_center).min(x.abs_diff(half));
            let dy = y.abs_diff(low_center).min(y.abs_diff(half));
            center_distance[index] = (dx + dy) as i64;
            tie_x[index] = dx as i64;
            tie_y[index] = dy as i64;
            quadrant[index] = usize::from(y >= half) * 2 + usize::from(x >= half);
        }
    }
    for (rank, &(x, y)) in PASTURE_SLOTS.iter().enumerate() {
        if x < board_size && y < board_size {
            pasture_rank[y * board_size + x] = rank as i64;
        }
    }

    Geometry {
        center_distance,
        pasture_rank,
        quadrant,
        tie_y,
        tie_x,
    }
}

fn select_limited(candidates: &[bool], count: i64) -> Vec<bool> {
    let mut selected = vec![false; candidates.len()];
    if count <= 0 {
        return selected;
    }
    let mut remaining = count as usize;
    for (index, &candidate) in candidates.iter().enumerate() {
        if !candidate {
            continue;
        }
        selected[index] = true;
        remaining -= 1;
        if remaining == 0 {
            break;
        }
    }
    selected
}

fn select_limited_by_distance(
    candidates: &[bool],
    count: i64,
    distance: &[i64],
    existing: &[bool],
    geometry: &Geometry,
) -> Vec<bool> {
    let mut selected = vec![false; candidates.len()];
    if count <= 0 {
        return selected;
    }

    let mut queues: [Vec<usize>; 4] = std::array::from_fn(|_| Vec::new());
    for (index, &candidate) in candidates.iter().enumerate() {
        if candidate {
            queues[geometry.quadrant[index]].push(index);
        }
    }
    for queue in &mut queues {
        queue.sort_unstable_by_key(|&index| {
            (
                distance[index],
                geometry.tie_y[index],
                geometry.tie_x[index],
                index,
            )
        });
    }

    let mut quadrant_counts = [0_i64; 4];
    for (index, &occupied) in existing.iter().enumerate() {
        if occupied {
            quadrant_counts[geometry.quadrant[index]] += 1;
        }
    }

    let mut pointers = [0_usize; 4];
    let available = queues.iter().map(Vec::len).sum::<usize>();
    let limit = (count as usize).min(available);

    for _ in 0..limit {
        let nearest = (0..4)
            .filter_map(|quadrant| {
                queues[quadrant]
                    .get(pointers[quadrant])
                    .map(|&index| distance[index])
            })
            .min();
        let Some(nearest) = nearest else {
            break;
        };

        let least_occupied = (0..4)
            .filter_map(|quadrant| {
                let index = *queues[quadrant].get(pointers[quadrant])?;
                (distance[index] == nearest).then_some(quadrant_counts[quadrant])
            })
            .min()
            .expect("nearest queue head must exist");

        let chosen = (0..4)
            .filter_map(|quadrant| {
                let index = *queues[quadrant].get(pointers[quadrant])?;
                if distance[index] != nearest || quadrant_counts[quadrant] != least_occupied {
                    return None;
                }
                Some((
                    (
                        geometry.tie_y[index],
                        geometry.tie_x[index],
                        index,
                    ),
                    quadrant,
                    index,
                ))
            })
            .min_by_key(|candidate| candidate.0)
            .expect("balanced queue head must exist");

        let (_, quadrant, index) = chosen;
        selected[index] = true;
        quadrant_counts[quadrant] += 1;
        pointers[quadrant] += 1;
    }

    selected
}

#[pyfunction]
pub(crate) fn propose_production_tasks<'py>(
    tiles: Bound<'py, PyArray6<f32>>,
    global_features: Bound<'py, PyArray3<f32>>,
    units: Bound<'py, PyArray5<f32>>,
    private: Bound<'py, PyArray3<f32>>,
    active_units: Bound<'py, PyArray3<bool>>,
    target_crop_counts: Bound<'py, PyArray3<i64>>,
    target_animal_counts: Bound<'py, PyArray3<i64>>,
    liquidate: Bound<'py, PyArray2<bool>>,
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
    if board_size == 0 || turns_per_day <= 0 || shed_capacity <= 0 || episode_steps <= 0 {
        return Err(PyValueError::new_err("production configuration must be positive"));
    }
    if tile_slots != board_size.saturating_mul(board_size) {
        return Err(PyValueError::new_err("tile_slots must equal board_size**2"));
    }

    let tile_shape = tiles.shape();
    if tile_shape.len() != 6
        || tile_shape[2] < 1
        || tile_shape[3] != board_size
        || tile_shape[4] != board_size
        || tile_shape[5] <= TILE_HARVESTABLE
    {
        return Err(PyValueError::new_err(format!(
            "tiles has incompatible shape {tile_shape:?}"
        )));
    }
    let num_envs = tile_shape[0];
    let players = tile_shape[1];

    let global_shape = global_features.shape();
    if global_shape.len() != 3 || global_shape[0] != num_envs || global_shape[1] != players {
        return Err(PyValueError::new_err("global_features batch shape does not match tiles"));
    }
    let unit_shape = units.shape();
    if unit_shape.len() != 5
        || unit_shape[0] != num_envs
        || unit_shape[1] != players
        || unit_shape[2] < 1
        || unit_shape[4] < UNIT_INVENTORY_START + ITEM_COUNT
    {
        return Err(PyValueError::new_err("units shape is incompatible with production tasks"));
    }
    let max_units = unit_shape[3];
    if active_units.shape() != [num_envs, players, max_units] {
        return Err(PyValueError::new_err("active_units shape does not match units"));
    }
    let private_shape = private.shape();
    if private_shape.len() != 3
        || private_shape[0] != num_envs
        || private_shape[1] != players
        || private_shape[2] < PRIVATE_SEED_START + CROP_COUNT
    {
        return Err(PyValueError::new_err("private shape is incompatible with production tasks"));
    }
    if target_crop_counts.shape() != [num_envs, players, CROP_COUNT] {
        return Err(PyValueError::new_err("target_crop_counts must have shape [N, P, 5]"));
    }
    if target_animal_counts.shape() != [num_envs, players, ANIMAL_COUNT] {
        return Err(PyValueError::new_err("target_animal_counts must have shape [N, P, 3]"));
    }
    if liquidate.shape() != [num_envs, players] {
        return Err(PyValueError::new_err("liquidate must have shape [N, P]"));
    }

    let output_shape = task_active.shape();
    if output_shape.len() != 3 || output_shape[0] != num_envs || output_shape[1] != players {
        return Err(PyValueError::new_err("task batch shape does not match observations"));
    }
    let capacity = output_shape[2];
    if capacity <= tile_slots + 2 {
        return Err(PyValueError::new_err(
            "task batch needs at least three global slots for production",
        ));
    }
    for (name, shape) in [
        ("task_kind", task_kind.shape()),
        ("task_target_x", task_target_x.shape()),
        ("task_target_y", task_target_y.shape()),
        ("task_item", task_item.shape()),
        ("task_quantity", task_quantity.shape()),
        ("task_priority", task_priority.shape()),
        ("task_deadline", task_deadline.shape()),
        ("task_estimated_value", task_estimated_value.shape()),
        ("task_required_item", task_required_item.shape()),
        ("task_required_count", task_required_count.shape()),
        ("task_exclusive", task_exclusive.shape()),
        ("task_work_role", task_work_role.shape()),
    ] {
        if shape != output_shape {
            return Err(PyValueError::new_err(format!(
                "{name} shape {shape:?} does not match task_active {output_shape:?}"
            )));
        }
    }
    for (name, contiguous) in [
        ("task_active", task_active.is_c_contiguous()),
        ("task_kind", task_kind.is_c_contiguous()),
        ("task_target_x", task_target_x.is_c_contiguous()),
        ("task_target_y", task_target_y.is_c_contiguous()),
        ("task_item", task_item.is_c_contiguous()),
        ("task_quantity", task_quantity.is_c_contiguous()),
        ("task_priority", task_priority.is_c_contiguous()),
        ("task_deadline", task_deadline.is_c_contiguous()),
        ("task_estimated_value", task_estimated_value.is_c_contiguous()),
        ("task_required_item", task_required_item.is_c_contiguous()),
        ("task_required_count", task_required_count.is_c_contiguous()),
        ("task_exclusive", task_exclusive.is_c_contiguous()),
        ("task_work_role", task_work_role.is_c_contiguous()),
    ] {
        if !contiguous {
            return Err(PyValueError::new_err(format!("{name} must be C-contiguous")));
        }
    }

    let tiles_guard = tiles.try_readonly()?;
    let global_guard = global_features.try_readonly()?;
    let units_guard = units.try_readonly()?;
    let private_guard = private.try_readonly()?;
    let active_units_guard = active_units.try_readonly()?;
    let target_crop_guard = target_crop_counts.try_readonly()?;
    let target_animal_guard = target_animal_counts.try_readonly()?;
    let liquidate_guard = liquidate.try_readonly()?;
    let tiles = tiles_guard.as_array();
    let global_features = global_guard.as_array();
    let units = units_guard.as_array();
    let private = private_guard.as_array();
    let active_units = active_units_guard.as_array();
    let target_crop_counts = target_crop_guard.as_array();
    let target_animal_counts = target_animal_guard.as_array();
    let liquidate = liquidate_guard.as_array();

    let mut active_guard = task_active.try_readwrite()?;
    let mut kind_guard = task_kind.try_readwrite()?;
    let mut target_x_guard = task_target_x.try_readwrite()?;
    let mut target_y_guard = task_target_y.try_readwrite()?;
    let mut item_guard = task_item.try_readwrite()?;
    let mut quantity_guard = task_quantity.try_readwrite()?;
    let mut priority_guard = task_priority.try_readwrite()?;
    let mut deadline_guard = task_deadline.try_readwrite()?;
    let mut estimated_value_guard = task_estimated_value.try_readwrite()?;
    let mut required_item_guard = task_required_item.try_readwrite()?;
    let mut required_count_guard = task_required_count.try_readwrite()?;
    let mut exclusive_guard = task_exclusive.try_readwrite()?;
    let mut work_role_guard = task_work_role.try_readwrite()?;

    let mut out = TaskOutput {
        active: active_guard.as_slice_mut()?,
        kind: kind_guard.as_slice_mut()?,
        target_x: target_x_guard.as_slice_mut()?,
        target_y: target_y_guard.as_slice_mut()?,
        item: item_guard.as_slice_mut()?,
        quantity: quantity_guard.as_slice_mut()?,
        priority: priority_guard.as_slice_mut()?,
        deadline: deadline_guard.as_slice_mut()?,
        estimated_value: estimated_value_guard.as_slice_mut()?,
        required_item: required_item_guard.as_slice_mut()?,
        required_count: required_count_guard.as_slice_mut()?,
        exclusive: exclusive_guard.as_slice_mut()?,
        work_role: work_role_guard.as_slice_mut()?,
    };

    let episode_days = (episode_steps + i64::from(turns_per_day) - 1) / i64::from(turns_per_day);
    let last_step = (episode_steps - 1).max(1) as f32;
    let shed_scale = shed_capacity as f32;
    let access = i16::try_from((board_size / 2).saturating_sub(1))
        .map_err(|_| PyValueError::new_err("board size does not fit task coordinates"))?;
    let geometry = build_geometry(board_size);

    for environment in 0..num_envs {
        for player in 0..players {
            let step = rounded_i64(global_features[[environment, player, 0]], last_step);
            let day = step / i64::from(turns_per_day);
            let hour = step % i64::from(turns_per_day);
            let productive = !liquidate[[environment, player]];
            let seat_base = (environment * players + player) * capacity;

            let mut empty_tile = vec![false; tile_slots];
            let mut weed = vec![false; tile_slots];
            let mut exhausted_ongoing = vec![false; tile_slots];
            let mut existing_pastures = vec![false; tile_slots];
            let mut existing_production = vec![false; tile_slots];
            let mut empty_pasture = vec![false; tile_slots];
            let mut crop_counts = [0_i64; CROP_COUNT];
            let mut animal_counts = [0_i64; ANIMAL_COUNT];
            let mut pasture_count = 0_i64;

            for y in 0..board_size {
                for x in 0..board_size {
                    let slot = y * board_size + x;
                    let tile = |channel: usize| tiles[[environment, player, 0, y, x, channel]];
                    empty_tile[slot] = tile(TILE_EMPTY) > 0.5;
                    weed[slot] = tile(TILE_WEED) > 0.5 && productive;
                    existing_pastures[slot] = tile(TILE_EMPTY_PASTURE) > 0.5
                        || tile(TILE_ANIMAL_START + 1) > 0.5
                        || tile(TILE_ANIMAL_START + 2) > 0.5;
                    empty_pasture[slot] = tile(TILE_EMPTY_PASTURE) > 0.5 && productive;
                    existing_production[slot] = (TILE_EXISTING_PRODUCTION_START
                        ..TILE_EXISTING_PRODUCTION_END)
                        .any(|channel| tile(channel) > 0.5);

                    for crop in 0..CROP_COUNT {
                        crop_counts[crop] += i64::from(tile(TILE_CROP_START + crop) > 0.5);
                    }
                    for animal in 0..ANIMAL_COUNT {
                        animal_counts[animal] +=
                            i64::from(tile(TILE_ANIMAL_START + animal) > 0.5);
                    }
                    pasture_count += i64::from(existing_pastures[slot]);

                    let crop_age = rounded_i64(tile(TILE_CROP_AGE), episode_days as f32);
                    let harvestable = tile(TILE_HARVESTABLE) > 0.5;
                    let tomato = tile(TILE_CROP_START + ITEM_TOMATO as usize) > 0.5;
                    let strawberry = tile(TILE_CROP_START + ITEM_STRAWBERRY as usize) > 0.5;
                    exhausted_ongoing[slot] = productive
                        && !harvestable
                        && day <= episode_days - 5
                        && ((tomato && crop_age >= 11) || (strawberry && crop_age >= 16));
                }
            }

            let weed_count = weed.iter().filter(|&&value| value).count() as i64;
            let weed_priority = if day >= 22 && weed_count >= 4 { 109.0 } else { 99.0 };
            for slot in 0..tile_slots {
                if weed[slot] {
                    propose_tile(
                        &mut out,
                        seat_base + slot,
                        TASK_CLEAR_WEED,
                        weed_priority,
                        -1,
                        -1,
                        0,
                        ROLE_FIELD,
                    );
                }
                if exhausted_ongoing[slot] {
                    propose_tile(
                        &mut out,
                        seat_base + slot,
                        TASK_CLEAR_WEED,
                        99.0,
                        -1,
                        -1,
                        0,
                        ROLE_FIELD,
                    );
                }
            }

            let target_pastures = target_animal_counts[[environment, player, 1]]
                + target_animal_counts[[environment, player, 2]];
            let missing_pastures = (target_pastures - pasture_count).max(0);
            let mut build_candidates = vec![false; tile_slots];
            for slot in 0..tile_slots {
                let rank = geometry.pasture_rank[slot];
                build_candidates[slot] = productive
                    && empty_tile[slot]
                    && rank >= 0
                    && rank < target_pastures;
            }
            let build = select_limited_by_distance(
                &build_candidates,
                missing_pastures,
                &geometry.pasture_rank,
                &existing_pastures,
                &geometry,
            );
            for (slot, &selected) in build.iter().enumerate() {
                if selected {
                    propose_tile(
                        &mut out,
                        seat_base + slot,
                        TASK_BUILD_PASTURE,
                        105.0,
                        -1,
                        -1,
                        0,
                        ROLE_LIVESTOCK,
                    );
                }
            }

            let safe_to_plant = hour < i64::from(turns_per_day - 2);
            let reserved_pasture_count = target_pastures.max(14);
            let mut empty = vec![false; tile_slots];
            for slot in 0..tile_slots {
                let rank = geometry.pasture_rank[slot];
                let reserved = rank >= 0 && rank < reserved_pasture_count;
                empty[slot] = productive && safe_to_plant && empty_tile[slot] && !reserved;
            }

            let mut seeds = [0_i64; CROP_COUNT];
            for crop in 0..CROP_COUNT {
                seeds[crop] = rounded_i64(
                    private[[environment, player, PRIVATE_SEED_START + crop]],
                    SEED_SCALE,
                );
            }
            let mut claimed = build.clone();
            let mut planned_seed_use = [0_i64; CROP_COUNT];
            let target_cash_crops = (1..CROP_COUNT)
                .map(|crop| target_crop_counts[[environment, player, crop]])
                .sum::<i64>();
            let existing_cash_crops = crop_counts[1..].iter().sum::<i64>();
            let mut cash_slots = (target_cash_crops - existing_cash_crops).max(0);

            for crop in [
                ITEM_WHEAT,
                ITEM_MELON,
                ITEM_CARROT,
                ITEM_TOMATO,
                ITEM_STRAWBERRY,
            ] {
                let crop_index = crop as usize;
                let deficit = (target_crop_counts[[environment, player, crop_index]]
                    - crop_counts[crop_index])
                    .max(0);
                let mut available = deficit.min(seeds[crop_index]);
                if crop != ITEM_WHEAT {
                    available = available.min(cash_slots);
                }
                let candidates = empty
                    .iter()
                    .zip(&claimed)
                    .map(|(&is_empty, &is_claimed)| is_empty && !is_claimed)
                    .collect::<Vec<_>>();
                let occupied = existing_production
                    .iter()
                    .zip(&claimed)
                    .map(|(&existing, &is_claimed)| existing || is_claimed)
                    .collect::<Vec<_>>();
                let selected = select_limited_by_distance(
                    &candidates,
                    available,
                    &geometry.center_distance,
                    &occupied,
                    &geometry,
                );
                let selected_count = selected.iter().filter(|&&value| value).count() as i64;
                for (slot, &is_selected) in selected.iter().enumerate() {
                    if is_selected {
                        claimed[slot] = true;
                        propose_tile(
                            &mut out,
                            seat_base + slot,
                            TASK_PLANT,
                            if (7..=8).contains(&day) || (11..=13).contains(&day) {
                                115.0
                            } else {
                                97.0
                            },
                            crop,
                            -1,
                            0,
                            ROLE_FIELD,
                        );
                    }
                }
                planned_seed_use[crop_index] += selected_count;
                if crop != ITEM_WHEAT {
                    cash_slots -= selected_count;
                }
            }

            let remaining_days = episode_days - day;
            for (crop, maturity_days) in [
                (ITEM_WHEAT, 4_i64),
                (ITEM_CARROT, 3_i64),
                (ITEM_MELON, 10_i64),
                (ITEM_TOMATO, 8_i64),
                (ITEM_STRAWBERRY, 10_i64),
            ] {
                let crop_index = crop as usize;
                let surplus = (seeds[crop_index] - planned_seed_use[crop_index]).max(0);
                let available = if remaining_days > maturity_days { surplus } else { 0 };
                let candidates = empty
                    .iter()
                    .zip(&claimed)
                    .map(|(&is_empty, &is_claimed)| is_empty && !is_claimed)
                    .collect::<Vec<_>>();
                let occupied = existing_production
                    .iter()
                    .zip(&claimed)
                    .map(|(&existing, &is_claimed)| existing || is_claimed)
                    .collect::<Vec<_>>();
                let selected = select_limited_by_distance(
                    &candidates,
                    available,
                    &geometry.center_distance,
                    &occupied,
                    &geometry,
                );
                let selected_count = selected.iter().filter(|&&value| value).count() as i64;
                for (slot, &is_selected) in selected.iter().enumerate() {
                    if is_selected {
                        claimed[slot] = true;
                        propose_tile(
                            &mut out,
                            seat_base + slot,
                            TASK_PLANT,
                            if day >= 22 { 105.0 } else { 104.0 },
                            crop,
                            -1,
                            0,
                            ROLE_FIELD,
                        );
                    }
                }
                planned_seed_use[crop_index] += selected_count;
            }

            let mut carried_cow = 0_i64;
            let mut carried_sheep = 0_i64;
            let mut needs_deposit = false;
            let mut premium_carried = false;
            for unit in 0..max_units {
                let mut unit_total = 0_i64;
                for item in 0..ITEM_COUNT {
                    let amount = rounded_i64(
                        units[[environment, player, 0, unit, UNIT_INVENTORY_START + item]],
                        shed_scale,
                    );
                    unit_total += amount;
                    if item == ITEM_COW as usize {
                        carried_cow += amount;
                    } else if item == ITEM_SHEEP as usize {
                        carried_sheep += amount;
                    }
                    if matches!(item, 3 | 4 | ITEM_MILK | ITEM_WOOL) && amount > 0 {
                        premium_carried = true;
                    }
                }
                if active_units[[environment, player, unit]] && unit_total > 0 {
                    needs_deposit = true;
                }
            }

            let mut claimed_pasture = vec![false; tile_slots];
            for (animal, animal_index, carried) in [
                (ITEM_COW, 1_usize, carried_cow),
                (ITEM_SHEEP, 2_usize, carried_sheep),
            ] {
                let deficit = (target_animal_counts[[environment, player, animal_index]]
                    - animal_counts[animal_index])
                    .max(0);
                let place_count = carried.min(deficit);
                let candidates = empty_pasture
                    .iter()
                    .zip(&claimed_pasture)
                    .map(|(&is_empty, &is_claimed)| is_empty && !is_claimed)
                    .collect::<Vec<_>>();
                let place = select_limited(&candidates, place_count);
                for (slot, &is_selected) in place.iter().enumerate() {
                    if is_selected {
                        claimed_pasture[slot] = true;
                        propose_tile(
                            &mut out,
                            seat_base + slot,
                            TASK_PLACE_ANIMAL,
                            145.0,
                            animal,
                            animal,
                            1,
                            ROLE_LIVESTOCK,
                        );
                    }
                }
            }

            let shed_cow = rounded_i64(
                private[[environment, player, ITEM_COW as usize]],
                shed_scale,
            );
            let shed_sheep = rounded_i64(
                private[[environment, player, ITEM_SHEEP as usize]],
                shed_scale,
            );
            let cow_deficit = (target_animal_counts[[environment, player, 1]]
                - animal_counts[1])
                .max(0);
            let sheep_deficit = (target_animal_counts[[environment, player, 2]]
                - animal_counts[2])
                .max(0);
            let cow_needed = cow_deficit > 0;
            let sheep_needed = sheep_deficit > 0;
            let fetch_cow = cow_needed && shed_cow > 0 && carried_cow == 0;
            let fetch_sheep = !fetch_cow && sheep_needed && shed_sheep > 0 && carried_sheep == 0;
            let fetch_animal = productive && (fetch_cow || fetch_sheep);
            let fetch_item = if fetch_cow { ITEM_COW } else { ITEM_SHEEP };
            let fetch_quantity = if fetch_cow {
                shed_cow.min(cow_deficit)
            } else {
                shed_sheep.min(sheep_deficit)
            };
            set_global(
                &mut out,
                seat_base + tile_slots + 2,
                fetch_animal,
                TASK_FETCH_ITEM,
                access,
                access,
                150.0,
                fetch_item,
                fetch_quantity,
                true,
                ROLE_LOGISTICS,
            );

            let deposit_priority = if premium_carried {
                if day == 6 { 145.0 } else { 112.0 }
            } else {
                60.0
            };
            set_global(
                &mut out,
                seat_base + tile_slots + 1,
                needs_deposit,
                TASK_DEPOSIT_INVENTORY,
                access,
                access,
                deposit_priority,
                -1,
                1,
                false,
                ROLE_LOGISTICS,
            );
        }
    }

    Ok(())
}
