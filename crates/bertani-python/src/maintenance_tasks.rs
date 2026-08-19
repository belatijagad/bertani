//! Native maintenance-task generation for the rule policy.
//!
//! The strategic layer still decides what the farm should become. This module
//! only translates the current encoded farm state into deterministic survival,
//! harvest, care, fertilizer, and shed-fetch tasks. Task-slot arbitration is
//! intentionally identical to Python `TaskBatch.propose_tiles`: a proposal
//! replaces the current tile task only when its priority is strictly greater.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use numpy::{PyArray2, PyArray3, PyArray4, PyArray5, PyArray6, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const ITEM_COUNT: usize = 12;
const ITEM_WHEAT: i16 = 0;
const ITEM_CARROT: usize = 1;
const ITEM_TOMATO: usize = 2;
const ITEM_STRAWBERRY: usize = 3;
const ITEM_MELON: usize = 4;
const ITEM_FERTILIZER: i16 = 8;

const TASK_NONE: i16 = 0;
const TASK_WATER: i16 = 1;
const TASK_FEED: i16 = 2;
const TASK_CARE: i16 = 3;
const TASK_HARVEST: i16 = 4;
const TASK_COLLECT_FERTILIZER: i16 = 5;
const TASK_FERTILIZE: i16 = 8;
const TASK_FETCH_ITEM: i16 = 12;

// Routing-only stage: active task with NONE kind.
const TASK_STAGE: i16 = TASK_NONE;
const MARKET_BUY_PRODUCT: i64 = 4;

const ROLE_ANY: i16 = 0;
const ROLE_LOGISTICS: i16 = 1;
const ROLE_LIVESTOCK: i16 = 2;
const ROLE_FIELD: i16 = 3;

const TILE_PLANT: usize = 3;
const TILE_ANIMAL_START: usize = 6;
const TILE_ANIMAL_END: usize = 9;
const TILE_CROP_START: usize = 9;
const TILE_CROP_AGE: usize = 14;
const TILE_WATERED_OR_FED: usize = 15;
const TILE_CARED: usize = 16;
const TILE_CONSECUTIVE_MISSED: usize = 17;
const TILE_FERTILIZER_DAYS: usize = 19;
const TILE_FERTILIZER_AVAILABLE: usize = 20;
const TILE_HARVESTABLE: usize = 23;

const UNIT_INVENTORY_START: usize = 5;

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

#[allow(clippy::too_many_arguments)]
#[inline]
fn propose_tile(
    out: &mut TaskOutput<'_>,
    index: usize,
    kind: i16,
    priority: f32,
    deadline: i16,
    estimated_value: f32,
    required_item: i16,
    required_count: i64,
    work_role: i16,
) {
    if priority <= out.priority[index] {
        return;
    }
    out.active[index] = true;
    out.kind[index] = kind;
    out.item[index] = -1;
    out.quantity[index] = 1;
    out.priority[index] = priority;
    out.deadline[index] = deadline;
    out.estimated_value[index] = estimated_value;
    out.required_item[index] = required_item;
    out.required_count[index] = required_count;
    out.exclusive[index] = true;
    out.work_role[index] = work_role;
}

#[allow(clippy::too_many_arguments)]
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
    deadline: i16,
    work_role: i16,
) {
    out.active[index] = active;
    out.kind[index] = if active { kind } else { TASK_NONE };
    out.target_x[index] = target_x;
    out.target_y[index] = target_y;
    out.item[index] = item;
    out.quantity[index] = quantity;
    out.priority[index] = if active { priority } else { f32::NEG_INFINITY };
    out.deadline[index] = deadline;
    out.estimated_value[index] = 0.0;
    out.required_item[index] = -1;
    out.required_count[index] = 0;
    out.exclusive[index] = true;
    out.work_role[index] = if active { work_role } else { ROLE_ANY };
}

#[inline]
fn rounded_i64(value: f32, scale: f32) -> i64 {
    // Encoded discrete values are exact integer/scale ratios. They never rely
    // on half-way tie behaviour, so Rust round() matches NumPy rint() here.
    (value * scale).round() as i64
}

#[pyfunction]
#[allow(
    clippy::needless_pass_by_value,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
pub(crate) fn propose_maintenance_tasks<'py>(
    tiles: Bound<'py, PyArray6<f32>>,
    global_features: Bound<'py, PyArray3<f32>>,
    units: Bound<'py, PyArray5<f32>>,
    private: Bound<'py, PyArray3<f32>>,
    active_units: Bound<'py, PyArray3<bool>>,
    seat_mask: Bound<'py, PyArray2<bool>>,
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
    if board_size == 0 || turns_per_day <= 0 || shed_capacity <= 0 || episode_steps <= 0 {
        return Err(PyValueError::new_err(
            "maintenance configuration must be positive",
        ));
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
        return Err(PyValueError::new_err(
            "global_features batch shape does not match tiles",
        ));
    }
    let unit_shape = units.shape();
    if unit_shape.len() != 5
        || unit_shape[0] != num_envs
        || unit_shape[1] != players
        || unit_shape[2] < 1
        || unit_shape[4] < UNIT_INVENTORY_START + ITEM_COUNT
    {
        return Err(PyValueError::new_err(
            "units shape is incompatible with maintenance tasks",
        ));
    }
    let max_units = unit_shape[3];
    if active_units.shape() != [num_envs, players, max_units] {
        return Err(PyValueError::new_err(
            "active_units shape does not match units",
        ));
    }
    if seat_mask.shape() != [num_envs, players] {
        return Err(PyValueError::new_err("seat_mask must have shape [N, P]"));
    }
    let market_shape = market_actions.shape();
    if market_shape.len() != 4
        || market_shape[0] != num_envs
        || market_shape[1] != players
        || market_shape[2] < 1
        || market_shape[3] != 3
    {
        return Err(PyValueError::new_err(
            "market_actions must have shape [N, P, O, 3]",
        ));
    }
    let max_market_orders = market_shape[2];
    if market_lengths.shape() != [num_envs, players] {
        return Err(PyValueError::new_err(
            "market_lengths must have shape [N, P]",
        ));
    }
    let private_shape = private.shape();
    if private_shape.len() != 3
        || private_shape[0] != num_envs
        || private_shape[1] != players
        || private_shape[2] < ITEM_COUNT
    {
        return Err(PyValueError::new_err(
            "private shape is incompatible with maintenance tasks",
        ));
    }

    let output_shape = task_active.shape();
    if output_shape.len() != 3 || output_shape[0] != num_envs || output_shape[1] != players {
        return Err(PyValueError::new_err(
            "task batch shape does not match observations",
        ));
    }
    let capacity = output_shape[2];
    if capacity <= tile_slots + 6 {
        return Err(PyValueError::new_err(
            "task batch needs at least seven global slots for maintenance",
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
        ("market_actions", market_actions.is_c_contiguous()),
        ("market_lengths", market_lengths.is_c_contiguous()),
        ("task_active", task_active.is_c_contiguous()),
        ("task_kind", task_kind.is_c_contiguous()),
        ("task_target_x", task_target_x.is_c_contiguous()),
        ("task_target_y", task_target_y.is_c_contiguous()),
        ("task_item", task_item.is_c_contiguous()),
        ("task_quantity", task_quantity.is_c_contiguous()),
        ("task_priority", task_priority.is_c_contiguous()),
        ("task_deadline", task_deadline.is_c_contiguous()),
        (
            "task_estimated_value",
            task_estimated_value.is_c_contiguous(),
        ),
        ("task_required_item", task_required_item.is_c_contiguous()),
        ("task_required_count", task_required_count.is_c_contiguous()),
        ("task_exclusive", task_exclusive.is_c_contiguous()),
        ("task_work_role", task_work_role.is_c_contiguous()),
    ] {
        if !contiguous {
            return Err(PyValueError::new_err(format!(
                "{name} must be C-contiguous"
            )));
        }
    }

    let tiles_guard = tiles.try_readonly()?;
    let global_guard = global_features.try_readonly()?;
    let units_guard = units.try_readonly()?;
    let private_guard = private.try_readonly()?;
    let active_units_guard = active_units.try_readonly()?;
    let seat_mask_guard = seat_mask.try_readonly()?;
    let market_actions_guard = market_actions.try_readonly()?;
    let market_lengths_guard = market_lengths.try_readonly()?;
    let tiles = tiles_guard.as_array();
    let global_features = global_guard.as_array();
    let units = units_guard.as_array();
    let private = private_guard.as_array();
    let active_units = active_units_guard.as_array();
    let seat_mask = seat_mask_guard.as_array();
    let market_actions = market_actions_guard.as_array();
    let market_lengths = market_lengths_guard.as_array();

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
    let deadline = i16::try_from(turns_per_day - 1)
        .map_err(|_| PyValueError::new_err("turns_per_day does not fit task deadline"))?;
    let access = i16::try_from((board_size / 2).saturating_sub(1))
        .map_err(|_| PyValueError::new_err("board size does not fit task coordinates"))?;

    for environment in 0..num_envs {
        for player in 0..players {
            if !seat_mask[[environment, player]] {
                continue;
            }
            let step = rounded_i64(global_features[[environment, player, 0]], last_step);
            let day = step / i64::from(turns_per_day);
            let hour = step % i64::from(turns_per_day);
            let seat_base = (environment * players + player) * capacity;
            let requested_orders = market_lengths[[environment, player]];
            let Ok(requested_orders) = usize::try_from(requested_orders) else {
                return Err(PyValueError::new_err("market length cannot be negative"));
            };
            if requested_orders > max_market_orders {
                return Err(PyValueError::new_err(
                    "market length is outside the market action capacity",
                ));
            }
            let mut pending_wheat_buy = 0_i64;
            for order in 0..requested_orders {
                if market_actions[[environment, player, order, 0]] == MARKET_BUY_PRODUCT
                    && market_actions[[environment, player, order, 1]]
                        == i64::from(ITEM_WHEAT)
                {
                    pending_wheat_buy +=
                        market_actions[[environment, player, order, 2]].max(0);
                }
            }

            let mut needs_feed_count = 0_i64;
            let mut feed_distances = Vec::<i64>::new();
            let mut needs_fertilizer_count = 0_i64;
            let mut maximum_feed_priority = f32::NEG_INFINITY;
            let mut maximum_fertilizer_priority = f32::NEG_INFINITY;

            for y in 0..board_size {
                for x in 0..board_size {
                    let slot = y * board_size + x;
                    let output_index = seat_base + slot;
                    let tile = |channel: usize| tiles[[environment, player, 0, y, x, channel]];

                    let plants = tile(TILE_PLANT) > 0.5;
                    let animals =
                        (TILE_ANIMAL_START..TILE_ANIMAL_END).any(|channel| tile(channel) > 0.5);
                    let watered_or_fed = tile(TILE_WATERED_OR_FED) > 0.5;
                    let cared = tile(TILE_CARED) > 0.5;
                    let consecutive_missed = tile(TILE_CONSECUTIVE_MISSED) * 2.0;
                    let harvestable = tile(TILE_HARVESTABLE) > 0.5;
                    let crop_age = rounded_i64(tile(TILE_CROP_AGE), episode_days as f32);

                    let crop = |item: usize| tile(TILE_CROP_START + item) > 0.5;
                    let one_time_ready = (crop(ITEM_WHEAT as usize) && crop_age >= 4)
                        || (crop(ITEM_CARROT) && crop_age >= 3)
                        || (crop(ITEM_MELON) && crop_age >= 10);
                    let ongoing = crop(ITEM_TOMATO) || crop(ITEM_STRAWBERRY);
                    let harvest_now = harvestable && (one_time_ready || ongoing || animals);

                    if animals && !cared {
                        propose_tile(
                            &mut out,
                            output_index,
                            TASK_CARE,
                            if watered_or_fed {
                                118.0
                            } else {
                                100.0 + 5.0 * consecutive_missed
                            },
                            deadline,
                            0.0,
                            -1,
                            0,
                            ROLE_LIVESTOCK,
                        );
                    }
                    if animals && tile(TILE_FERTILIZER_AVAILABLE) > 0.5 {
                        propose_tile(
                            &mut out,
                            output_index,
                            TASK_COLLECT_FERTILIZER,
                            95.0,
                            -1,
                            100.0,
                            -1,
                            0,
                            ROLE_LIVESTOCK,
                        );
                    }

                    let wheat_bonus = crop(ITEM_WHEAT as usize) && (2..=4).contains(&crop_age);
                    let carrot_bonus = crop(ITEM_CARROT) && (2..=3).contains(&crop_age);
                    let melon_bonus = crop(ITEM_MELON) && (6..=10).contains(&crop_age);
                    let tomato_production = crop(ITEM_TOMATO) && (8..=11).contains(&crop_age);
                    let strawberry_production = crop(ITEM_STRAWBERRY)
                        && (10..=16).contains(&crop_age)
                        && ((crop_age - 10) % 2 == 0);
                    let yield_water = wheat_bonus
                        || carrot_bonus
                        || melon_bonus
                        || tomato_production
                        || strawberry_production;
                    let needs_water =
                        plants && !watered_or_fed && (consecutive_missed >= 1.0 || yield_water);

                    let one_time_final_water =
                        one_time_ready && (wheat_bonus || carrot_bonus || melon_bonus);

                    let normal_water_priority = 105.0 + 20.0 * consecutive_missed;

                    let water_priority = if one_time_final_water {
                        normal_water_priority.max(115.0)
                    } else {
                        normal_water_priority
                    };

                    if needs_water {
                        propose_tile(
                            &mut out,
                            output_index,
                            TASK_WATER,
                            water_priority,
                            deadline,
                            0.0,
                            -1,
                            0,
                            ROLE_FIELD,
                        );
                    }

                    let fertilizer_days = tile(TILE_FERTILIZER_DAYS) * 3.0;
                    // Fertilizer is only valuable to ongoing crops on a scheduled
                    // production day.  Applying it earlier wastes part or all of the
                    // three-day window; e.g. Strawberry should naturally synchronize
                    // around ages 10/12 and 14/16 rather than being fertilized every
                    // three days from planting.
                    let ongoing_production_today = tomato_production || strawberry_production;
                    // The production rule retires an ongoing crop once its final
                    // scheduled yield has been consumed. Do not let a late
                    // maintenance proposal override that CLEAR_WEED task.
                    let exhausted_ongoing = !harvestable
                        && ((crop(ITEM_TOMATO) && crop_age >= 11)
                            || (crop(ITEM_STRAWBERRY) && crop_age >= 16));
                    let ongoing_needing_fertilizer = ongoing_production_today
                        && !harvestable
                        && !exhausted_ongoing
                        && fertilizer_days < 1.0;
                    if ongoing_needing_fertilizer {
                        needs_fertilizer_count += 1;
                        // WATER must happen first when both are outstanding.  Once the
                        // crop is watered, make FERTILIZE more urgent than routine
                        // HARVEST so the end-of-day production receives the bonus.
                        let fertilizer_priority = if watered_or_fed { 115.0 } else { 104.0 };
                        maximum_fertilizer_priority =
                            maximum_fertilizer_priority.max(fertilizer_priority);
                        propose_tile(
                            &mut out,
                            output_index,
                            TASK_FERTILIZE,
                            fertilizer_priority,
                            deadline,
                            0.0,
                            ITEM_FERTILIZER,
                            1,
                            ROLE_FIELD,
                        );
                    }

                    if harvest_now && plants {
                        propose_tile(
                            &mut out,
                            output_index,
                            TASK_HARVEST,
                            110.0,
                            -1,
                            0.0,
                            -1,
                            0,
                            ROLE_FIELD,
                        );
                    }
                    if harvest_now && animals {
                        propose_tile(
                            &mut out,
                            output_index,
                            TASK_HARVEST,
                            if day == 6 { 145.0 } else { 110.0 },
                            -1,
                            0.0,
                            -1,
                            0,
                            ROLE_LIVESTOCK,
                        );
                    }

                    let feed_priority = 120.0 + 30.0 * consecutive_missed;
                    maximum_feed_priority = maximum_feed_priority.max(feed_priority);
                    let needs_feed = animals && !watered_or_fed;
                    if needs_feed {
                        needs_feed_count += 1;
                        let shed_access = usize::try_from(access).unwrap_or(0);
                        let distance = x.abs_diff(shed_access) + y.abs_diff(shed_access);
                        feed_distances.push(i64::try_from(distance).unwrap_or(i64::MAX));
                        propose_tile(
                            &mut out,
                            output_index,
                            TASK_FEED,
                            feed_priority,
                            deadline,
                            0.0,
                            ITEM_WHEAT,
                            1,
                            ROLE_LIVESTOCK,
                        );
                    }
                }
            }

            let mut carried_wheat = 0_i64;
            let mut carried_fertilizer = 0_i64;
            for unit in 0..max_units {
                if !active_units[[environment, player, unit]] {
                    continue;
                }
                carried_wheat += rounded_i64(
                    units[[environment, player, 0, unit, UNIT_INVENTORY_START]],
                    shed_scale,
                );
                carried_fertilizer += rounded_i64(
                    units[[
                        environment,
                        player,
                        0,
                        unit,
                        UNIT_INVENTORY_START + ITEM_FERTILIZER as usize,
                    ]],
                    shed_scale,
                );
            }
            let available_wheat = rounded_i64(
                private[[environment, player, ITEM_WHEAT as usize]],
                shed_scale,
            );
            let available_fertilizer = rounded_i64(
                private[[environment, player, ITEM_FERTILIZER as usize]],
                shed_scale,
            );

            let wheat_missing = (needs_feed_count - carried_wheat).max(0);
            let total_wheat_fetch = wheat_missing.min(available_wheat);
            let wheat_quotient = total_wheat_fetch / 2;
            let wheat_remainder = total_wheat_fetch % 2;
            let feed_fetch_priority = maximum_feed_priority + 1.0;
            for (index, extra_slot) in [0_usize, 3_usize].into_iter().enumerate() {
                let quantity = wheat_quotient + i64::from(index == 0 && wheat_remainder == 1);
                set_global(
                    &mut out,
                    seat_base + tile_slots + extra_slot,
                    quantity > 0,
                    TASK_FETCH_ITEM,
                    access,
                    access,
                    feed_fetch_priority,
                    ITEM_WHEAT,
                    quantity,
                    deadline,
                    ROLE_LOGISTICS,
                );
            }

            // Market orders resolve after unit actions. If today's plan buys
            // Wheat but none is currently fetchable, keep the feed workflow
            // alive by pre-positioning workers at the shed in the *same* slots
            // that become FETCH_ITEM once the next observation confirms stock.
            //
            // Work backwards from FEED@deadline:
            //   reach shed at S -> next-turn PICKUP -> d moves -> FEED
            // therefore S <= deadline - d - 2.
            let pending_feed_wheat = (wheat_missing - total_wheat_fetch)
                .max(0)
                .min(pending_wheat_buy);
            if total_wheat_fetch == 0 && pending_feed_wheat > 0 {
                feed_distances.sort_unstable();
                let feasible_count = usize::try_from(pending_feed_wheat)
                    .unwrap_or(0)
                    .min(feed_distances.len());
                let pending_distances = &feed_distances[..feasible_count];

                let stage_quantities = [
                    (pending_feed_wheat + 1) / 2,
                    pending_feed_wheat / 2,
                ];
                let mut distance_cursor = 0_usize;

                for (index, extra_slot) in [0_usize, 3_usize].into_iter().enumerate() {
                    let requested_quantity = stage_quantities[index];
                    if requested_quantity <= 0 {
                        continue;
                    }

                    let requested = usize::try_from(requested_quantity).unwrap_or(0);
                    let end = distance_cursor
                        .saturating_add(requested)
                        .min(pending_distances.len());
                    if end <= distance_cursor {
                        continue;
                    }
                    let max_feed_distance = pending_distances[distance_cursor..end]
                        .iter()
                        .copied()
                        .max()
                        .unwrap_or(0);
                    distance_cursor = end;

                    let latest_stage_hour =
                        i64::from(deadline) - max_feed_distance - 2;
                    let stage_active = latest_stage_hour >= hour;
                    let stage_deadline = i16::try_from(latest_stage_hour)
                        .unwrap_or(-1);

                    set_global(
                        &mut out,
                        seat_base + tile_slots + extra_slot,
                        stage_active,
                        TASK_STAGE,
                        access,
                        access,
                        feed_fetch_priority,
                        ITEM_WHEAT,
                        requested_quantity,
                        stage_deadline,
                        ROLE_LOGISTICS,
                    );
                }
            }

            let fertilizer_missing = (needs_fertilizer_count - carried_fertilizer).max(0);
            let total_fertilizer_fetch = fertilizer_missing.min(available_fertilizer);
            set_global(
                &mut out,
                seat_base + tile_slots + 6,
                total_fertilizer_fetch > 0,
                TASK_FETCH_ITEM,
                access,
                access,
                maximum_fertilizer_priority + 1.0,
                ITEM_FERTILIZER,
                total_fertilizer_fetch,
                -1,
                ROLE_LOGISTICS,
            );
        }
    }

    Ok(())
}
