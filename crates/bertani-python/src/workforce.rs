//! Persistent native implementation of the current territorial workforce rule.
//!
//! This mirrors the Python `TerritorialWorkforcePlanner` mechanics exactly:
//! role assignment, day-persistent quadrant territories, inventory overrides,
//! and planting-capacity reservation. Strategic worker-count decisions remain
//! in Python `StrategicIntent`.

#![allow(clippy::all, clippy::pedantic)]

use numpy::{
    PyArray3, PyArray4, PyArray5, PyArray6, PyArrayMethods, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const ROLE_ANY: i16 = 0;
const ROLE_LOGISTICS: i16 = 1;
const ROLE_LIVESTOCK: i16 = 2;
const ROLE_FIELD: i16 = 3;

const ZONE_ANY: i16 = -1;
const QUADRANTS: usize = 4;
const TASK_KIND_COUNT: usize = 14;
const TASK_WATER: i16 = 1;
const TASK_FEED: i16 = 2;
const TASK_CARE: i16 = 3;
const TASK_HARVEST: i16 = 4;
const TASK_CLEAR_WEED: i16 = 6;
const TASK_PLANT: i16 = 7;

const ITEM_COUNT: usize = 12;
const ITEM_COW: usize = 10;
const ITEM_SHEEP: usize = 11;
const TILE_ANIMAL_START: usize = 6;
const UNIT_INVENTORY_START: usize = 5;

#[pyclass]
pub(crate) struct NativeWorkforcePlanner {
    shed_capacity: i64,
    turns_per_day: i64,
    last_step: i64,
    daily_zone: Vec<i16>,
    last_day: Vec<i64>,
    shape: Option<(usize, usize, usize)>,
}

#[pymethods]
impl NativeWorkforcePlanner {
    #[new]
    fn new(shed_capacity: i64, turns_per_day: i64, episode_steps: i64) -> PyResult<Self> {
        if shed_capacity <= 0 || turns_per_day <= 0 || episode_steps <= 0 {
            return Err(PyValueError::new_err("workforce configuration must be positive"));
        }
        Ok(Self {
            shed_capacity,
            turns_per_day,
            last_step: (episode_steps - 1).max(1),
            daily_zone: Vec::new(),
            last_day: Vec::new(),
            shape: None,
        })
    }

    #[allow(clippy::needless_pass_by_value)]
    fn plan<'py>(
        &mut self,
        global_features: Bound<'py, PyArray3<f32>>,
        farms: Bound<'py, PyArray4<f32>>,
        tiles: Bound<'py, PyArray6<f32>>,
        units: Bound<'py, PyArray5<f32>>,
        active_units: Bound<'py, PyArray3<bool>>,
        task_active: Bound<'py, PyArray3<bool>>,
        task_kind: Bound<'py, PyArray3<i16>>,
        task_target_x: Bound<'py, PyArray3<i16>>,
        task_target_y: Bound<'py, PyArray3<i16>>,
        task_work_role: Bound<'py, PyArray3<i16>>,
        out_role: Bound<'py, PyArray3<i16>>,
        out_zone: Bound<'py, PyArray3<i16>>,
        out_reserved_by_kind: Bound<'py, PyArray3<i16>>,
        board_size: usize,
    ) -> PyResult<()> {
        if board_size == 0 {
            return Err(PyValueError::new_err("board_size must be positive"));
        }
        let active_shape = active_units.shape();
        if active_shape.len() != 3 {
            return Err(PyValueError::new_err("active_units must have shape [N, P, U]"));
        }
        let num_envs = active_shape[0];
        let players = active_shape[1];
        let max_units = active_shape[2];
        let shape = (num_envs, players, max_units);

        if out_role.shape() != [num_envs, players, max_units]
            || out_zone.shape() != [num_envs, players, max_units]
            || out_reserved_by_kind.shape() != [num_envs, players, TASK_KIND_COUNT]
        {
            return Err(PyValueError::new_err("workforce output shape mismatch"));
        }
        if global_features.shape().len() != 3
            || global_features.shape()[0] != num_envs
            || global_features.shape()[1] != players
        {
            return Err(PyValueError::new_err("global_features batch shape mismatch"));
        }
        let farm_shape = farms.shape();
        if farm_shape.len() != 4
            || farm_shape[0] != num_envs
            || farm_shape[1] != players
            || farm_shape[2] < 1
            || farm_shape[3] < 8
        {
            return Err(PyValueError::new_err("farms shape mismatch"));
        }
        let tile_shape = tiles.shape();
        if tile_shape.len() != 6
            || tile_shape[0] != num_envs
            || tile_shape[1] != players
            || tile_shape[2] < 1
            || tile_shape[3] != board_size
            || tile_shape[4] != board_size
            || tile_shape[5] < TILE_ANIMAL_START + 3
        {
            return Err(PyValueError::new_err("tiles shape mismatch"));
        }
        let unit_shape = units.shape();
        if unit_shape.len() != 5
            || unit_shape[0] != num_envs
            || unit_shape[1] != players
            || unit_shape[2] < 1
            || unit_shape[3] != max_units
            || unit_shape[4] < UNIT_INVENTORY_START + ITEM_COUNT
        {
            return Err(PyValueError::new_err("units shape mismatch"));
        }
        let task_shape = task_active.shape();
        if task_shape.len() != 3 || task_shape[0] != num_envs || task_shape[1] != players {
            return Err(PyValueError::new_err("task batch shape mismatch"));
        }
        let task_capacity = task_shape[2];
        if task_kind.shape() != [num_envs, players, task_capacity]
            || task_target_x.shape() != [num_envs, players, task_capacity]
            || task_target_y.shape() != [num_envs, players, task_capacity]
            || task_work_role.shape() != [num_envs, players, task_capacity]
        {
            return Err(PyValueError::new_err("task field shape mismatch"));
        }

        if self.shape != Some(shape) {
            self.daily_zone = vec![ZONE_ANY; num_envs * players * max_units];
            self.last_day = vec![-1; num_envs * players];
            self.shape = Some(shape);
        }

        let global_guard = global_features.readonly();
        let farms_guard = farms.readonly();
        let tiles_guard = tiles.readonly();
        let units_guard = units.readonly();
        let active_guard = active_units.readonly();
        let task_active_guard = task_active.readonly();
        let task_kind_guard = task_kind.readonly();
        let target_x_guard = task_target_x.readonly();
        let target_y_guard = task_target_y.readonly();
        let task_role_guard = task_work_role.readonly();

        let global = global_guard.as_array();
        let farms = farms_guard.as_array();
        let tiles = tiles_guard.as_array();
        let units = units_guard.as_array();
        let active = active_guard.as_array();
        let task_active = task_active_guard.as_array();
        let task_kind = task_kind_guard.as_array();
        let target_x = target_x_guard.as_array();
        let target_y = target_y_guard.as_array();
        let task_role = task_role_guard.as_array();

        let mut role_guard = out_role.try_readwrite()?;
        let mut zone_guard = out_zone.try_readwrite()?;
        let mut reserved_guard = out_reserved_by_kind.try_readwrite()?;
        let role = role_guard.as_slice_mut()?;
        let zone = zone_guard.as_slice_mut()?;
        let reserved = reserved_guard.as_slice_mut()?;
        role.fill(ROLE_ANY);
        zone.fill(ZONE_ANY);
        reserved.fill(0);

        let mut active_limit = 0usize;
        let mut active_slots = Vec::with_capacity(max_units);
        for worker in 0..max_units {
            let mut any = false;
            'seat_scan: for environment in 0..num_envs {
                for player in 0..players {
                    if active[[environment, player, worker]] {
                        any = true;
                        break 'seat_scan;
                    }
                }
            }
            if any {
                active_slots.push(worker);
                active_limit = worker + 1;
            }
        }

        let seat_count = num_envs * players;
        let tile_slots = board_size * board_size;
        let mut demand = vec![[0.0_f32; QUADRANTS]; seat_count];
        let mut effective_demand = vec![[0.0_f32; QUADRANTS]; seat_count];
        let mut assigned_count = vec![[0.0_f32; QUADRANTS]; seat_count];
        let mut livestock_workers = vec![0_i16; seat_count];
        let mut days = vec![0_i64; seat_count];
        let mut hours = vec![0_i64; seat_count];

        for environment in 0..num_envs {
            for player in 0..players {
                let seat = environment * players + player;
                let mut animal_total = 0_i16;
                for animal in 0..3 {
                    let mut animal_sum = 0.0_f32;
                    for y in 0..board_size {
                        for x in 0..board_size {
                            animal_sum += tiles[[
                                environment,
                                player,
                                0,
                                y,
                                x,
                                TILE_ANIMAL_START + animal,
                            ]];
                        }
                    }
                    animal_total += animal_sum.round_ties_even() as i16;
                }
                livestock_workers[seat] = if animal_total > 0 {
                    ((animal_total + 3) / 4).clamp(1, 3)
                } else {
                    0
                };

                let step = (global[[environment, player, 0]] * self.last_step as f32)
                    .round_ties_even() as i64;
                let day = step / self.turns_per_day;
                let hour = step % self.turns_per_day;
                days[seat] = day;
                hours[seat] = hour;

                if day != self.last_day[seat] {
                    let start = seat * max_units;
                    self.daily_zone[start..start + max_units].fill(ZONE_ANY);
                }
                for worker in 0..max_units {
                    if !active[[environment, player, worker]] {
                        self.daily_zone[seat * max_units + worker] = ZONE_ANY;
                    }
                }
                self.last_day[seat] = day;

                for task in 0..task_capacity {
                    if !task_active[[environment, player, task]]
                        || task >= tile_slots
                        || task_role[[environment, player, task]] == ROLE_LOGISTICS
                    {
                        continue;
                    }
                    let x = target_x[[environment, player, task]];
                    let y = target_y[[environment, player, task]];
                    if x < 0 || y < 0 {
                        continue;
                    }
                    let quadrant = (if y as usize >= board_size / 2 { 2 } else { 0 })
                        + if x as usize >= board_size / 2 { 1 } else { 0 };
                    let kind = task_kind[[environment, player, task]];
                    let mut weight = 1.0_f32;
                    if matches!(kind, TASK_WATER | TASK_FEED | TASK_CARE) {
                        weight += 1.0;
                    }
                    if matches!(kind, TASK_HARVEST | TASK_CLEAR_WEED | TASK_PLANT) {
                        weight += 0.5;
                    }
                    demand[seat][quadrant] += weight;
                }

                let mut total_demand = 0.0_f32;
                for quadrant in 0..QUADRANTS {
                    let unlocked = farms[[environment, player, 0, 4 + quadrant]]
                        .round_ties_even()
                        != 0.0;
                    if !unlocked {
                        demand[seat][quadrant] = 0.0;
                    }
                    total_demand += demand[seat][quadrant];
                }
                for quadrant in 0..QUADRANTS {
                    effective_demand[seat][quadrant] = if total_demand > 0.0 {
                        demand[seat][quadrant]
                    } else if farms[[environment, player, 0, 4 + quadrant]]
                        .round_ties_even()
                        != 0.0
                    {
                        1.0
                    } else {
                        0.0
                    };
                }

                for worker in 0..active_limit {
                    if !active[[environment, player, worker]] {
                        continue;
                    }
                    let out_index = seat * max_units + worker;
                    if worker == 0 {
                        role[out_index] = ROLE_LOGISTICS;
                    } else if (worker as i16) <= livestock_workers[seat] {
                        role[out_index] = ROLE_LIVESTOCK;
                    } else {
                        role[out_index] = ROLE_FIELD;
                    }
                }
            }
        }

        // Count already-persistent territories before assigning newly active
        // workers. Worker order is intentionally stable and matches Python.
        for environment in 0..num_envs {
            for player in 0..players {
                let seat = environment * players + player;
                for worker in 1..active_limit {
                    if !active[[environment, player, worker]] {
                        continue;
                    }
                    let existing = self.daily_zone[seat * max_units + worker];
                    if (0..4).contains(&existing) {
                        assigned_count[seat][existing as usize] += 1.0;
                    }
                }
            }
        }

        for &worker in &active_slots {
            if worker == 0 {
                continue;
            }
            for environment in 0..num_envs {
                for player in 0..players {
                    let seat = environment * players + player;
                    if !active[[environment, player, worker]] {
                        continue;
                    }
                    let cache_index = seat * max_units + worker;
                    if self.daily_zone[cache_index] != ZONE_ANY {
                        continue;
                    }
                    let mut best_quadrant = 0usize;
                    let mut best_pressure = effective_demand[seat][0]
                        / (assigned_count[seat][0] + 1.0);
                    for quadrant in 1..QUADRANTS {
                        let pressure = effective_demand[seat][quadrant]
                            / (assigned_count[seat][quadrant] + 1.0);
                        if pressure > best_pressure {
                            best_pressure = pressure;
                            best_quadrant = quadrant;
                        }
                    }
                    self.daily_zone[cache_index] = best_quadrant as i16;
                    assigned_count[seat][best_quadrant] += 1.0;
                }
            }
        }

        for environment in 0..num_envs {
            for player in 0..players {
                let seat = environment * players + player;
                let mut active_count = 0usize;
                for worker in 0..active_limit {
                    if !active[[environment, player, worker]] {
                        continue;
                    }
                    active_count += 1;
                    let out_index = seat * max_units + worker;
                    if worker != 0 {
                        zone[out_index] = self.daily_zone[out_index];
                    }

                    let mut inventory_total = 0_i64;
                    let mut animal_inventory_total = 0_i64;
                    for item in 0..ITEM_COUNT {
                        let count = (units[[
                            environment,
                            player,
                            0,
                            worker,
                            UNIT_INVENTORY_START + item,
                        ]] * self.shed_capacity as f32)
                            .round_ties_even() as i64;
                        inventory_total += count;
                        if (ITEM_COW..=ITEM_SHEEP).contains(&item) {
                            animal_inventory_total += count;
                        }
                    }
                    if inventory_total > 0 {
                        role[out_index] = ROLE_LOGISTICS;
                    }
                    if animal_inventory_total > 0 {
                        role[out_index] = ROLE_LIVESTOCK;
                    }
                }

                let mut plant_backlog = 0usize;
                for task in 0..task_capacity {
                    if task_active[[environment, player, task]]
                        && task_kind[[environment, player, task]] == TASK_PLANT
                    {
                        plant_backlog += 1;
                    }
                }
                if days[seat] >= 14
                    && hours[seat] < self.turns_per_day - 4
                    && plant_backlog >= 8
                    && active_count >= 8
                {
                    reserved[seat * TASK_KIND_COUNT + TASK_PLANT as usize] = 1;
                }
            }
        }

        Ok(())
    }
}
