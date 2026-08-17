//! Native extraction of dense rule-policy features from encoded observations.
//!
//! The strategic intent logic stays in Python. This module only removes the
//! repeated NumPy reductions/casts needed to reconstruct exact integer state.

#![allow(clippy::all, clippy::pedantic)]

use numpy::{
    PyArray2, PyArray3, PyArray4, PyArray6, PyArrayMethods, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const CROP_COUNT: usize = 5;
const ANIMAL_COUNT: usize = 3;
const ITEM_COUNT: usize = 12;
const SEED_COUNT: usize = 5;
const SHOP_COUNT: usize = 8;

const TILE_ANIMAL_START: usize = 6;
const TILE_CROP_START: usize = 9;

#[inline]
fn rounded_i64(value: f32, scale: f32) -> i64 {
    (value * scale).round_ties_even() as i64
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn extract_rule_features<'py>(
    global_features: Bound<'py, PyArray3<f32>>,
    farms: Bound<'py, PyArray4<f32>>,
    tiles: Bound<'py, PyArray6<f32>>,
    private: Bound<'py, PyArray3<f32>>,
    out_step: Bound<'py, PyArray2<i64>>,
    out_day: Bound<'py, PyArray2<i64>>,
    out_hour: Bound<'py, PyArray2<i64>>,
    out_money: Bound<'py, PyArray2<f64>>,
    out_crop_counts: Bound<'py, PyArray3<i64>>,
    out_animal_counts: Bound<'py, PyArray3<i64>>,
    out_shed: Bound<'py, PyArray3<i64>>,
    out_seeds: Bound<'py, PyArray3<i64>>,
    out_shop_counts: Bound<'py, PyArray3<i64>>,
    out_opponent_crop_counts: Bound<'py, PyArray3<i64>>,
    out_market_price_ratios: Bound<'py, PyArray3<f32>>,
    episode_steps: i64,
    turns_per_day: i64,
    starting_money: i64,
    shed_capacity: i64,
) -> PyResult<()> {
    if episode_steps <= 0 || turns_per_day <= 0 || starting_money <= 0 || shed_capacity <= 0 {
        return Err(PyValueError::new_err("rule feature configuration must be positive"));
    }

    let global_shape = global_features.shape();
    if global_shape.len() != 3 || global_shape[2] < 30 {
        return Err(PyValueError::new_err("global_features has incompatible shape"));
    }
    let num_envs = global_shape[0];
    let players = global_shape[1];

    let farm_shape = farms.shape();
    if farm_shape.len() != 4
        || farm_shape[0] != num_envs
        || farm_shape[1] != players
        || farm_shape[2] < 1
        || farm_shape[3] < 1
    {
        return Err(PyValueError::new_err("farms has incompatible shape"));
    }

    let tile_shape = tiles.shape();
    if tile_shape.len() != 6
        || tile_shape[0] != num_envs
        || tile_shape[1] != players
        || tile_shape[2] < 2
        || tile_shape[5] < TILE_CROP_START + CROP_COUNT
    {
        return Err(PyValueError::new_err("tiles has incompatible shape"));
    }
    let board_h = tile_shape[3];
    let board_w = tile_shape[4];

    let private_shape = private.shape();
    if private_shape.len() != 3
        || private_shape[0] != num_envs
        || private_shape[1] != players
        || private_shape[2] < ITEM_COUNT + SEED_COUNT
    {
        return Err(PyValueError::new_err("private has incompatible shape"));
    }

    let seat_shape = [num_envs, players];
    if out_step.shape() != seat_shape
        || out_day.shape() != seat_shape
        || out_hour.shape() != seat_shape
        || out_money.shape() != seat_shape
    {
        return Err(PyValueError::new_err("scalar rule feature outputs have incompatible shape"));
    }
    if out_crop_counts.shape() != [num_envs, players, CROP_COUNT]
        || out_animal_counts.shape() != [num_envs, players, ANIMAL_COUNT]
        || out_shed.shape() != [num_envs, players, ITEM_COUNT]
        || out_seeds.shape() != [num_envs, players, SEED_COUNT]
        || out_shop_counts.shape() != [num_envs, players, SHOP_COUNT]
        || out_opponent_crop_counts.shape() != [num_envs, players, CROP_COUNT]
        || out_market_price_ratios.shape() != [num_envs, players, 9]
    {
        return Err(PyValueError::new_err("vector rule feature outputs have incompatible shape"));
    }

    let global_guard = global_features.readonly();
    let farms_guard = farms.readonly();
    let tiles_guard = tiles.readonly();
    let private_guard = private.readonly();
    let global = global_guard.as_array();
    let farms = farms_guard.as_array();
    let tiles = tiles_guard.as_array();
    let private = private_guard.as_array();

    let mut step_guard = out_step.try_readwrite()?;
    let mut day_guard = out_day.try_readwrite()?;
    let mut hour_guard = out_hour.try_readwrite()?;
    let mut money_guard = out_money.try_readwrite()?;
    let mut crop_guard = out_crop_counts.try_readwrite()?;
    let mut animal_guard = out_animal_counts.try_readwrite()?;
    let mut shed_guard = out_shed.try_readwrite()?;
    let mut seeds_guard = out_seeds.try_readwrite()?;
    let mut shops_guard = out_shop_counts.try_readwrite()?;
    let mut opponent_crop_guard = out_opponent_crop_counts.try_readwrite()?;
    let mut price_ratio_guard = out_market_price_ratios.try_readwrite()?;

    let step_out = step_guard.as_slice_mut()?;
    let day_out = day_guard.as_slice_mut()?;
    let hour_out = hour_guard.as_slice_mut()?;
    let money_out = money_guard.as_slice_mut()?;
    let crop_out = crop_guard.as_slice_mut()?;
    let animal_out = animal_guard.as_slice_mut()?;
    let shed_out = shed_guard.as_slice_mut()?;
    let seeds_out = seeds_guard.as_slice_mut()?;
    let shops_out = shops_guard.as_slice_mut()?;
    let opponent_crop_out = opponent_crop_guard.as_slice_mut()?;
    let price_ratio_out = price_ratio_guard.as_slice_mut()?;

    let last_step = (episode_steps - 1).max(1);
    for environment in 0..num_envs {
        for player in 0..players {
            let seat = environment * players + player;
            let step = rounded_i64(global[[environment, player, 0]], last_step as f32);
            step_out[seat] = step;
            day_out[seat] = step / turns_per_day;
            hour_out[seat] = step % turns_per_day;
            money_out[seat] = f64::from(farms[[environment, player, 0, 0]])
                * starting_money as f64;

            for crop in 0..CROP_COUNT {
                let channel = TILE_CROP_START + crop;
                let mut own_sum = 0.0_f32;
                let mut opponent_sum = 0.0_f32;
                for y in 0..board_h {
                    for x in 0..board_w {
                        own_sum += tiles[[environment, player, 0, y, x, channel]];
                        opponent_sum += tiles[[environment, player, 1, y, x, channel]];
                    }
                }
                crop_out[seat * CROP_COUNT + crop] = own_sum.round_ties_even() as i64;
                opponent_crop_out[seat * CROP_COUNT + crop] =
                    opponent_sum.round_ties_even() as i64;
            }

            for animal in 0..ANIMAL_COUNT {
                let channel = TILE_ANIMAL_START + animal;
                let mut total = 0.0_f32;
                for y in 0..board_h {
                    for x in 0..board_w {
                        total += tiles[[environment, player, 0, y, x, channel]];
                    }
                }
                animal_out[seat * ANIMAL_COUNT + animal] = total.round_ties_even() as i64;
            }

            for item in 0..ITEM_COUNT {
                shed_out[seat * ITEM_COUNT + item] = rounded_i64(
                    private[[environment, player, item]],
                    shed_capacity as f32,
                );
            }
            for seed in 0..SEED_COUNT {
                seeds_out[seat * SEED_COUNT + seed] =
                    rounded_i64(private[[environment, player, ITEM_COUNT + seed]], 10.0);
            }
            for product in 0..9 {
                price_ratio_out[seat * 9 + product] =
                    global[[environment, player, 5 + product * 2]];
            }
            for shop in 0..SHOP_COUNT {
                shops_out[seat * SHOP_COUNT + shop] =
                    rounded_i64(global[[environment, player, 22 + shop]], 8.0);
            }
        }
    }
    Ok(())
}
