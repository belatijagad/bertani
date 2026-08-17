//! Native implementation of the current rule-based market policy.
//!
//! This is intentionally a backend for the *rule policy*, not the strategic
//! interface used by a future learned policy. Python still supplies the
//! high-level StrategicIntent arrays; this module mirrors the existing
//! EconomyMarketRule exactly and writes directly into MarketPlanBatch buffers.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::manual_range_contains
)]

use numpy::{
    PyArray2, PyArray3, PyArray4, PyArray5, PyArray6, PyArrayMethods,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const ITEM_COUNT: usize = 12;
const PRODUCT_COUNT: usize = 9;
const CROP_COUNT: usize = 5;
const ANIMAL_TARGET_COUNT: usize = 3;

const ITEM_WHEAT: usize = 0;
const ITEM_CARROT: usize = 1;
const ITEM_TOMATO: usize = 2;
const ITEM_STRAWBERRY: usize = 3;
const ITEM_MELON: usize = 4;
const ITEM_EGG: usize = 5;
const ITEM_MILK: usize = 6;
const ITEM_WOOL: usize = 7;
const ITEM_FERTILIZER: usize = 8;
const ITEM_COW: usize = 10;
const ITEM_SHEEP: usize = 11;

const MARKET_HIRE: i64 = 1;
const MARKET_BUY_LAND: i64 = 2;
const MARKET_BUY_SEED: i64 = 3;
const MARKET_BUY_PRODUCT: i64 = 4;
const MARKET_BUY_ANIMAL: i64 = 5;
const MARKET_SELL: i64 = 6;

const TILE_ANIMAL_START: usize = 6;
const TILE_CROP_START: usize = 9;
const TILE_CROP_AGE: usize = 14;
const UNIT_INVENTORY_START: usize = 5;

const SHOP_DEMAND: [[i64; PRODUCT_COUNT]; 8] = [
    [1, 0, 0, 0, 0, 1, 0, 0, 0], // bakery
    [1, 0, 0, 1, 0, 1, 0, 0, 0], // brunch spot
    [1, 1, 1, 1, 0, 0, 0, 0, 0], // farmers market
    [1, 0, 0, 1, 0, 0, 1, 0, 0], // ice cream shop
    [0, 2, 0, 0, 0, 0, 0, 0, 0], // pet cafe
    [1, 0, 1, 0, 0, 0, 1, 0, 0], // pizza shop
    [0, 0, 0, 1, 0, 0, 1, 0, 0], // smoothie shop
    [0, 0, 0, 0, 0, 0, 0, 2, 0], // yarn store
];

const MARKET_BASE_PRICES: [f64; PRODUCT_COUNT] = [
    25.0, 35.0, 60.0, 120.0, 250.0, 50.0, 160.0, 200.0, 100.0,
];

const SALE_BATCHES: [i64; PRODUCT_COUNT] = [
    7, // wheat
    4, // carrot
    0, // tomato: accidental inventory uses full shed count
    8, // strawberry
    12, // melon
    0, // egg: accidental inventory uses full shed count
    6, // milk
    4, // wool
    18, // fertilizer
];

const SEED_BUY_BATCHES: [i64; CROP_COUNT] = [8, 4, 4, 4, 12];
const REPLACEMENT_AGES: [i64; CROP_COUNT] = [3, 2, 99, 99, 99];

#[inline]
fn rounded_i64(value: f32, scale: f32) -> i64 {
    // Encoded values are integer/scale ratios. round_ties_even mirrors
    // numpy.rint for completeness at the boundary.
    (value * scale).round_ties_even() as i64
}

#[inline]
fn rounded_price(ratio: f32, base: f64) -> i64 {
    ((f64::from(ratio) * base).round_ties_even() as i64).max(1)
}

struct MarketOutput<'a> {
    actions: &'a mut [i64],
    lengths: &'a mut [i64],
    overflow: &'a mut [bool],
    players: usize,
    max_orders: usize,
}

impl MarketOutput<'_> {
    #[inline]
    fn append(
        &mut self,
        environment: usize,
        player: usize,
        operation: i64,
        item: usize,
        count: i64,
    ) {
        let seat = environment * self.players + player;
        let Ok(slot) = usize::try_from(self.lengths[seat]) else {
            self.overflow[seat] = true;
            return;
        };
        if slot >= self.max_orders {
            self.overflow[seat] = true;
            return;
        }
        let base = (seat * self.max_orders + slot) * 3;
        self.actions[base] = operation;
        self.actions[base + 1] = item as i64;
        self.actions[base + 2] = count;
        self.lengths[seat] += 1;
    }
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value, clippy::similar_names)]
pub(crate) fn propose_rule_market<'py>(
    global_features: Bound<'py, PyArray3<f32>>,
    farms: Bound<'py, PyArray4<f32>>,
    tiles: Bound<'py, PyArray6<f32>>,
    units: Bound<'py, PyArray5<f32>>,
    private: Bound<'py, PyArray3<f32>>,
    active_units: Bound<'py, PyArray3<bool>>,
    target_hands: Bound<'py, PyArray2<i64>>,
    wheat_reserve: Bound<'py, PyArray2<i64>>,
    target_crop_counts: Bound<'py, PyArray3<i64>>,
    target_animal_counts: Bound<'py, PyArray3<i64>>,
    liquidate: Bound<'py, PyArray2<bool>>,
    market_actions: Bound<'py, PyArray4<i64>>,
    market_lengths: Bound<'py, PyArray2<i64>>,
    market_overflow: Bound<'py, PyArray2<bool>>,
    starting_money: i64,
    shed_capacity: i64,
    episode_steps: i64,
    turns_per_day: i64,
) -> PyResult<()> {
    if starting_money <= 0 || shed_capacity <= 0 || episode_steps <= 0 || turns_per_day <= 0 {
        return Err(PyValueError::new_err("market configuration must be positive"));
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
        || farm_shape[3] < 8
    {
        return Err(PyValueError::new_err("farms has incompatible shape"));
    }

    let tile_shape = tiles.shape();
    if tile_shape.len() != 6
        || tile_shape[0] != num_envs
        || tile_shape[1] != players
        || tile_shape[2] < 1
        || tile_shape[5] <= TILE_CROP_AGE
    {
        return Err(PyValueError::new_err("tiles has incompatible shape"));
    }
    let board_y = tile_shape[3];
    let board_x = tile_shape[4];

    let unit_shape = units.shape();
    if unit_shape.len() != 5
        || unit_shape[0] != num_envs
        || unit_shape[1] != players
        || unit_shape[2] < 1
        || unit_shape[4] < UNIT_INVENTORY_START + ITEM_COUNT
    {
        return Err(PyValueError::new_err("units has incompatible shape"));
    }
    let max_units = unit_shape[3];
    if active_units.shape() != [num_envs, players, max_units] {
        return Err(PyValueError::new_err("active_units shape does not match units"));
    }

    let private_shape = private.shape();
    if private_shape.len() != 3
        || private_shape[0] != num_envs
        || private_shape[1] != players
        || private_shape[2] < ITEM_COUNT + CROP_COUNT
    {
        return Err(PyValueError::new_err("private has incompatible shape"));
    }

    let seat_shape = [num_envs, players];
    if target_hands.shape() != seat_shape
        || wheat_reserve.shape() != seat_shape
        || liquidate.shape() != seat_shape
        || market_lengths.shape() != seat_shape
        || market_overflow.shape() != seat_shape
    {
        return Err(PyValueError::new_err("seat-level array shape mismatch"));
    }
    if target_crop_counts.shape() != [num_envs, players, CROP_COUNT]
        || target_animal_counts.shape() != [num_envs, players, ANIMAL_TARGET_COUNT]
    {
        return Err(PyValueError::new_err("intent target shape mismatch"));
    }

    let action_shape = market_actions.shape();
    if action_shape.len() != 4
        || action_shape[0] != num_envs
        || action_shape[1] != players
        || action_shape[3] != 3
    {
        return Err(PyValueError::new_err("market_actions has incompatible shape"));
    }
    let max_orders = action_shape[2];

    for (name, contiguous) in [
        ("market_actions", market_actions.is_c_contiguous()),
        ("market_lengths", market_lengths.is_c_contiguous()),
        ("market_overflow", market_overflow.is_c_contiguous()),
    ] {
        if !contiguous {
            return Err(PyValueError::new_err(format!("{name} must be C-contiguous")));
        }
    }

    let global_guard = global_features.try_readonly()?;
    let farms_guard = farms.try_readonly()?;
    let tiles_guard = tiles.try_readonly()?;
    let units_guard = units.try_readonly()?;
    let private_guard = private.try_readonly()?;
    let active_guard = active_units.try_readonly()?;
    let target_hands_guard = target_hands.try_readonly()?;
    let wheat_reserve_guard = wheat_reserve.try_readonly()?;
    let target_crop_guard = target_crop_counts.try_readonly()?;
    let target_animal_guard = target_animal_counts.try_readonly()?;
    let liquidate_guard = liquidate.try_readonly()?;

    let global = global_guard.as_array();
    let farms = farms_guard.as_array();
    let tiles = tiles_guard.as_array();
    let units = units_guard.as_array();
    let private = private_guard.as_array();
    let active_units = active_guard.as_array();
    let target_hands = target_hands_guard.as_array();
    let wheat_reserve = wheat_reserve_guard.as_array();
    let target_crop_counts = target_crop_guard.as_array();
    let target_animal_counts = target_animal_guard.as_array();
    let liquidate = liquidate_guard.as_array();

    let mut actions_guard = market_actions.try_readwrite()?;
    let mut lengths_guard = market_lengths.try_readwrite()?;
    let mut overflow_guard = market_overflow.try_readwrite()?;
    let mut out = MarketOutput {
        actions: actions_guard.as_slice_mut()?,
        lengths: lengths_guard.as_slice_mut()?,
        overflow: overflow_guard.as_slice_mut()?,
        players,
        max_orders,
    };

    let last_step = (episode_steps - 1).max(1) as f32;
    let episode_days = (episode_steps + turns_per_day - 1) / turns_per_day;
    let shed_scale = shed_capacity as f32;

    for environment in 0..num_envs {
        for player in 0..players {
            if liquidate[[environment, player]] {
                // The reference Python rule gates almost every market decision
                // behind `active = ~intent.liquidate`, but HIRE is intentionally
                // (historically) not active-gated. Preserve that exact behavior:
                // liquidation sales have already been appended by VectorRulePolicy,
                // then this rule may fill remaining order slots with daily hires.
                let mut active_count = 0_i64;
                for worker in 0..max_units {
                    active_count += i64::from(active_units[[environment, player, worker]]);
                }
                let hands = (active_count - 1).max(0);
                let wanted_hands = target_hands[[environment, player]];
                let missing_hands = (wanted_hands - hands).max(0);
                let money = farms[[environment, player, 0, 0]] * starting_money as f32;
                if wanted_hands > 0 && money >= 12.0 {
                    for _ in 0..missing_hands {
                        out.append(environment, player, MARKET_HIRE, 0, 0);
                    }
                }
                continue;
            }

            let mut shed = [0_i64; ITEM_COUNT];
            let mut seeds = [0_i64; CROP_COUNT];
            for item in 0..ITEM_COUNT {
                shed[item] = rounded_i64(private[[environment, player, item]], shed_scale);
            }
            for crop in 0..CROP_COUNT {
                seeds[crop] = rounded_i64(private[[environment, player, ITEM_COUNT + crop]], 10.0);
            }

            let mut crops = [0_i64; CROP_COUNT];
            let mut replacement_seeds = [0_i64; CROP_COUNT];
            let mut animal_counts = [0_i64; ANIMAL_TARGET_COUNT];
            let mut ongoing_count = 0_i64;

            for y in 0..board_y {
                for x in 0..board_x {
                    for animal in 0..ANIMAL_TARGET_COUNT {
                        if tiles[[environment, player, 0, y, x, TILE_ANIMAL_START + animal]] > 0.5 {
                            animal_counts[animal] += 1;
                        }
                    }
                    let crop_age = rounded_i64(
                        tiles[[environment, player, 0, y, x, TILE_CROP_AGE]],
                        episode_days as f32,
                    );
                    for crop in 0..CROP_COUNT {
                        if tiles[[environment, player, 0, y, x, TILE_CROP_START + crop]] > 0.5 {
                            crops[crop] += 1;
                            if crop_age >= REPLACEMENT_AGES[crop] {
                                replacement_seeds[crop] += 1;
                            }
                        }
                    }
                    if tiles[[environment, player, 0, y, x, TILE_CROP_START + ITEM_TOMATO]] > 0.5 {
                        ongoing_count += 1;
                    }
                    if tiles[[environment, player, 0, y, x, TILE_CROP_START + ITEM_STRAWBERRY]] > 0.5 {
                        ongoing_count += 1;
                    }
                }
            }

            let mut carried_wheat = 0_i64;
            let mut carried_cows = 0_i64;
            let mut carried_sheep = 0_i64;
            let mut active_count = 0_i64;
            for worker in 0..max_units {
                if !active_units[[environment, player, worker]] {
                    continue;
                }
                active_count += 1;
                carried_wheat += rounded_i64(
                    units[[environment, player, 0, worker, UNIT_INVENTORY_START + ITEM_WHEAT]],
                    shed_scale,
                );
                carried_cows += rounded_i64(
                    units[[environment, player, 0, worker, UNIT_INVENTORY_START + ITEM_COW]],
                    shed_scale,
                );
                carried_sheep += rounded_i64(
                    units[[environment, player, 0, worker, UNIT_INVENTORY_START + ITEM_SHEEP]],
                    shed_scale,
                );
            }

            // NumPy keeps this multiplication in float32; preserve that
            // rounding before later mixed int/float expressions promote to f64.
            let money_f32 = farms[[environment, player, 0, 0]] * starting_money as f32;
            let money = f64::from(money_f32);
            let hands = (active_count - 1).max(0);
            let shed_total: i64 = shed.iter().sum();
            let pressure = shed_total >= shed_capacity * 7 / 10;

            let step = rounded_i64(global[[environment, player, 0]], last_step);
            let day = step / turns_per_day;
            let post_town_demand = step % 4 == 1;
            let town_tick = step % 4 == 0;
            let town_center_tick = step % 24 == 0;

            let mut ratios = [0_f32; PRODUCT_COUNT];
            let mut current_prices = [0_i64; PRODUCT_COUNT];
            for item in 0..PRODUCT_COUNT {
                ratios[item] = global[[environment, player, 5 + item * 2]];
                current_prices[item] = rounded_price(ratios[item], MARKET_BASE_PRICES[item]);
            }
            let mut shops = [0_i64; 8];
            for shop in 0..8 {
                shops[shop] = rounded_i64(global[[environment, player, 22 + shop]], 8.0);
            }

            let mut unlocked_sum = 0_f32;
            for channel in 4..8 {
                unlocked_sum += farms[[environment, player, 0, channel]];
            }
            let unlocked = unlocked_sum.round_ties_even() as i64;

            let (next_land_cost, next_land_day) = match unlocked {
                1 => (1_000_i64, 7_i64),
                2 => (2_000_i64, 11_i64),
                _ => (0_i64, -1_i64),
            };
            let reserve_for_land = next_land_day >= 0 && day == next_land_day - 1;
            let target_land_day = next_land_day >= 0 && day >= next_land_day;
            let land_shortfall = ((next_land_cost as f64 - money).max(0.0)) as i64;

            let melon_price = current_prices[ITEM_MELON];
            let melon_needed = if melon_price > 0 {
                (land_shortfall + melon_price - 1) / melon_price
            } else {
                0
            };
            let land_finance_melon = shed[ITEM_MELON].min(melon_needed);
            let estimated_after_melon = money + (land_finance_melon * melon_price) as f64;
            let remaining_shortfall = ((next_land_cost as f64 - estimated_after_melon).max(0.0)) as i64;
            let wool_price = current_prices[ITEM_WOOL];
            let wool_needed = if wool_price > 0 {
                (remaining_shortfall + wool_price - 1) / wool_price
            } else {
                0
            };
            let land_finance_wool = shed[ITEM_WOOL].min(wool_needed);

            let financing = target_land_day && land_shortfall > 0;
            if financing && land_finance_melon > 0 {
                out.append(environment, player, MARKET_SELL, ITEM_MELON, land_finance_melon);
            }
            if financing && land_finance_wool > 0 {
                out.append(environment, player, MARKET_SELL, ITEM_WOOL, land_finance_wool);
            }

            let estimated_land_cash = money
                + if financing {
                    (land_finance_melon * melon_price + land_finance_wool * wool_price) as f64
                } else {
                    0.0
                };
            let scheduled_first_expansion = target_land_day && unlocked == 1 && estimated_land_cash >= 1_000.0;
            let scheduled_third_expansion = target_land_day && unlocked == 2 && estimated_land_cash >= 2_000.0;
            let yarn_expansion_early = day >= 12
                && unlocked == 3
                && target_animal_counts[[environment, player, 2]] >= 12
                && money >= 4_000.0;
            let early_land_buy = scheduled_first_expansion || scheduled_third_expansion || yarn_expansion_early;
            if early_land_buy {
                out.append(environment, player, MARKET_BUY_LAND, 0, 0);
            }

            let expansion_financing = day >= 6 && day <= 8 && unlocked <= 2 && money < 2_000.0;

            let fertilizer_reserve = 9_i64.min((ongoing_count + 2) / 3);
            let fertilizer_surplus = (shed[ITEM_FERTILIZER] - fertilizer_reserve).max(0);
            if fertilizer_surplus > 0 {
                out.append(
                    environment,
                    player,
                    MARKET_SELL,
                    ITEM_FERTILIZER,
                    fertilizer_surplus.min(SALE_BATCHES[ITEM_FERTILIZER]),
                );
            }

            for &item in &[ITEM_MILK, ITEM_WOOL, ITEM_MELON, ITEM_STRAWBERRY, ITEM_CARROT] {
                let mut shop_demand = 0_i64;
                for shop in 0..8 {
                    shop_demand += shops[shop] * SHOP_DEMAND[shop][item];
                }
                let demand_now = i64::from(town_center_tick)
                    + if town_tick { shop_demand } else { 0 };
                let sale_window = post_town_demand || (town_tick && demand_now == 0);
                let normal_count = shed[item].min(SALE_BATCHES[item]);
                let count = if item == ITEM_WOOL && expansion_financing {
                    shed[item]
                } else {
                    normal_count
                };
                let sell = count > 0
                    && (sale_window || pressure || (item == ITEM_WOOL && expansion_financing));
                if sell {
                    out.append(environment, player, MARKET_SELL, item, count);
                }
            }

            for &item in &[ITEM_TOMATO, ITEM_EGG] {
                let count = shed[item];
                if count > 0 && (post_town_demand || pressure) {
                    out.append(environment, player, MARKET_SELL, item, count);
                }
            }

            let wheat_surplus = (shed[ITEM_WHEAT] - wheat_reserve[[environment, player]]).max(0);
            let wheat_count = wheat_surplus.min(SALE_BATCHES[ITEM_WHEAT]);
            if wheat_count > 0
                && ((ratios[ITEM_WHEAT] >= 1.0 && post_town_demand) || pressure)
            {
                out.append(environment, player, MARKET_SELL, ITEM_WHEAT, wheat_count);
            }

            let land_buy = early_land_buy;
            let wheat_owned = shed[ITEM_WHEAT] + carried_wheat;
            let wheat_shortfall = (wheat_reserve[[environment, player]] - wheat_owned).max(0);
            if wheat_shortfall > 0 {
                out.append(
                    environment,
                    player,
                    MARKET_BUY_PRODUCT,
                    ITEM_WHEAT,
                    wheat_shortfall.min(4),
                );
            }

            let owned_cows = animal_counts[1] + shed[ITEM_COW] + carried_cows;
            let owned_sheep = animal_counts[2] + shed[ITEM_SHEEP] + carried_sheep;
            let missing_cows = (target_animal_counts[[environment, player, 1]] - owned_cows).max(0);
            let missing_sheep = (target_animal_counts[[environment, player, 2]] - owned_sheep).max(0);
            let expansion_ready = ((day < 6) || unlocked >= 2 || land_buy) && !reserve_for_land;
            let land_cost = match unlocked.min(3) {
                0 => 0_i64,
                1 => 1_000_i64,
                2 => 2_000_i64,
                _ => 4_000_i64,
            };
            let animal_cash_reserve = if target_animal_counts[[environment, player, 2]] >= 8 {
                800_i64
            } else if day < 6 {
                0_i64
            } else {
                200_i64
            };
            let scheduled_land_reserve = if reserve_for_land { next_land_cost } else { 0 };
            let mut budget = (money
                - animal_cash_reserve as f64
                - scheduled_land_reserve as f64
                - if land_buy { land_cost as f64 } else { 0.0 })
                .max(0.0) as i64;
            let buy_sheep = missing_sheep.min(budget / 500);
            budget -= buy_sheep * 500;
            let buy_cows = missing_cows.min(budget / 400);
            let establishing_second_field = day >= 7 && day <= 9;
            if expansion_ready && !establishing_second_field && buy_sheep > 0 {
                out.append(environment, player, MARKET_BUY_ANIMAL, ITEM_SHEEP, buy_sheep);
            }
            if expansion_ready && !establishing_second_field && buy_cows > 0 {
                out.append(environment, player, MARKET_BUY_ANIMAL, ITEM_COW, buy_cows);
            }

            let wanted_hands = target_hands[[environment, player]];
            let missing_hands = (wanted_hands - hands).max(0);
            let can_hire = wanted_hands > 0 && money >= 12.0;
            let essential_hands = 5_i64;
            if can_hire {
                for _ in 0..missing_hands.min(essential_hands) {
                    out.append(environment, player, MARKET_HIRE, 0, 0);
                }
            }

            let mut total_target = 0_i64;
            let mut total_replacement = 0_i64;
            let mut total_crops = 0_i64;
            let mut total_seeds = 0_i64;
            for crop in 0..CROP_COUNT {
                total_target += target_crop_counts[[environment, player, crop]];
                total_replacement += replacement_seeds[crop];
                total_crops += crops[crop];
                total_seeds += seeds[crop];
            }
            let total_missing = (total_target + total_replacement - total_crops - total_seeds).max(0);
            let wheat_missing = (
                target_crop_counts[[environment, player, ITEM_WHEAT]]
                    + replacement_seeds[ITEM_WHEAT]
                    - crops[ITEM_WHEAT]
                    - seeds[ITEM_WHEAT]
            )
            .max(0);
            let mut cash_seed_missing = (total_missing - wheat_missing).max(0);

            for crop in 0..CROP_COUNT {
                let missing = if crop == ITEM_WHEAT {
                    wheat_missing
                } else {
                    let preferred_deficit = (
                        target_crop_counts[[environment, player, crop]] - crops[crop]
                    )
                    .max(0);
                    let value = cash_seed_missing.min(preferred_deficit);
                    cash_seed_missing -= value;
                    value
                };
                if missing <= 0 {
                    continue;
                }
                let mut batch_limit = SEED_BUY_BATCHES[crop];
                if crop == ITEM_WHEAT && day >= 11 {
                    batch_limit = 12;
                } else if crop == ITEM_STRAWBERRY {
                    if day >= 7 && day <= 9 {
                        batch_limit = 9;
                    }
                    if day >= 11 {
                        batch_limit = 16;
                    }
                }
                out.append(
                    environment,
                    player,
                    MARKET_BUY_SEED,
                    crop,
                    missing.min(batch_limit),
                );
            }

            if expansion_ready && establishing_second_field && buy_sheep > 0 {
                out.append(environment, player, MARKET_BUY_ANIMAL, ITEM_SHEEP, buy_sheep);
            }
            if expansion_ready && establishing_second_field && buy_cows > 0 {
                out.append(environment, player, MARKET_BUY_ANIMAL, ITEM_COW, buy_cows);
            }

            if can_hire && missing_hands > essential_hands {
                for _ in essential_hands..missing_hands {
                    out.append(environment, player, MARKET_HIRE, 0, 0);
                }
            }
        }
    }

    Ok(())
}
