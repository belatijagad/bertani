//! Fixed-shape, player-relative observations and local action masks.
//!
//! The masks answer whether one count of an action can affect the current
//! state. They deliberately do not attempt to prove that a *joint* action is
//! feasible: two units can both see the same last seed, and simultaneous
//! market orders can change one another's quote or remaining shed capacity.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use kaggriculture_core::{Animal, Crop, Farm, Item, Product, Quadrant, Sim, Structure, Tile};

use crate::action::{
    ITEM_COUNT, MARKET_ACTION_COUNT, MARKET_BUY_ANIMAL, MARKET_BUY_LAND, MARKET_BUY_PRODUCT,
    MARKET_BUY_SEED, MARKET_HIRE, MARKET_NONE, MARKET_SELL, UNIT_ACTION_COUNT, UNIT_BUILD_COOP,
    UNIT_BUILD_PASTURE, UNIT_CARE, UNIT_COLLECT_FERTILIZER, UNIT_DIG, UNIT_DROP, UNIT_EAST,
    UNIT_FEED, UNIT_FERTILIZE, UNIT_HARVEST, UNIT_NORTH, UNIT_PASS, UNIT_PICKUP, UNIT_PLACE,
    UNIT_PLANT, UNIT_SOUTH, UNIT_WATER, UNIT_WEST,
};

pub(crate) const GLOBAL_CHANNELS: usize = 30;
pub(crate) const FARM_CHANNELS: usize = 9;
pub(crate) const TILE_CHANNELS: usize = 24;
pub(crate) const UNIT_CHANNELS: usize = 29;
pub(crate) const PRIVATE_CHANNELS: usize = 17;

const RELATIVE_FARMS: usize = 2;
const TILE_KIND_CHANNELS: usize = 9;
const TILE_CROP_CHANNELS: usize = 5;
const UNIT_SCALAR_CHANNELS: usize = 5;
const UNIT_INVENTORY_CHANNELS: usize = ITEM_COUNT;
const UNIT_ORDER_CHANNELS: usize = ITEM_COUNT;
const LAND_PRICES: [i64; 3] = [1_000, 2_000, 4_000];

/// Offsets for one flattened observation row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ObservationSpec {
    pub(crate) board_size: usize,
    pub(crate) max_units: usize,
    pub(crate) global: usize,
    pub(crate) farms: usize,
    pub(crate) tiles: usize,
    pub(crate) units: usize,
    pub(crate) private: usize,
    pub(crate) total: usize,
}

impl ObservationSpec {
    #[must_use]
    pub(crate) fn new(board_size: usize, max_units: usize) -> Self {
        assert!(board_size > 0, "board_size must be positive");
        assert!(max_units > 0, "max_units must include the farmer");

        let global = 0;
        let farms = GLOBAL_CHANNELS;
        let tiles = checked_add(farms, checked_mul(RELATIVE_FARMS, FARM_CHANNELS));
        let tile_count = checked_mul(board_size, board_size);
        let units = checked_add(
            tiles,
            checked_mul(checked_mul(RELATIVE_FARMS, tile_count), TILE_CHANNELS),
        );
        let private = checked_add(
            units,
            checked_mul(checked_mul(RELATIVE_FARMS, max_units), UNIT_CHANNELS),
        );
        let total = checked_add(private, PRIVATE_CHANNELS);
        Self {
            board_size,
            max_units,
            global,
            farms,
            tiles,
            units,
            private,
            total,
        }
    }

    fn farm_offset(self, relative_farm: usize) -> usize {
        self.farms + relative_farm * FARM_CHANNELS
    }

    fn tile_offset(self, relative_farm: usize, tile: usize) -> usize {
        self.tiles + (relative_farm * self.board_size * self.board_size + tile) * TILE_CHANNELS
    }

    fn unit_offset(self, relative_farm: usize, unit: usize) -> usize {
        self.units + (relative_farm * self.max_units + unit) * UNIT_CHANNELS
    }
}

/// Offsets for one flattened action-mask row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MaskSpec {
    pub(crate) max_units: usize,
    pub(crate) unit_ops: usize,
    pub(crate) unit_args: usize,
    pub(crate) market_ops: usize,
    pub(crate) market_args: usize,
    pub(crate) total: usize,
}

impl MaskSpec {
    #[must_use]
    pub(crate) fn new(max_units: usize) -> Self {
        assert!(max_units > 0, "max_units must include the farmer");

        let unit_ops = 0;
        let unit_args = checked_mul(max_units, UNIT_ACTION_COUNT);
        let market_ops = checked_add(unit_args, checked_mul(unit_args, ITEM_COUNT));
        let market_args = checked_add(market_ops, MARKET_ACTION_COUNT);
        let total = checked_add(market_args, checked_mul(MARKET_ACTION_COUNT, ITEM_COUNT));
        Self {
            max_units,
            unit_ops,
            unit_args,
            market_ops,
            market_args,
            total,
        }
    }

    fn unit_op(self, unit: usize, operation: i64) -> usize {
        self.unit_ops + unit * UNIT_ACTION_COUNT + operation_index(operation)
    }

    fn unit_arg(self, unit: usize, operation: i64, argument: usize) -> usize {
        self.unit_args
            + (unit * UNIT_ACTION_COUNT + operation_index(operation)) * ITEM_COUNT
            + argument
    }

    fn market_op(self, operation: i64) -> usize {
        self.market_ops + operation_index(operation)
    }

    fn market_arg(self, operation: i64, argument: usize) -> usize {
        self.market_args + operation_index(operation) * ITEM_COUNT + argument
    }
}

/// Encode one seat's player-relative observation and locally feasible actions.
pub(crate) fn encode(
    sim: &Sim,
    viewer: usize,
    observation: &mut [f32],
    mask: &mut [u8],
    unit_active: &mut [u8],
    overflow: &mut bool,
) -> Result<(), String> {
    if viewer >= RELATIVE_FARMS {
        return Err(format!("viewer must be 0 or 1, got {viewer}"));
    }
    if unit_active.is_empty() {
        return Err("unit_active must have at least one slot for the farmer".to_owned());
    }

    let observation_spec = ObservationSpec::new(sim.config.board_size, unit_active.len());
    let mask_spec = MaskSpec::new(unit_active.len());
    require_len("observation", observation, observation_spec.total)?;
    require_len("action mask", mask, mask_spec.total)?;
    validate_state(sim)?;

    observation.fill(0.0);
    mask.fill(0);
    unit_active.fill(0);
    *overflow = 1 + sim.state.farms[viewer].hands.len() > unit_active.len();

    encode_global(sim, observation_spec, observation);
    for (relative_farm, player) in [viewer, 1 - viewer].into_iter().enumerate() {
        encode_farm(
            sim,
            player,
            relative_farm,
            viewer,
            observation_spec,
            observation,
        );
    }
    encode_private(sim, viewer, observation_spec, observation);
    encode_masks(sim, viewer, mask_spec, mask, unit_active)?;
    Ok(())
}

fn checked_add(left: usize, right: usize) -> usize {
    left.checked_add(right)
        .expect("tensor dimensions must fit in usize")
}

fn checked_mul(left: usize, right: usize) -> usize {
    left.checked_mul(right)
        .expect("tensor dimensions must fit in usize")
}

fn operation_index(operation: i64) -> usize {
    usize::try_from(operation).expect("action IDs are nonnegative")
}

fn require_len<T>(name: &str, values: &[T], expected: usize) -> Result<(), String> {
    if values.len() == expected {
        Ok(())
    } else {
        Err(format!(
            "{name} buffer has {} values; expected exactly {expected}",
            values.len()
        ))
    }
}

fn validate_state(sim: &Sim) -> Result<(), String> {
    let tile_count = sim
        .config
        .board_size
        .checked_mul(sim.config.board_size)
        .ok_or_else(|| "board dimensions overflow usize".to_owned())?;
    for (player, farm) in sim.state.farms.iter().enumerate() {
        if farm.tiles.len() != tile_count {
            return Err(format!(
                "farm {player} has {} tiles; expected {tile_count}",
                farm.tiles.len()
            ));
        }
        let units = 1 + farm.hands.len();
        if farm.private.inventories.len() < units {
            return Err(format!(
                "farm {player} has {} unit inventories; expected at least {units}",
                farm.private.inventories.len()
            ));
        }
    }
    Ok(())
}

fn encode_global(sim: &Sim, spec: ObservationSpec, output: &mut [f32]) {
    let config = &sim.config;
    let state = &sim.state;
    let last_step = config.episode_steps.saturating_sub(1);
    let last_day = last_step / config.turns_per_day;
    let last_hour = config.turns_per_day.saturating_sub(1);
    output[spec.global] = ratio_u32(state.step, last_step);
    output[spec.global + 1] = ratio_u32(state.day, last_day);
    output[spec.global + 2] = ratio_u32(state.hour, last_hour);
    output[spec.global + 3] = ratio_u32(last_step.saturating_sub(state.step), last_step);

    let mut cursor = spec.global + 4;
    for product in Product::ALL {
        let index = product.index();
        let params = config.market_params[index];
        output[cursor] = ratio_f64(
            (state.market.inventory[index] - params.initial_inventory) as f64,
            params.threshold,
        );
        output[cursor + 1] = ratio_f64(state.market.prices[index], params.base);
        cursor += 2;
    }

    let mut counts = [0_usize; 8];
    for shop in &state.town.unlocked_shops {
        counts[shop.index()] += 1;
    }
    for (index, count) in counts.into_iter().enumerate() {
        output[cursor + index] = count as f32 / 8.0;
    }
}

fn encode_farm(
    sim: &Sim,
    player: usize,
    relative_farm: usize,
    viewer: usize,
    spec: ObservationSpec,
    output: &mut [f32],
) {
    let farm = &sim.state.farms[player];
    let offset = spec.farm_offset(relative_farm);
    output[offset] = ratio_f64(farm.money, sim.config.starting_money.max(1) as f64);
    output[offset + 1] = coordinate(farm.farmer.x, spec.board_size);
    output[offset + 2] = coordinate(farm.farmer.y, spec.board_size);
    output[offset + 3] = farm.hands.len() as f32 / spec.max_units.saturating_sub(1).max(1) as f32;
    for (index, quadrant) in Quadrant::ALL.into_iter().enumerate() {
        output[offset + 4 + index] = bool_value(farm.unlocked_quadrants.contains(&quadrant));
    }
    output[offset + 8] = farm.hires_today as f32 / spec.max_units as f32;

    for (tile_index, tile) in farm.tiles.iter().enumerate() {
        encode_tile(
            sim,
            tile,
            spec.tile_offset(relative_farm, tile_index),
            output,
        );
    }
    encode_units(
        farm,
        player == viewer,
        relative_farm,
        spec,
        sim.config.shed_capacity,
        output,
    );
}

fn encode_tile(sim: &Sim, tile: &Tile, offset: usize, output: &mut [f32]) {
    let numeric = offset + TILE_KIND_CHANNELS + TILE_CROP_CHANNELS;
    let episode_days = sim
        .config
        .episode_steps
        .div_ceil(sim.config.turns_per_day)
        .max(1);
    match tile {
        Tile::Empty => output[offset] = 1.0,
        Tile::Locked => output[offset + 1] = 1.0,
        Tile::Weed => output[offset + 2] = 1.0,
        Tile::Plant(plant) => {
            output[offset + 3] = 1.0;
            output[offset + TILE_KIND_CHANNELS + plant.crop.index()] = 1.0;
            let age = i64::from(sim.state.day).saturating_sub(i64::from(plant.planted_day));
            output[numeric] = ratio_i64(age.max(0), i64::from(episode_days));
            output[numeric + 1] = bool_value(plant.watered_today);
            output[numeric + 3] = ratio_i64(i64::from(plant.consecutive_unwatered), 2);
            output[numeric + 4] = ratio_i64(plant.yield_units, 6);
            let fertilizer_days = i64::from(plant.fertilized_until_day)
                .saturating_sub(i64::from(sim.state.day))
                .saturating_add(1)
                .max(0);
            output[numeric + 5] = ratio_i64(fertilizer_days, 3);
            if plant.max_lifespan_step >= 0 {
                output[numeric + 8] = ratio_i64(
                    plant
                        .max_lifespan_step
                        .saturating_sub(i64::from(sim.state.step))
                        .max(0),
                    i64::from(sim.config.episode_steps),
                );
            }
            let mature = age >= i64::from(plant.crop.first_yield_day());
            output[numeric + 9] = bool_value(mature && plant.yield_units > 0);
        }
        Tile::Structure { kind, animal: None } => {
            output[offset + structure_kind(*kind)] = 1.0;
        }
        Tile::Structure {
            animal: Some(animal),
            ..
        } => {
            output[offset + 6 + animal.animal.index()] = 1.0;
            let age = i64::from(sim.state.day).saturating_sub(i64::from(animal.placed_day));
            output[numeric] = ratio_i64(age.max(0), i64::from(episode_days));
            output[numeric + 1] = bool_value(animal.fed_today);
            output[numeric + 2] = bool_value(animal.cared_today);
            output[numeric + 3] = ratio_i64(i64::from(animal.consecutive_unfed), 2);
            output[numeric + 4] = ratio_i64(animal.yield_units, 6);
            output[numeric + 6] = bool_value(animal.fertilizer_available);
            output[numeric + 7] = ratio_i64(animal.pending_care_bonus, 6);
            output[numeric + 9] = bool_value(animal.yield_units > 0);
        }
    }
}

fn structure_kind(structure: Structure) -> usize {
    match structure {
        Structure::Coop => 4,
        Structure::Pasture => 5,
    }
}

fn encode_units(
    farm: &Farm,
    inventories_visible: bool,
    relative_farm: usize,
    spec: ObservationSpec,
    shed_capacity: i64,
    output: &mut [f32],
) {
    let represented = (1 + farm.hands.len()).min(spec.max_units);
    for unit in 0..represented {
        let position = if unit == 0 {
            farm.farmer
        } else {
            farm.hands[unit - 1]
        };
        let offset = spec.unit_offset(relative_farm, unit);
        output[offset] = 1.0;
        output[offset + 1] = bool_value(unit == 0);
        output[offset + 2] = coordinate(position.x, spec.board_size);
        output[offset + 3] = coordinate(position.y, spec.board_size);
        if !inventories_visible {
            continue;
        }

        output[offset + 4] = 1.0;
        let inventory = &farm.private.inventories[unit];
        for item in Item::ALL {
            output[offset + UNIT_SCALAR_CHANNELS + item.index()] =
                ratio_i64(inventory.get(item), shed_capacity);
        }
        let order_offset = offset + UNIT_SCALAR_CHANNELS + UNIT_INVENTORY_CHANNELS;
        output[order_offset..order_offset + UNIT_ORDER_CHANNELS].fill(-1.0);
        for (order_index, (item, _)) in inventory.iter().take(ITEM_COUNT).enumerate() {
            output[order_offset + order_index] = item.index() as f32 / (ITEM_COUNT - 1) as f32;
        }
    }
}

fn encode_private(sim: &Sim, viewer: usize, spec: ObservationSpec, output: &mut [f32]) {
    let private = &sim.state.farms[viewer].private;
    for item in Item::ALL {
        output[spec.private + item.index()] =
            ratio_i64(private.shed[item.index()], sim.config.shed_capacity);
    }
    for crop in Crop::ALL {
        output[spec.private + ITEM_COUNT + crop.index()] =
            ratio_i64(private.seeds[crop.index()], 10);
    }
}

fn coordinate(value: usize, board_size: usize) -> f32 {
    value as f32 / board_size.saturating_sub(1).max(1) as f32
}

fn ratio_u32(value: u32, denominator: u32) -> f32 {
    value as f32 / denominator.max(1) as f32
}

fn ratio_i64(value: i64, denominator: i64) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        value as f32 / denominator as f32
    }
}

fn ratio_f64(value: f64, denominator: f64) -> f32 {
    if denominator == 0.0 {
        0.0
    } else {
        (value / denominator) as f32
    }
}

const fn bool_value(value: bool) -> f32 {
    if value { 1.0 } else { 0.0 }
}

fn encode_masks(
    sim: &Sim,
    viewer: usize,
    spec: MaskSpec,
    mask: &mut [u8],
    unit_active: &mut [u8],
) -> Result<(), String> {
    let farm = &sim.state.farms[viewer];
    let represented = (1 + farm.hands.len()).min(spec.max_units);
    for (unit, active) in unit_active.iter_mut().enumerate() {
        set_unit_no_arg(mask, spec, unit, UNIT_PASS, true);
        if unit >= represented {
            continue;
        }
        *active = 1;
        if sim.state.done {
            continue;
        }
        encode_unit_mask(sim, farm, unit, spec, mask)?;
    }

    set_market_no_arg(mask, spec, MARKET_NONE, true);
    if !sim.state.done {
        encode_market_mask(sim, viewer, spec, mask);
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn encode_unit_mask(
    sim: &Sim,
    farm: &Farm,
    unit: usize,
    spec: MaskSpec,
    mask: &mut [u8],
) -> Result<(), String> {
    let position = if unit == 0 {
        farm.farmer
    } else {
        farm.hands[unit - 1]
    };
    let inventory = farm
        .private
        .inventories
        .get(unit)
        .ok_or_else(|| format!("viewer farm has no inventory for active unit slot {unit}"))?;
    let board_size = sim.config.board_size;
    set_unit_no_arg(mask, spec, unit, UNIT_NORTH, position.y > 0);
    set_unit_no_arg(mask, spec, unit, UNIT_SOUTH, position.y + 1 < board_size);
    set_unit_no_arg(mask, spec, unit, UNIT_EAST, position.x + 1 < board_size);
    set_unit_no_arg(mask, spec, unit, UNIT_WEST, position.x > 0);

    let shed_adjacent = is_shed_adjacent(position.x, position.y, board_size);
    if shed_adjacent {
        for item in Item::ALL {
            set_unit_arg(
                mask,
                spec,
                unit,
                UNIT_PICKUP,
                item.index(),
                farm.private.shed[item.index()] > 0,
            );
        }
    }
    set_unit_no_arg(
        mask,
        spec,
        unit,
        UNIT_DROP,
        shed_adjacent && inventory.total() > 0,
    );

    let tile = farm.tile(board_size, position);
    let shed_has_room = farm.private.shed_total() < sim.config.shed_capacity;
    for item in Item::ALL {
        let carried = inventory.get(item) > 0;
        let animal_placement = item.as_animal().is_some_and(|animal| {
            matches!(
                tile,
                Tile::Structure { kind, animal: None } if *kind == animal.structure()
            )
        });
        set_unit_arg(
            mask,
            spec,
            unit,
            UNIT_PLACE,
            item.index(),
            carried && (animal_placement || (shed_adjacent && shed_has_room)),
        );
    }

    let owned = !matches!(tile, Tile::Locked);
    for crop in Crop::ALL {
        set_unit_arg(
            mask,
            spec,
            unit,
            UNIT_PLANT,
            crop.index(),
            owned && matches!(tile, Tile::Empty) && farm.private.seeds[crop.index()] > 0,
        );
    }

    let can_water = matches!(tile, Tile::Plant(plant) if !plant.watered_today);
    set_unit_no_arg(mask, spec, unit, UNIT_WATER, owned && can_water);
    let can_harvest = match tile {
        Tile::Plant(plant) => {
            plant.yield_units > 0
                && i64::from(sim.state.day) - i64::from(plant.planted_day)
                    >= i64::from(plant.crop.first_yield_day())
        }
        Tile::Structure {
            animal: Some(animal),
            ..
        } => animal.yield_units > 0,
        _ => false,
    };
    set_unit_no_arg(mask, spec, unit, UNIT_HARVEST, owned && can_harvest);
    set_unit_no_arg(
        mask,
        spec,
        unit,
        UNIT_FERTILIZE,
        owned && matches!(tile, Tile::Plant(_)) && inventory.get(Item::Fertilizer) > 0,
    );
    let can_dig = !matches!(
        tile,
        Tile::Empty
            | Tile::Locked
            | Tile::Structure {
                animal: Some(_),
                ..
            }
    );
    set_unit_no_arg(mask, spec, unit, UNIT_DIG, can_dig);
    set_unit_no_arg(
        mask,
        spec,
        unit,
        UNIT_BUILD_COOP,
        owned && matches!(tile, Tile::Empty),
    );
    set_unit_no_arg(
        mask,
        spec,
        unit,
        UNIT_BUILD_PASTURE,
        owned && matches!(tile, Tile::Empty),
    );
    let can_feed = matches!(
        tile,
        Tile::Structure { animal: Some(animal), .. } if !animal.fed_today
    ) && inventory.get(Item::Wheat) > 0;
    set_unit_no_arg(mask, spec, unit, UNIT_FEED, owned && can_feed);
    let can_collect = matches!(
        tile,
        Tile::Structure { animal: Some(animal), .. } if animal.fertilizer_available
    );
    set_unit_no_arg(
        mask,
        spec,
        unit,
        UNIT_COLLECT_FERTILIZER,
        owned && can_collect,
    );
    let can_care = matches!(
        tile,
        Tile::Structure { animal: Some(animal), .. } if !animal.cared_today
    );
    set_unit_no_arg(mask, spec, unit, UNIT_CARE, owned && can_care);
    Ok(())
}

fn encode_market_mask(sim: &Sim, viewer: usize, spec: MaskSpec, mask: &mut [u8]) {
    let farm = &sim.state.farms[viewer];
    let hire_cost = i128::from(sim.config.farm_hand_cost_multiplier)
        .saturating_mul(fibonacci(farm.hires_today));
    set_market_no_arg(
        mask,
        spec,
        MARKET_HIRE,
        !float_less_than_integer(farm.money, hire_cost),
    );

    let extra_land = farm.unlocked_quadrants.len().saturating_sub(1);
    let land_affordable = LAND_PRICES
        .get(extra_land)
        .is_some_and(|cost| !float_less_than_integer(farm.money, i128::from(*cost)));
    set_market_no_arg(mask, spec, MARKET_BUY_LAND, land_affordable);

    for crop in Crop::ALL {
        set_market_arg(
            mask,
            spec,
            MARKET_BUY_SEED,
            crop.index(),
            !float_less_than_integer(farm.money, i128::from(crop.seed_cost())),
        );
    }

    let shed_has_room = farm.private.shed_total() < sim.config.shed_capacity;
    for product in [Product::Wheat, Product::Fertilizer] {
        let inventory = sim.state.market.inventory[product.index()];
        let quote = sim.market_price(product, inventory.saturating_sub(1));
        set_market_arg(
            mask,
            spec,
            MARKET_BUY_PRODUCT,
            product.index(),
            shed_has_room && !float_less_than_integer(farm.money, i128::from(quote)),
        );
    }
    for animal in Animal::ALL {
        set_market_arg(
            mask,
            spec,
            MARKET_BUY_ANIMAL,
            animal.item().index(),
            shed_has_room && !float_less_than_integer(farm.money, i128::from(animal.cost())),
        );
    }
    for product in Product::ALL {
        set_market_arg(
            mask,
            spec,
            MARKET_SELL,
            product.index(),
            farm.private.shed[product.index()] > 0,
        );
    }
}

fn set_unit_no_arg(mask: &mut [u8], spec: MaskSpec, unit: usize, operation: i64, valid: bool) {
    if valid {
        mask[spec.unit_op(unit, operation)] = 1;
        mask[spec.unit_arg(unit, operation, 0)] = 1;
    }
}

fn set_unit_arg(
    mask: &mut [u8],
    spec: MaskSpec,
    unit: usize,
    operation: i64,
    argument: usize,
    valid: bool,
) {
    if valid {
        mask[spec.unit_op(unit, operation)] = 1;
        mask[spec.unit_arg(unit, operation, argument)] = 1;
    }
}

fn set_market_no_arg(mask: &mut [u8], spec: MaskSpec, operation: i64, valid: bool) {
    if valid {
        mask[spec.market_op(operation)] = 1;
        mask[spec.market_arg(operation, 0)] = 1;
    }
}

fn set_market_arg(mask: &mut [u8], spec: MaskSpec, operation: i64, argument: usize, valid: bool) {
    if valid {
        mask[spec.market_op(operation)] = 1;
        mask[spec.market_arg(operation, argument)] = 1;
    }
}

fn is_shed_adjacent(x: usize, y: usize, board_size: usize) -> bool {
    let half = board_size / 2;
    [
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    ]
    .contains(&(x, y))
}

fn fibonacci(index: usize) -> i128 {
    let (mut current, mut next) = (1_i128, 1_i128);
    for _ in 0..index {
        (current, next) = (next, current.saturating_add(next));
    }
    current
}

/// Match the core's exact comparison between binary floats and integers.
fn float_less_than_integer(value: f64, integer: i128) -> bool {
    if value.is_nan() || value == f64::INFINITY {
        return false;
    }
    if value == f64::NEG_INFINITY {
        return true;
    }
    (value.floor() as i128) < integer
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use kaggriculture_core::{Action, Config, Inventory, Item, PlacedAnimal, Shop, UnitAction};

    use super::*;

    fn buffers(sim: &Sim, max_units: usize) -> (Vec<f32>, Vec<u8>, Vec<u8>) {
        let observation =
            vec![f32::NAN; ObservationSpec::new(sim.config.board_size, max_units).total];
        let mask = vec![1; MaskSpec::new(max_units).total];
        let active = vec![1; max_units];
        (observation, mask, active)
    }

    fn valid(mask: &[u8], index: usize) -> bool {
        mask[index] != 0
    }

    #[test]
    fn specs_have_stable_contiguous_offsets() {
        let observation = ObservationSpec::new(10, 64);
        assert_eq!(observation.global, 0);
        assert_eq!(observation.farms, 30);
        assert_eq!(observation.tiles, 48);
        assert_eq!(observation.units, 4_848);
        assert_eq!(observation.private, 8_560);
        assert_eq!(observation.total, 8_577);

        let mask = MaskSpec::new(64);
        assert_eq!(mask.unit_ops, 0);
        assert_eq!(mask.unit_args, 1_152);
        assert_eq!(mask.market_ops, 14_976);
        assert_eq!(mask.market_args, 14_983);
        assert_eq!(mask.total, 15_067);
    }

    #[test]
    fn observations_are_relative_and_do_not_leak_opponent_inventory() {
        let mut sim = Sim::default();
        sim.state.farms[0].money = 1_500.0;
        sim.state.farms[1].money = 6_000.0;
        sim.state.farms[0].private.shed[Item::Milk.index()] = 2;
        sim.state.farms[1].private.shed[Item::Carrot.index()] = 20;
        sim.state.town.unlocked_shops = vec![Shop::Bakery, Shop::Bakery];
        sim.step([
            &Action {
                farmer: UnitAction::Pickup {
                    item: Item::Milk,
                    count: 1,
                },
                ..Action::default()
            },
            &Action::pass(),
        ]);

        let max_units = 3;
        let spec = ObservationSpec::new(sim.config.board_size, max_units);
        let (mut obs0, mut mask0, mut active0) = buffers(&sim, max_units);
        let mut overflow = false;
        encode(&sim, 0, &mut obs0, &mut mask0, &mut active0, &mut overflow).unwrap();
        assert!(!overflow);
        assert_eq!(obs0[spec.farm_offset(0)], 0.5);
        assert_eq!(obs0[spec.farm_offset(1)], 2.0);
        assert_eq!(obs0[spec.private + Item::Milk.index()], 0.01);
        assert_eq!(obs0[spec.global + 22 + Shop::Bakery.index()], 0.25);

        let own_unit = spec.unit_offset(0, 0);
        assert_eq!(obs0[own_unit + 4], 1.0);
        assert_eq!(
            obs0[own_unit + UNIT_SCALAR_CHANNELS + Item::Milk.index()],
            0.01
        );
        let own_order = own_unit + UNIT_SCALAR_CHANNELS + UNIT_INVENTORY_CHANNELS;
        assert_eq!(obs0[own_order], Item::Milk.index() as f32 / 11.0);
        assert_eq!(obs0[own_order + 1], -1.0);

        let (mut obs1, mut mask1, mut active1) = buffers(&sim, max_units);
        encode(&sim, 1, &mut obs1, &mut mask1, &mut active1, &mut overflow).unwrap();
        assert_eq!(obs1[spec.farm_offset(0)], 2.0);
        assert_eq!(obs1[spec.farm_offset(1)], 0.5);
        assert_eq!(obs1[spec.private + Item::Carrot.index()], 0.2);
        let opponent_unit = spec.unit_offset(1, 0);
        assert_eq!(obs1[opponent_unit + 4], 0.0);
        assert!(
            obs1[opponent_unit + UNIT_SCALAR_CHANNELS..opponent_unit + UNIT_CHANNELS]
                .iter()
                .all(|value| *value == 0.0)
        );
    }

    #[test]
    fn masks_cover_initial_and_contextual_one_count_actions() {
        let mut sim = Sim::default();
        sim.state.farms[0].private.shed[Item::Milk.index()] = 1;
        sim.state.farms[0].private.shed[Item::Egg.index()] = 1;
        sim.state.farms[0].private.seeds[Crop::Carrot.index()] = 1;
        let max_units = 2;
        let spec = MaskSpec::new(max_units);
        let (mut observation, mut mask, mut active) = buffers(&sim, max_units);
        let mut overflow = false;
        encode(
            &sim,
            0,
            &mut observation,
            &mut mask,
            &mut active,
            &mut overflow,
        )
        .unwrap();

        assert_eq!(active, [1, 0]);
        for operation in [UNIT_PASS, UNIT_NORTH, UNIT_SOUTH, UNIT_EAST, UNIT_WEST] {
            assert!(valid(&mask, spec.unit_op(0, operation)));
            assert!(valid(&mask, spec.unit_arg(0, operation, 0)));
        }
        assert!(valid(
            &mask,
            spec.unit_arg(0, UNIT_PICKUP, Item::Milk.index())
        ));
        assert!(valid(
            &mask,
            spec.unit_arg(0, UNIT_PLANT, Crop::Carrot.index())
        ));
        assert!(valid(&mask, spec.unit_op(0, UNIT_BUILD_COOP)));
        assert!(valid(&mask, spec.unit_op(0, UNIT_BUILD_PASTURE)));
        assert!(!valid(&mask, spec.unit_op(0, UNIT_DROP)));
        assert!(valid(&mask, spec.unit_op(1, UNIT_PASS)));
        assert_eq!(
            mask[spec.unit_ops + UNIT_ACTION_COUNT..spec.unit_args]
                .iter()
                .filter(|value| **value != 0)
                .count(),
            1
        );

        assert!(valid(&mask, spec.market_op(MARKET_NONE)));
        assert!(valid(&mask, spec.market_op(MARKET_HIRE)));
        assert!(valid(&mask, spec.market_op(MARKET_BUY_LAND)));
        assert!(valid(
            &mask,
            spec.market_arg(MARKET_BUY_SEED, Crop::Melon.index())
        ));
        assert!(valid(
            &mask,
            spec.market_arg(MARKET_BUY_PRODUCT, Item::Wheat.index())
        ));
        assert!(valid(
            &mask,
            spec.market_arg(MARKET_BUY_PRODUCT, Item::Fertilizer.index())
        ));
        assert!(!valid(
            &mask,
            spec.market_arg(MARKET_BUY_PRODUCT, Item::Carrot.index())
        ));
        assert!(valid(
            &mask,
            spec.market_arg(MARKET_BUY_ANIMAL, Item::Sheep.index())
        ));
        assert!(valid(
            &mask,
            spec.market_arg(MARKET_SELL, Item::Egg.index())
        ));
        assert!(!valid(
            &mask,
            spec.market_arg(MARKET_SELL, Item::Carrot.index())
        ));
    }

    #[test]
    fn plant_masks_follow_tile_state_and_carried_inventory() {
        let mut sim = Sim::default();
        sim.state.farms[0].private.shed[Item::Fertilizer.index()] = 1;
        sim.state.farms[0].private.seeds[Crop::Carrot.index()] = 1;
        sim.step([
            &Action {
                farmer: UnitAction::Pickup {
                    item: Item::Fertilizer,
                    count: 1,
                },
                ..Action::default()
            },
            &Action::pass(),
        ]);
        sim.step([
            &Action {
                farmer: UnitAction::Plant(Crop::Carrot),
                ..Action::default()
            },
            &Action::pass(),
        ]);

        let max_units = 1;
        let spec = MaskSpec::new(max_units);
        let (mut observation, mut mask, mut active) = buffers(&sim, max_units);
        let mut overflow = false;
        encode(
            &sim,
            0,
            &mut observation,
            &mut mask,
            &mut active,
            &mut overflow,
        )
        .unwrap();
        assert!(valid(&mask, spec.unit_op(0, UNIT_WATER)));
        assert!(valid(&mask, spec.unit_op(0, UNIT_FERTILIZE)));
        assert!(valid(&mask, spec.unit_op(0, UNIT_DIG)));
        assert!(!valid(&mask, spec.unit_op(0, UNIT_HARVEST)));
        assert!(!valid(&mask, spec.unit_op(0, UNIT_BUILD_COOP)));

        sim.step([
            &Action {
                farmer: UnitAction::Water,
                ..Action::default()
            },
            &Action::pass(),
        ]);
        encode(
            &sim,
            0,
            &mut observation,
            &mut mask,
            &mut active,
            &mut overflow,
        )
        .unwrap();
        assert!(!valid(&mask, spec.unit_op(0, UNIT_WATER)));
    }

    #[test]
    fn animal_channels_and_masks_cover_every_local_interaction() {
        let mut sim = Sim::default();
        sim.state.farms[0].private.shed[Item::Wheat.index()] = 1;
        sim.step([
            &Action {
                farmer: UnitAction::Pickup {
                    item: Item::Wheat,
                    count: 1,
                },
                ..Action::default()
            },
            &Action::pass(),
        ]);
        let position = sim.state.farms[0].farmer;
        let tile_index = position.y * sim.config.board_size + position.x;
        sim.state.farms[0].tiles[tile_index] = Tile::Structure {
            kind: Structure::Coop,
            animal: Some(PlacedAnimal {
                animal: Animal::Goose,
                placed_day: 0,
                yield_units: 2,
                consecutive_unfed: 1,
                fed_today: false,
                cared_today: false,
                fertilizer_available: true,
                pending_care_bonus: 3,
            }),
        };

        let max_units = 1;
        let observation_spec = ObservationSpec::new(sim.config.board_size, max_units);
        let mask_spec = MaskSpec::new(max_units);
        let (mut observation, mut mask, mut active) = buffers(&sim, max_units);
        let mut overflow = false;
        encode(
            &sim,
            0,
            &mut observation,
            &mut mask,
            &mut active,
            &mut overflow,
        )
        .unwrap();

        let tile = observation_spec.tile_offset(0, tile_index);
        let numeric = tile + TILE_KIND_CHANNELS + TILE_CROP_CHANNELS;
        assert_eq!(observation[tile + 6], 1.0);
        assert_eq!(observation[numeric + 3], 0.5);
        assert_eq!(observation[numeric + 4], 2.0 / 6.0);
        assert_eq!(observation[numeric + 6], 1.0);
        assert_eq!(observation[numeric + 7], 0.5);
        assert_eq!(observation[numeric + 9], 1.0);
        assert!(valid(&mask, mask_spec.unit_op(0, UNIT_HARVEST)));
        assert!(valid(&mask, mask_spec.unit_op(0, UNIT_FEED)));
        assert!(valid(&mask, mask_spec.unit_op(0, UNIT_COLLECT_FERTILIZER)));
        assert!(valid(&mask, mask_spec.unit_op(0, UNIT_CARE)));
        assert!(!valid(&mask, mask_spec.unit_op(0, UNIT_DIG)));
    }

    #[test]
    fn detects_overflow_and_terminal_masks_only_allow_no_ops() {
        let config = Config {
            episode_steps: 2,
            ..Config::default()
        };
        let mut sim = Sim::new(config);
        let spawn = sim.state.farms[0].farmer;
        sim.state.farms[0].hands.push(spawn);
        sim.state.farms[0]
            .private
            .inventories
            .push(Inventory::default());

        let max_units = 1;
        let spec = MaskSpec::new(max_units);
        let (mut observation, mut mask, mut active) = buffers(&sim, max_units);
        let mut overflow = false;
        encode(
            &sim,
            0,
            &mut observation,
            &mut mask,
            &mut active,
            &mut overflow,
        )
        .unwrap();
        assert!(overflow);

        sim.step([&Action::pass(), &Action::pass()]);
        encode(
            &sim,
            0,
            &mut observation,
            &mut mask,
            &mut active,
            &mut overflow,
        )
        .unwrap();
        assert!(sim.state.done);
        assert_eq!(
            mask[spec.unit_ops..spec.unit_args]
                .iter()
                .filter(|value| **value != 0)
                .count(),
            1
        );
        assert!(valid(&mask, spec.unit_op(0, UNIT_PASS)));
        assert_eq!(
            mask[spec.market_ops..spec.market_args]
                .iter()
                .filter(|value| **value != 0)
                .count(),
            1
        );
        assert!(valid(&mask, spec.market_op(MARKET_NONE)));
    }

    #[test]
    fn rejects_wrong_buffer_sizes_without_partial_output() {
        let sim = Sim::default();
        let mut observation = vec![7.0; ObservationSpec::new(10, 1).total - 1];
        let mut mask = vec![1; MaskSpec::new(1).total];
        let mut active = vec![1];
        let mut overflow = true;
        let error = encode(
            &sim,
            0,
            &mut observation,
            &mut mask,
            &mut active,
            &mut overflow,
        )
        .unwrap_err();
        assert!(error.contains("observation buffer"));
        assert!(observation.iter().all(|value| *value == 7.0));
        assert!(mask.iter().all(|value| *value == 1));
        assert_eq!(active[0], 1);
        assert!(overflow);
    }
}
