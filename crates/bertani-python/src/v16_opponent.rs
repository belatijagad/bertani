//! Independent native executor for the preserved V16-RC5 opponent.
//!
//! This module owns the immutable action trace and all per-episode repair
//! state. It reads typed simulator state directly and does not depend on the
//! neural observation encoder or any rule-based planning module.

use kaggriculture_core::{Item, Product, State, Tile, shop_products};
use numpy::{PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::action::{
    ACTION_FIELD_COUNT, MARKET_NONE, MARKET_SELL, PLAYER_COUNT, UNIT_BUILD_PASTURE, UNIT_DIG,
    UNIT_PICKUP, UNIT_PLANT,
};
use crate::vec_env::{NativeVecEnv, Slot};

const FRONT_RUN_ITEMS: [Item; 4] = [Item::Melon, Item::Milk, Item::Strawberry, Item::Wool];

#[derive(Clone, Copy, Debug, Default)]
struct ActionRow([i64; ACTION_FIELD_COUNT]);

impl ActionRow {
    const fn operation(self) -> i64 {
        self.0[0]
    }

    fn item(self) -> usize {
        // Invalid trace IDs deliberately cannot match any real Item.
        match usize::try_from(self.0[1]) {
            Ok(value) => value,
            Err(_) => usize::MAX,
        }
    }

    const fn count(self) -> i64 {
        self.0[2]
    }
}

#[derive(Debug)]
struct Trace {
    units: Vec<ActionRow>,
    market: Vec<ActionRow>,
    market_lengths: Vec<usize>,
    steps: usize,
    units_per_step: usize,
    orders_per_step: usize,
}

impl Trace {
    fn unit(&self, step: usize, unit: usize) -> ActionRow {
        self.units[step * self.units_per_step + unit]
    }

    fn market(&self, step: usize, order: usize) -> ActionRow {
        self.market[step * self.orders_per_step + order]
    }

    fn step(&self, step: u32) -> usize {
        usize::try_from(step)
            .unwrap_or(usize::MAX)
            .min(self.steps - 1)
    }
}

#[derive(Debug)]
struct PlayerState {
    repair_active: Vec<bool>,
    repair_start: Vec<i64>,
    repair_intended: Vec<ActionRow>,
    due_step: i64,
    due: [i64; Product::COUNT],
}

impl PlayerState {
    fn new(max_units: usize) -> Self {
        Self {
            repair_active: vec![false; max_units],
            repair_start: vec![-1; max_units],
            repair_intended: vec![ActionRow::default(); max_units],
            due_step: -1,
            due: [0; Product::COUNT],
        }
    }

    fn reset(&mut self) {
        self.repair_active.fill(false);
        self.repair_start.fill(-1);
        self.repair_intended.fill(ActionRow::default());
        self.due_step = -1;
        self.due.fill(0);
    }
}

#[derive(Debug)]
struct EnvironmentState {
    episode_id: u64,
    players: [PlayerState; PLAYER_COUNT],
}

impl EnvironmentState {
    fn new(max_units: usize) -> Self {
        Self {
            episode_id: u64::MAX,
            players: [PlayerState::new(max_units), PlayerState::new(max_units)],
        }
    }

    fn sync_episode(&mut self, episode_id: u64) {
        if self.episode_id != episode_id {
            for player in &mut self.players {
                player.reset();
            }
            self.episode_id = episode_id;
        }
    }
}

/// Rust-owned V16 trace executor used only by the Python V16 wrapper.
#[pyclass(module = "bertani._rust")]
pub struct NativeV16Opponent {
    trace: Trace,
    environments: Vec<EnvironmentState>,
}

#[pymethods]
impl NativeV16Opponent {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        trace_units: Bound<'_, PyArray3<i16>>,
        trace_market: Bound<'_, PyArray3<i16>>,
        trace_market_lengths: Bound<'_, PyArray1<i16>>,
    ) -> PyResult<Self> {
        require_c_order("trace_units", trace_units.is_c_contiguous())?;
        require_c_order("trace_market", trace_market.is_c_contiguous())?;
        require_c_order(
            "trace_market_lengths",
            trace_market_lengths.is_c_contiguous(),
        )?;
        let unit_shape = trace_units.shape();
        let market_shape = trace_market.shape();
        if unit_shape.len() != 3 || unit_shape[2] != ACTION_FIELD_COUNT {
            return Err(PyValueError::new_err(format!(
                "trace_units must have shape [steps, units, {ACTION_FIELD_COUNT}]"
            )));
        }
        if market_shape.len() != 3 || market_shape[2] != ACTION_FIELD_COUNT {
            return Err(PyValueError::new_err(format!(
                "trace_market must have shape [steps, orders, {ACTION_FIELD_COUNT}]"
            )));
        }
        if unit_shape[0] == 0 || unit_shape[1] == 0 || market_shape[1] == 0 {
            return Err(PyValueError::new_err(
                "V16 trace dimensions must be positive",
            ));
        }
        if market_shape[0] != unit_shape[0] || trace_market_lengths.shape() != [unit_shape[0]] {
            return Err(PyValueError::new_err(
                "V16 trace arrays must have the same step dimension",
            ));
        }

        let units = trace_units
            .readonly()
            .as_slice()?
            .chunks_exact(ACTION_FIELD_COUNT)
            .map(|row| ActionRow([i64::from(row[0]), i64::from(row[1]), i64::from(row[2])]))
            .collect();
        let market = trace_market
            .readonly()
            .as_slice()?
            .chunks_exact(ACTION_FIELD_COUNT)
            .map(|row| ActionRow([i64::from(row[0]), i64::from(row[1]), i64::from(row[2])]))
            .collect();
        let lengths = trace_market_lengths.readonly();
        let market_lengths = lengths
            .as_slice()?
            .iter()
            .enumerate()
            .map(|(step, &length)| {
                usize::try_from(length)
                    .ok()
                    .filter(|&value| value <= market_shape[1])
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "trace market length {length} at step {step} exceeds {}",
                            market_shape[1]
                        ))
                    })
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(Self {
            trace: Trace {
                units,
                market,
                market_lengths,
                steps: unit_shape[0],
                units_per_step: unit_shape[1],
                orders_per_step: market_shape[1],
            },
            environments: Vec::new(),
        })
    }

    fn reset(&mut self) {
        self.environments.clear();
    }

    #[allow(clippy::needless_pass_by_value)]
    fn act_into(
        &mut self,
        environment: PyRef<'_, NativeVecEnv>,
        seats: Bound<'_, PyArray1<i64>>,
        unit_actions: Bound<'_, PyArray4<i16>>,
        market_actions: Bound<'_, PyArray4<i16>>,
        market_lengths: Bound<'_, PyArray2<i16>>,
    ) -> PyResult<()> {
        let slots = environment.slots();
        let n = slots.len();
        let max_units = environment.max_units_native();
        let max_orders = environment.max_orders_native();
        require_shape("seats", seats.shape(), &[n])?;
        require_shape(
            "unit_actions",
            unit_actions.shape(),
            &[n, PLAYER_COUNT, max_units, ACTION_FIELD_COUNT],
        )?;
        require_shape(
            "market_actions",
            market_actions.shape(),
            &[n, PLAYER_COUNT, max_orders, ACTION_FIELD_COUNT],
        )?;
        require_shape("market_lengths", market_lengths.shape(), &[n, PLAYER_COUNT])?;
        require_c_order("seats", seats.is_c_contiguous())?;
        require_c_order("unit_actions", unit_actions.is_c_contiguous())?;
        require_c_order("market_actions", market_actions.is_c_contiguous())?;
        require_c_order("market_lengths", market_lengths.is_c_contiguous())?;
        if self.trace.orders_per_step > max_orders {
            return Err(PyValueError::new_err(format!(
                "V16 trace has {} market slots but environment allows {max_orders}",
                self.trace.orders_per_step
            )));
        }

        let seat_values = seats.readonly();
        let seat_values = seat_values.as_slice()?;
        if let Some((index, seat)) = seat_values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, seat)| !matches!(seat, 0 | 1))
        {
            return Err(PyValueError::new_err(format!(
                "seat {index} is {seat}; expected 0 or 1"
            )));
        }
        if self.environments.len() != n {
            self.environments = (0..n).map(|_| EnvironmentState::new(max_units)).collect();
        }

        let mut unit_actions = unit_actions.try_readwrite()?;
        let mut market_actions = market_actions.try_readwrite()?;
        let mut market_lengths = market_lengths.try_readwrite()?;
        let unit_values = unit_actions.as_slice_mut()?;
        let market_values = market_actions.as_slice_mut()?;
        let length_values = market_lengths.as_slice_mut()?;
        unit_values.fill(0);
        market_values.fill(0);
        length_values.fill(0);

        let trace = &self.trace;
        let unit_stride = PLAYER_COUNT * max_units * ACTION_FIELD_COUNT;
        let market_stride = PLAYER_COUNT * max_orders * ACTION_FIELD_COUNT;
        self.environments
            .par_iter_mut()
            .zip(slots.par_iter())
            .zip(seat_values.par_iter().copied())
            .zip(unit_values.par_chunks_mut(unit_stride))
            .zip(market_values.par_chunks_mut(market_stride))
            .zip(length_values.par_chunks_mut(PLAYER_COUNT))
            .for_each(|(((((policy, slot), seat), units), market), lengths)| {
                act_environment(
                    trace,
                    policy,
                    slot,
                    usize::try_from(seat).unwrap_or_default(),
                    max_units,
                    max_orders,
                    units,
                    market,
                    lengths,
                );
            });
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn act_environment(
    trace: &Trace,
    policy: &mut EnvironmentState,
    slot: &Slot,
    player: usize,
    max_units: usize,
    max_orders: usize,
    units: &mut [i16],
    market: &mut [i16],
    lengths: &mut [i16],
) {
    policy.sync_episode(slot.episode_id);
    let state = &slot.sim.state;
    let step = trace.step(state.step);
    let farm = &state.farms[player];
    let active_units = 1 + farm.hands.len();
    let unit_limit = active_units.min(max_units).min(trace.units_per_step);
    for unit in 0..unit_limit {
        set_unit(units, player, unit, max_units, trace.unit(step, unit));
    }
    let market_length = trace.market_lengths[step].min(max_orders);
    for order in 0..market_length {
        set_market(market, player, order, max_orders, trace.market(step, order));
    }
    lengths[player] = compact_value(i64::try_from(market_length).unwrap_or(i64::MAX));

    let player_state = &mut policy.players[player];
    apply_weed_repair(
        trace,
        player_state,
        state,
        player,
        slot.sim.config.board_size,
        max_units,
        units,
    );
    apply_front_run(
        trace,
        player_state,
        state,
        player,
        max_units,
        max_orders,
        units,
        market,
        &mut lengths[player],
    );
}

fn apply_weed_repair(
    trace: &Trace,
    policy: &mut PlayerState,
    state: &State,
    player: usize,
    board_size: usize,
    max_units: usize,
    units: &mut [i16],
) {
    let step = i64::from(state.step);
    let farm = &state.farms[player];
    let active_units = (1 + farm.hands.len()).min(max_units);
    for unit in 0..max_units {
        if !policy.repair_active[unit] {
            continue;
        }
        if unit >= active_units || step - policy.repair_start[unit] > 9 {
            policy.repair_active[unit] = false;
            continue;
        }
        let age = step - policy.repair_start[unit];
        if age == 1 {
            set_unit(units, player, unit, max_units, policy.repair_intended[unit]);
        } else if (2..=9).contains(&age) && unit < trace.units_per_step {
            let replay_step = trace.step(state.step.saturating_sub(1));
            set_unit(
                units,
                player,
                unit,
                max_units,
                trace.unit(replay_step, unit),
            );
        }
    }

    for unit in 0..active_units {
        if policy.repair_active[unit] {
            continue;
        }
        let intended = get_unit(units, player, unit, max_units);
        if !matches!(intended.operation(), UNIT_BUILD_PASTURE | UNIT_PLANT) {
            continue;
        }
        let position = if unit == 0 {
            farm.farmer
        } else {
            farm.hands[unit - 1]
        };
        if matches!(farm.tile(board_size, position), Tile::Weed) {
            policy.repair_active[unit] = true;
            policy.repair_start[unit] = step;
            policy.repair_intended[unit] = intended;
            set_unit(units, player, unit, max_units, ActionRow([UNIT_DIG, 0, 0]));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_front_run(
    trace: &Trace,
    policy: &mut PlayerState,
    state: &State,
    player: usize,
    max_units: usize,
    max_orders: usize,
    units: &[i16],
    market: &mut [i16],
    market_length: &mut i16,
) {
    let step = i64::from(state.step);
    if policy.due_step == step {
        repay_due(policy, player, max_orders, market, market_length);
    } else if policy.due_step >= 0 && policy.due_step < step {
        policy.due_step = -1;
        policy.due.fill(0);
    }

    let future = trace.step(state.step.saturating_add(1));
    let farm = &state.farms[player];
    let mut moved_any = false;
    for item in FRONT_RUN_ITEMS {
        let item_index = item.index();
        let target: i64 = (0..trace.market_lengths[future])
            .map(|order| trace.market(future, order))
            .filter(|row| row.operation() == MARKET_SELL && row.item() == item_index)
            .map(ActionRow::count)
            .sum();
        if target <= 0 || product_demand(state, item) > 0 {
            continue;
        }
        let pickup_reserve: i64 = (0..max_units)
            .map(|unit| get_unit(units, player, unit, max_units))
            .filter(|row| row.operation() == UNIT_PICKUP && row.item() == item_index)
            .map(ActionRow::count)
            .sum();
        let active_orders = usize::try_from(*market_length).unwrap_or(0).min(max_orders);
        let mut existing_quantity = 0;
        let mut existing_slot = None;
        for order in 0..active_orders {
            let row = get_market(market, player, order, max_orders);
            if row.operation() == MARKET_SELL && row.item() == item_index {
                existing_quantity += row.count();
                existing_slot.get_or_insert(order);
            }
        }
        let quantity =
            target.min((farm.private.shed[item_index] - pickup_reserve - existing_quantity).max(0));
        if quantity <= 0 {
            continue;
        }
        if let Some(order) = existing_slot {
            let mut row = get_market(market, player, order, max_orders);
            row.0[2] += quantity;
            set_market(market, player, order, max_orders, row);
        } else if active_orders < max_orders {
            set_market(
                market,
                player,
                active_orders,
                max_orders,
                ActionRow([
                    MARKET_SELL,
                    i64::try_from(item_index).unwrap_or(i64::MAX),
                    quantity,
                ]),
            );
            *market_length += 1;
        } else {
            continue;
        }
        policy.due[item_index] = quantity;
        moved_any = true;
    }
    if moved_any {
        policy.due_step = step + 1;
    }
}

fn repay_due(
    policy: &mut PlayerState,
    player: usize,
    max_orders: usize,
    market: &mut [i16],
    market_length: &mut i16,
) {
    let active_orders = usize::try_from(*market_length).unwrap_or(0).min(max_orders);
    for item in FRONT_RUN_ITEMS {
        let item_index = item.index();
        let mut remaining = policy.due[item_index];
        for order in 0..active_orders {
            if remaining <= 0 {
                break;
            }
            let mut row = get_market(market, player, order, max_orders);
            if row.operation() == MARKET_SELL && row.item() == item_index {
                let reduction = row.count().min(remaining);
                row.0[2] -= reduction;
                remaining -= reduction;
                if row.count() <= 0 {
                    row = ActionRow::default();
                }
                set_market(market, player, order, max_orders, row);
            }
        }
    }
    compact_market(player, max_orders, market, market_length);
    policy.due_step = -1;
    policy.due.fill(0);
}

fn product_demand(state: &State, item: Item) -> i64 {
    let Some(product) = item.as_product() else {
        return 0;
    };
    let mut demand = i64::from(state.step % 24 == 0 && product != Product::Fertilizer);
    if state.step % 4 == 0 {
        for &shop in &state.town.unlocked_shops {
            let products = shop_products(shop);
            if products.contains(&product) {
                demand += if products.len() == 1 { 2 } else { 1 };
            }
        }
    }
    demand
}

fn compact_market(player: usize, max_orders: usize, market: &mut [i16], market_length: &mut i16) {
    let active_orders = usize::try_from(*market_length).unwrap_or(0).min(max_orders);
    let mut write = 0;
    for read in 0..active_orders {
        let row = get_market(market, player, read, max_orders);
        if row.operation() != MARKET_NONE {
            if write != read {
                set_market(market, player, write, max_orders, row);
            }
            write += 1;
        }
    }
    for order in write..active_orders {
        set_market(market, player, order, max_orders, ActionRow::default());
    }
    *market_length = compact_value(i64::try_from(write).unwrap_or(i64::MAX));
}

fn unit_offset(player: usize, unit: usize, max_units: usize) -> usize {
    (player * max_units + unit) * ACTION_FIELD_COUNT
}

fn market_offset(player: usize, order: usize, max_orders: usize) -> usize {
    (player * max_orders + order) * ACTION_FIELD_COUNT
}

fn get_unit(values: &[i16], player: usize, unit: usize, max_units: usize) -> ActionRow {
    let offset = unit_offset(player, unit, max_units);
    ActionRow([
        i64::from(values[offset]),
        i64::from(values[offset + 1]),
        i64::from(values[offset + 2]),
    ])
}

fn set_unit(values: &mut [i16], player: usize, unit: usize, max_units: usize, row: ActionRow) {
    let offset = unit_offset(player, unit, max_units);
    for (output, value) in values[offset..offset + ACTION_FIELD_COUNT]
        .iter_mut()
        .zip(row.0)
    {
        *output = compact_value(value);
    }
}

fn get_market(values: &[i16], player: usize, order: usize, max_orders: usize) -> ActionRow {
    let offset = market_offset(player, order, max_orders);
    ActionRow([
        i64::from(values[offset]),
        i64::from(values[offset + 1]),
        i64::from(values[offset + 2]),
    ])
}

fn set_market(values: &mut [i16], player: usize, order: usize, max_orders: usize, row: ActionRow) {
    let offset = market_offset(player, order, max_orders);
    for (output, value) in values[offset..offset + ACTION_FIELD_COUNT]
        .iter_mut()
        .zip(row.0)
    {
        *output = compact_value(value);
    }
}

fn compact_value(value: i64) -> i16 {
    match i16::try_from(value) {
        Ok(value) => value,
        Err(_) if value < 0 => i16::MIN,
        Err(_) => i16::MAX,
    }
}

fn require_shape(name: &str, actual: &[usize], expected: &[usize]) -> PyResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} has shape {actual:?}; expected {expected:?}"
        )))
    }
}

fn require_c_order(name: &str, contiguous: bool) -> PyResult<()> {
    if contiguous {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be a C-contiguous NumPy array"
        )))
    }
}
