//! Batched simulator ownership and the NumPy/PyO3 boundary.

use kaggriculture_core::{Action, Config, Crop, Item, Product, Sim, State, Structure, Tile};
use numpy::{PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::action::{
    ACTION_FIELD_COUNT, ITEM_COUNT, MARKET_ACTION_COUNT, PLAYER_COUNT, UNIT_ACTION_COUNT,
    decode_actions,
};
use crate::encoding::{
    FARM_CHANNELS, GLOBAL_CHANNELS, MaskSpec, ObservationSpec, PRIVATE_CHANNELS, TILE_CHANNELS,
    UNIT_CHANNELS, encode,
};
use crate::snapshot::state_snapshot;

#[derive(Clone, Debug)]
struct TerminalRecord {
    state: State,
    seed: u64,
    episode_id: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct Slot {
    pub(crate) sim: Sim,
    base_seed: u64,
    pub(crate) episode_id: u64,
    terminal: Option<TerminalRecord>,
}

/// Rust-owned batch of independent Kaggriculture simulations.
#[derive(Debug)]
pub(crate) struct VecEnvCore {
    slots: Vec<Slot>,
    observation_spec: ObservationSpec,
    mask_spec: MaskSpec,
    max_units: usize,
    max_orders: usize,
    auto_reset: bool,
}

struct OutputBuffers<'a> {
    observations: &'a mut [f32],
    action_masks: &'a mut [u8],
    unit_active: &'a mut [u8],
    rewards: &'a mut [f64],
    dones: &'a mut [u8],
    episode_ids: &'a mut [u64],
    overflows: &'a mut [u8],
}

impl VecEnvCore {
    fn new(
        num_envs: usize,
        base_config: &Config,
        seed: u64,
        max_units: usize,
        auto_reset: bool,
    ) -> Result<Self, String> {
        if num_envs == 0 {
            return Err("num_envs must be at least 1".to_owned());
        }
        validate_config(base_config)?;

        let episode_transitions = base_config.episode_steps.saturating_sub(1).max(1);
        let observable_hire_rounds = base_config
            .turns_per_day
            .saturating_sub(1)
            .min(episode_transitions);
        let exact_unit_bound = usize::try_from(observable_hire_rounds)
            .map_err(|_| "turns_per_day does not fit usize")?
            .checked_mul(base_config.max_market_orders_per_turn)
            .and_then(|value| value.checked_add(1))
            .ok_or("the unit-slot bound overflows usize")?;
        let max_units = if max_units == 0 {
            exact_unit_bound
        } else {
            max_units
        };
        if max_units < exact_unit_bound {
            return Err(format!(
                "max_units={max_units} is too small; this configuration can require {exact_unit_bound} slots"
            ));
        }
        validate_tensor_dimensions(base_config.board_size, max_units)?;

        let observation_spec = ObservationSpec::new(base_config.board_size, max_units);
        let mask_spec = MaskSpec::new(max_units);
        let max_orders = base_config.max_market_orders_per_turn;
        let slots = (0..num_envs)
            .map(|index| {
                let index = u64::try_from(index).map_err(|_| "environment index exceeds u64")?;
                let slot_seed = seed.wrapping_add(index);
                let mut config = base_config.clone();
                config.seed = slot_seed;
                Ok(Slot {
                    sim: Sim::new(config),
                    base_seed: slot_seed,
                    episode_id: 0,
                    terminal: None,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        Ok(Self {
            slots,
            observation_spec,
            mask_spec,
            max_units,
            max_orders,
            auto_reset,
        })
    }

    fn num_envs(&self) -> usize {
        self.slots.len()
    }

    fn v9_fingerprints(&self, seats: &[i64], output: &mut [u64]) -> Result<(), String> {
        require_len("seats", seats.len(), self.num_envs())?;
        require_len("fingerprints", output.len(), self.num_envs() * 6)?;
        if let Some((index, seat)) = seats
            .iter()
            .copied()
            .enumerate()
            .find(|(_, seat)| !matches!(seat, 0 | 1))
        {
            return Err(format!("seat {index} is {seat}; expected 0 or 1"));
        }

        self.slots
            .par_iter()
            .zip(seats.par_iter().copied())
            .zip(output.par_chunks_mut(6))
            .for_each(|((slot, seat), row)| {
                let seat = usize::try_from(seat).unwrap_or_default();
                let (state_a, state_b) = v9_state_fingerprint(&slot.sim.state, seat);
                let (town_a, town_b) = v9_town_fingerprint(&slot.sim.state);
                row.copy_from_slice(&[
                    state_a,
                    state_b,
                    town_a,
                    town_b,
                    u64::from(slot.sim.state.step),
                    (1 + slot.sim.state.farms[seat].hands.len()) as u64,
                ]);
            });
        Ok(())
    }

    fn validate_output_lengths(&self, output: &OutputBuffers<'_>) -> Result<(), String> {
        let n = self.num_envs();
        require_len(
            "observations",
            output.observations.len(),
            checked_product(&[n, PLAYER_COUNT, self.observation_spec.total])?,
        )?;
        require_len(
            "action_masks",
            output.action_masks.len(),
            checked_product(&[n, PLAYER_COUNT, self.mask_spec.total])?,
        )?;
        require_len(
            "unit_active",
            output.unit_active.len(),
            checked_product(&[n, PLAYER_COUNT, self.max_units])?,
        )?;
        require_len("rewards", output.rewards.len(), n * PLAYER_COUNT)?;
        require_len("dones", output.dones.len(), n * PLAYER_COUNT)?;
        require_len("episode_ids", output.episode_ids.len(), n)?;
        require_len("overflows", output.overflows.len(), n * PLAYER_COUNT)?;
        Ok(())
    }

    #[allow(clippy::needless_pass_by_value)]
    fn reset_into(
        &mut self,
        seeds: Option<&[u64]>,
        output: OutputBuffers<'_>,
    ) -> Result<(), String> {
        self.validate_output_lengths(&output)?;
        if let Some(seeds) = seeds {
            require_len("seeds", seeds.len(), self.num_envs())?;
        }

        for (index, slot) in self.slots.iter_mut().enumerate() {
            if let Some(seeds) = seeds {
                slot.base_seed = seeds[index];
            }
            slot.episode_id = 0;
            slot.terminal = None;
            slot.sim.config.seed = slot.base_seed;
            slot.sim.reset();
        }

        self.encode_all(output, false)
    }

    #[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
    fn step_into(
        &mut self,
        unit_actions: &[i64],
        market_actions: &[i64],
        market_lengths: &[i64],
        output: OutputBuffers<'_>,
    ) -> Result<(), String> {
        self.validate_output_lengths(&output)?;
        let unit_env_len = checked_product(&[PLAYER_COUNT, self.max_units, ACTION_FIELD_COUNT])?;
        let market_env_len = checked_product(&[PLAYER_COUNT, self.max_orders, ACTION_FIELD_COUNT])?;
        require_len(
            "unit_actions",
            unit_actions.len(),
            checked_product(&[self.num_envs(), unit_env_len])?,
        )?;
        require_len(
            "market_actions",
            market_actions.len(),
            checked_product(&[self.num_envs(), market_env_len])?,
        )?;
        require_len(
            "market_lengths",
            market_lengths.len(),
            checked_product(&[self.num_envs(), PLAYER_COUNT])?,
        )?;
        if let Some((index, _)) = self
            .slots
            .iter()
            .enumerate()
            .find(|(_, slot)| slot.sim.state.done)
        {
            return Err(format!(
                "environment {index} is terminal; call reset before stepping again"
            ));
        }

        // Decode every row before mutating any simulator. A malformed batch is
        // therefore transactional: callers can correct it and retry safely.
        let decoded = self
            .slots
            .iter()
            .enumerate()
            .map(|(index, slot)| {
                let unit_start = index * unit_env_len;
                let market_start = index * market_env_len;
                let length_start = index * PLAYER_COUNT;
                let lengths = [
                    decode_length(market_lengths[length_start], index, 0)?,
                    decode_length(market_lengths[length_start + 1], index, 1)?,
                ];
                decode_actions(
                    &unit_actions[unit_start..unit_start + unit_env_len],
                    &market_actions[market_start..market_start + market_env_len],
                    self.max_units,
                    self.max_orders,
                    [
                        slot.sim.state.farms[0].hands.len(),
                        slot.sim.state.farms[1].hands.len(),
                    ],
                    lengths,
                )
                .map_err(|error| format!("environment {index}: {error}"))
            })
            .collect::<Result<Vec<[Action; PLAYER_COUNT]>, String>>()?;

        let observation_env_len = PLAYER_COUNT * self.observation_spec.total;
        let mask_env_len = PLAYER_COUNT * self.mask_spec.total;
        let active_env_len = PLAYER_COUNT * self.max_units;
        let observation_spec = self.observation_spec;
        let mask_spec = self.mask_spec;
        let auto_reset = self.auto_reset;
        let seed_stride = u64::try_from(self.num_envs())
            .map_err(|_| "num_envs does not fit in the u64 seed schedule")?;

        self.slots
            .par_iter_mut()
            .zip(decoded.into_par_iter())
            .zip(output.observations.par_chunks_mut(observation_env_len))
            .zip(output.action_masks.par_chunks_mut(mask_env_len))
            .zip(output.unit_active.par_chunks_mut(active_env_len))
            .zip(output.rewards.par_chunks_mut(PLAYER_COUNT))
            .zip(output.dones.par_chunks_mut(PLAYER_COUNT))
            .zip(output.episode_ids.par_iter_mut())
            .zip(output.overflows.par_chunks_mut(PLAYER_COUNT))
            .try_for_each(
                |(
                    (
                        ((((((slot, actions), observations), masks), active), rewards), dones),
                        episode_id,
                    ),
                    overflows,
                )| {
                    slot.sim.step([&actions[0], &actions[1]]);
                    let done = slot.sim.state.done;
                    rewards[0] = slot.sim.reward(0);
                    rewards[1] = slot.sim.reward(1);
                    dones.fill(u8::from(done));

                    if done {
                        slot.terminal = Some(TerminalRecord {
                            state: slot.sim.state.clone(),
                            seed: slot.sim.config.seed,
                            episode_id: slot.episode_id,
                        });
                        if auto_reset {
                            slot.episode_id = slot.episode_id.wrapping_add(1);
                            slot.sim.config.seed = slot
                                .base_seed
                                .wrapping_add(slot.episode_id.wrapping_mul(seed_stride));
                            slot.sim.reset();
                        }
                    }
                    *episode_id = slot.episode_id;
                    encode_slot(
                        slot,
                        observation_spec,
                        mask_spec,
                        observations,
                        masks,
                        active,
                        overflows,
                    )
                },
            )
    }

    #[allow(clippy::needless_pass_by_value)]
    fn encode_all(
        &self,
        output: OutputBuffers<'_>,
        done_from_transition: bool,
    ) -> Result<(), String> {
        let observation_env_len = PLAYER_COUNT * self.observation_spec.total;
        let mask_env_len = PLAYER_COUNT * self.mask_spec.total;
        let active_env_len = PLAYER_COUNT * self.max_units;
        let observation_spec = self.observation_spec;
        let mask_spec = self.mask_spec;

        self.slots
            .par_iter()
            .zip(output.observations.par_chunks_mut(observation_env_len))
            .zip(output.action_masks.par_chunks_mut(mask_env_len))
            .zip(output.unit_active.par_chunks_mut(active_env_len))
            .zip(output.rewards.par_chunks_mut(PLAYER_COUNT))
            .zip(output.dones.par_chunks_mut(PLAYER_COUNT))
            .zip(output.episode_ids.par_iter_mut())
            .zip(output.overflows.par_chunks_mut(PLAYER_COUNT))
            .try_for_each(
                |(
                    ((((((slot, observations), masks), active), rewards), dones), episode_id),
                    overflows,
                )| {
                    rewards.fill(0.0);
                    dones.fill(u8::from(done_from_transition && slot.sim.state.done));
                    *episode_id = slot.episode_id;
                    encode_slot(
                        slot,
                        observation_spec,
                        mask_spec,
                        observations,
                        masks,
                        active,
                        overflows,
                    )
                },
            )
    }
}

fn encode_slot(
    slot: &Slot,
    observation_spec: ObservationSpec,
    mask_spec: MaskSpec,
    observations: &mut [f32],
    masks: &mut [u8],
    active: &mut [u8],
    overflows: &mut [u8],
) -> Result<(), String> {
    for (player, overflow_output) in overflows.iter_mut().enumerate() {
        let obs_start = player * observation_spec.total;
        let mask_start = player * mask_spec.total;
        let active_start = player * observation_spec.max_units;
        let mut overflow = false;
        encode(
            &slot.sim,
            player,
            &mut observations[obs_start..obs_start + observation_spec.total],
            &mut masks[mask_start..mask_start + mask_spec.total],
            &mut active[active_start..active_start + observation_spec.max_units],
            &mut overflow,
        )?;
        *overflow_output = u8::from(overflow);
        if overflow {
            return Err(format!(
                "player {player} has more units than max_units={} permits",
                observation_spec.max_units
            ));
        }
    }
    Ok(())
}

fn decode_length(value: i64, environment: usize, player: usize) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| {
        format!("market_lengths[{environment}, {player}] must be a nonnegative integer")
    })
}

fn validate_config(config: &Config) -> Result<(), String> {
    if config.board_size < 4 {
        return Err("board_size must be at least 4".to_owned());
    }
    if config.episode_steps == 0 {
        return Err("episode_steps must be at least 1".to_owned());
    }
    if config.starting_money < 0 {
        return Err("starting_money cannot be negative".to_owned());
    }
    if config.turns_per_day == 0 {
        return Err("turns_per_day must be positive".to_owned());
    }
    if config.max_market_orders_per_turn == 0 {
        return Err("max_market_orders must be positive".to_owned());
    }
    if config.shed_capacity <= 0 {
        return Err("shed_capacity must be positive".to_owned());
    }
    if config.town_shop_unlock_interval == 0
        || config.town_shop_sell_interval == 0
        || config.town_center_sell_interval == 0
    {
        return Err("town intervals must be positive".to_owned());
    }
    if config.farm_hand_cost_multiplier < 0 {
        return Err("farm_hand_cost_multiplier cannot be negative".to_owned());
    }
    if !config.weed_spawn_chance.is_finite() || config.weed_spawn_chance < 0.0 {
        return Err("weed_spawn_chance must be finite and nonnegative".to_owned());
    }
    Ok(())
}

fn checked_product(values: &[usize]) -> Result<usize, String> {
    values
        .iter()
        .try_fold(1_usize, |product, value| product.checked_mul(*value))
        .ok_or_else(|| "buffer dimensions overflow usize".to_owned())
}

fn validate_tensor_dimensions(board_size: usize, max_units: usize) -> Result<(), String> {
    let farm_values = checked_product(&[PLAYER_COUNT, FARM_CHANNELS])?;
    let tile_values = checked_product(&[PLAYER_COUNT, board_size, board_size, TILE_CHANNELS])?;
    let unit_values = checked_product(&[PLAYER_COUNT, max_units, UNIT_CHANNELS])?;
    [
        GLOBAL_CHANNELS,
        farm_values,
        tile_values,
        unit_values,
        PRIVATE_CHANNELS,
    ]
    .into_iter()
    .try_fold(0_usize, usize::checked_add)
    .ok_or_else(|| "observation dimensions overflow usize".to_owned())?;

    let unit_operations = checked_product(&[max_units, UNIT_ACTION_COUNT])?;
    let unit_arguments = checked_product(&[unit_operations, ITEM_COUNT])?;
    let market_arguments = checked_product(&[MARKET_ACTION_COUNT, ITEM_COUNT])?;
    [
        unit_operations,
        unit_arguments,
        MARKET_ACTION_COUNT,
        market_arguments,
    ]
    .into_iter()
    .try_fold(0_usize, usize::checked_add)
    .ok_or_else(|| "action-mask dimensions overflow usize".to_owned())?;
    Ok(())
}

// Two independently-seeded SplitMix-style accumulators make collisions
// negligible while keeping this hot path allocation-free. The fingerprint is
// follows V9's feature tuple field-for-field, avoiding hundreds of Python dict
// and string allocations per simulator slot.
#[derive(Clone, Copy)]
struct Fingerprint(u64, u64);

impl Fingerprint {
    const fn new() -> Self {
        Self(0x243f_6a88_85a3_08d3, 0x1319_8a2e_0370_7344)
    }

    fn push(&mut self, value: u64) {
        self.0 = fingerprint_mix(self.0 ^ value);
        self.1 = fingerprint_mix(self.1 ^ value.rotate_left(29));
    }

    const fn finish(self) -> (u64, u64) {
        (self.0, self.1)
    }
}

fn fingerprint_mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

const fn signed_bits(value: i64) -> u64 {
    u64::from_ne_bytes(value.to_ne_bytes())
}

fn v9_town_fingerprint(state: &State) -> (u64, u64) {
    let mut hash = Fingerprint::new();
    hash.push(state.town.unlocked_shops.len() as u64);
    for &shop in &state.town.unlocked_shops {
        hash.push(shop as u64);
    }
    hash.finish()
}

#[allow(clippy::cast_possible_truncation)]
fn v9_state_fingerprint(state: &State, seat: usize) -> (u64, u64) {
    const V9_ITEMS: [Item; 12] = [
        Item::Carrot,
        Item::Cow,
        Item::Egg,
        Item::Fertilizer,
        Item::Goose,
        Item::Melon,
        Item::Milk,
        Item::Sheep,
        Item::Strawberry,
        Item::Tomato,
        Item::Wheat,
        Item::Wool,
    ];
    const V9_SEEDS: [Crop; 5] = [
        Crop::Carrot,
        Crop::Melon,
        Crop::Strawberry,
        Crop::Tomato,
        Crop::Wheat,
    ];
    const V9_PRICES: [Product; 9] = [
        Product::Carrot,
        Product::Egg,
        Product::Fertilizer,
        Product::Melon,
        Product::Milk,
        Product::Strawberry,
        Product::Tomato,
        Product::Wheat,
        Product::Wool,
    ];

    let mut hash = Fingerprint::new();
    let farm = &state.farms[seat];
    hash.push(signed_bits(farm.money.round_ties_even() as i64));
    hash.push((farm.farmer.x * 10 + farm.farmer.y) as u64);
    hash.push(farm.hands.len() as u64);
    for hand in &farm.hands {
        hash.push((hand.x * 10 + hand.y) as u64);
    }

    hash.push(farm.private.inventories.len() as u64);
    for inventory in &farm.private.inventories {
        for item in V9_ITEMS {
            hash.push(signed_bits(inventory.get(item)));
        }
    }
    for crop in V9_SEEDS {
        hash.push(signed_bits(farm.private.seeds[crop.index()]));
    }
    for item in V9_ITEMS {
        hash.push(signed_bits(farm.private.shed[item.index()]));
    }

    for tile in &farm.tiles {
        match tile {
            Tile::Empty => hash.push(0),
            Tile::Locked => hash.push(1),
            Tile::Weed => hash.push(2),
            Tile::Plant(plant) => {
                hash.push(3);
                hash.push(plant.crop as u64);
                hash.push(signed_bits(i64::from(plant.planted_day)));
                hash.push(u64::from(plant.watered_today));
                hash.push(signed_bits(i64::from(plant.consecutive_unwatered)));
                hash.push(signed_bits(plant.yield_units));
                hash.push(signed_bits(plant.max_lifespan_step));
                hash.push(signed_bits(i64::from(plant.fertilized_until_day)));
            }
            Tile::Structure { kind, animal } => match kind {
                // V9's tile signature intentionally treats every coop as the
                // same "COOP" token, even when it contains a goose.
                Structure::Coop => hash.push(4),
                Structure::Pasture => {
                    hash.push(5);
                    match animal {
                        None => hash.push(u64::MAX),
                        Some(animal) => {
                            hash.push(animal.animal as u64);
                            hash.push(signed_bits(animal.yield_units));
                            hash.push(signed_bits(i64::from(animal.consecutive_unfed)));
                            hash.push(u64::from(animal.fed_today));
                            hash.push(u64::from(animal.cared_today));
                            hash.push(u64::from(animal.fertilizer_available));
                            hash.push(signed_bits(animal.pending_care_bonus));
                        }
                    }
                }
            },
        }
    }

    for product in V9_PRICES {
        hash.push(signed_bits(state.market.prices[product.index()] as i64));
    }
    hash.push(farm.unlocked_quadrants.len() as u64);
    hash.finish()
}

fn require_len(name: &str, actual: usize, expected: usize) -> Result<(), String> {
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "{name} has {actual} elements; expected exactly {expected}"
        ))
    }
}

/// Private native class. [`bertani.vec_env.VecEnv`](../../../src/bertani/vec_env.py)
/// owns the `NumPy` buffers and presents the ergonomic public API.
#[pyclass(module = "bertani._rust")]
pub struct NativeVecEnv {
    core: VecEnvCore,
}

impl NativeVecEnv {
    pub(crate) fn slots(&self) -> &[Slot] {
        &self.core.slots
    }

    pub(crate) const fn max_units_native(&self) -> usize {
        self.core.max_units
    }

    pub(crate) const fn max_orders_native(&self) -> usize {
        self.core.max_orders
    }
}

#[pymethods]
impl NativeVecEnv {
    #[new]
    #[pyo3(signature = (
        num_envs,
        seed=0,
        max_units=0,
        auto_reset=true,
        episode_steps=720,
        board_size=10,
        starting_money=3_000,
        max_market_orders=10,
        turns_per_day=24,
        shed_capacity=100,
        weed_spawn_chance=0.005,
        town_shop_unlock_interval=3,
        town_shop_sell_interval=4,
        town_center_sell_interval=24,
        farm_hand_cost_multiplier=1
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_envs: usize,
        seed: u64,
        max_units: usize,
        auto_reset: bool,
        episode_steps: u32,
        board_size: usize,
        starting_money: i64,
        max_market_orders: usize,
        turns_per_day: u32,
        shed_capacity: i64,
        weed_spawn_chance: f64,
        town_shop_unlock_interval: u32,
        town_shop_sell_interval: u32,
        town_center_sell_interval: u32,
        farm_hand_cost_multiplier: i64,
    ) -> PyResult<Self> {
        let config = Config {
            episode_steps,
            board_size,
            starting_money,
            max_market_orders_per_turn: max_market_orders,
            turns_per_day,
            shed_capacity,
            weed_spawn_chance,
            town_shop_unlock_interval,
            town_shop_sell_interval,
            town_center_sell_interval,
            farm_hand_cost_multiplier,
            seed,
            ..Config::default()
        };
        let core = VecEnvCore::new(num_envs, &config, seed, max_units, auto_reset)
            .map_err(PyValueError::new_err)?;
        Ok(Self { core })
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.core.num_envs()
    }

    #[getter]
    const fn max_units(&self) -> usize {
        self.core.max_units
    }

    #[getter]
    const fn max_orders(&self) -> usize {
        self.core.max_orders
    }

    #[getter]
    const fn observation_size(&self) -> usize {
        self.core.observation_spec.total
    }

    #[getter]
    const fn mask_size(&self) -> usize {
        self.core.mask_spec.total
    }

    #[getter]
    const fn board_size(&self) -> usize {
        self.core.observation_spec.board_size
    }

    #[getter]
    const fn auto_reset(&self) -> bool {
        self.core.auto_reset
    }

    #[setter]
    fn set_auto_reset(&mut self, value: bool) {
        self.core.auto_reset = value;
    }

    fn buffer_specs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let n = self.core.num_envs();
        let observation = self.core.observation_spec;
        let mask = self.core.mask_spec;
        let specs = PyDict::new(py);
        specs.set_item("observation_shape", (n, PLAYER_COUNT, observation.total))?;
        specs.set_item("action_mask_shape", (n, PLAYER_COUNT, mask.total))?;
        specs.set_item("unit_active_shape", (n, PLAYER_COUNT, self.core.max_units))?;
        specs.set_item("reward_shape", (n, PLAYER_COUNT))?;
        specs.set_item("done_shape", (n, PLAYER_COUNT))?;
        specs.set_item("episode_id_shape", (n,))?;
        specs.set_item("overflow_shape", (n, PLAYER_COUNT))?;
        specs.set_item(
            "unit_action_shape",
            (n, PLAYER_COUNT, self.core.max_units, ACTION_FIELD_COUNT),
        )?;
        specs.set_item(
            "market_action_shape",
            (n, PLAYER_COUNT, self.core.max_orders, ACTION_FIELD_COUNT),
        )?;
        specs.set_item("market_length_shape", (n, PLAYER_COUNT))?;
        specs.set_item("observation_dtype", "float32")?;
        specs.set_item("action_mask_dtype", "uint8")?;
        specs.set_item("unit_active_dtype", "uint8")?;
        specs.set_item("reward_dtype", "float64")?;
        specs.set_item("done_dtype", "uint8")?;
        specs.set_item("episode_id_dtype", "uint64")?;
        specs.set_item("overflow_dtype", "uint8")?;
        specs.set_item("unit_action_dtype", "int64")?;
        specs.set_item("market_action_dtype", "int64")?;
        specs.set_item("market_length_dtype", "int64")?;

        specs.set_item("observation_global", observation.global)?;
        specs.set_item("observation_farms", observation.farms)?;
        specs.set_item("observation_tiles", observation.tiles)?;
        specs.set_item("observation_units", observation.units)?;
        specs.set_item("observation_private", observation.private)?;
        specs.set_item("global_channels", GLOBAL_CHANNELS)?;
        specs.set_item("farm_channels", FARM_CHANNELS)?;
        specs.set_item("tile_channels", TILE_CHANNELS)?;
        specs.set_item("unit_channels", UNIT_CHANNELS)?;
        specs.set_item("private_channels", PRIVATE_CHANNELS)?;

        specs.set_item("mask_unit_ops", mask.unit_ops)?;
        specs.set_item("mask_unit_args", mask.unit_args)?;
        specs.set_item("mask_market_ops", mask.market_ops)?;
        specs.set_item("mask_market_args", mask.market_args)?;
        specs.set_item("unit_action_count", UNIT_ACTION_COUNT)?;
        specs.set_item("market_action_count", MARKET_ACTION_COUNT)?;
        specs.set_item("item_count", ITEM_COUNT)?;
        Ok(specs)
    }

    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn reset_into<'py>(
        &mut self,
        seeds: Option<Bound<'py, PyArray1<u64>>>,
        observations: Bound<'py, PyArray3<f32>>,
        action_masks: Bound<'py, PyArray3<u8>>,
        unit_active: Bound<'py, PyArray3<u8>>,
        rewards: Bound<'py, PyArray2<f64>>,
        dones: Bound<'py, PyArray2<u8>>,
        episode_ids: Bound<'py, PyArray1<u64>>,
        overflows: Bound<'py, PyArray2<u8>>,
    ) -> PyResult<()> {
        let n = self.core.num_envs();
        check_shape(
            "observations",
            observations.shape(),
            &[n, PLAYER_COUNT, self.core.observation_spec.total],
        )?;
        check_shape(
            "action_masks",
            action_masks.shape(),
            &[n, PLAYER_COUNT, self.core.mask_spec.total],
        )?;
        check_shape(
            "unit_active",
            unit_active.shape(),
            &[n, PLAYER_COUNT, self.core.max_units],
        )?;
        check_shape("rewards", rewards.shape(), &[n, PLAYER_COUNT])?;
        check_shape("dones", dones.shape(), &[n, PLAYER_COUNT])?;
        check_shape("episode_ids", episode_ids.shape(), &[n])?;
        check_shape("overflows", overflows.shape(), &[n, PLAYER_COUNT])?;
        check_c_order("observations", observations.is_c_contiguous())?;
        check_c_order("action_masks", action_masks.is_c_contiguous())?;
        check_c_order("unit_active", unit_active.is_c_contiguous())?;
        check_c_order("rewards", rewards.is_c_contiguous())?;
        check_c_order("dones", dones.is_c_contiguous())?;
        check_c_order("episode_ids", episode_ids.is_c_contiguous())?;
        check_c_order("overflows", overflows.is_c_contiguous())?;
        if let Some(seeds) = &seeds {
            check_shape("seeds", seeds.shape(), &[n])?;
            check_c_order("seeds", seeds.is_c_contiguous())?;
        }

        let seeds = seeds
            .as_ref()
            .map(|array| borrow_array("seeds", array.try_readonly()))
            .transpose()?;
        let mut observations = borrow_array("observations", observations.try_readwrite())?;
        let mut action_masks = borrow_array("action_masks", action_masks.try_readwrite())?;
        let mut unit_active = borrow_array("unit_active", unit_active.try_readwrite())?;
        let mut rewards = borrow_array("rewards", rewards.try_readwrite())?;
        let mut dones = borrow_array("dones", dones.try_readwrite())?;
        let mut episode_ids = borrow_array("episode_ids", episode_ids.try_readwrite())?;
        let mut overflows = borrow_array("overflows", overflows.try_readwrite())?;
        let seed_values = seeds
            .as_ref()
            .map(|array| contiguous_read("seeds", array.as_slice()))
            .transpose()?;
        let output = OutputBuffers {
            observations: contiguous_write("observations", observations.as_slice_mut())?,
            action_masks: contiguous_write("action_masks", action_masks.as_slice_mut())?,
            unit_active: contiguous_write("unit_active", unit_active.as_slice_mut())?,
            rewards: contiguous_write("rewards", rewards.as_slice_mut())?,
            dones: contiguous_write("dones", dones.as_slice_mut())?,
            episode_ids: contiguous_write("episode_ids", episode_ids.as_slice_mut())?,
            overflows: contiguous_write("overflows", overflows.as_slice_mut())?,
        };
        self.core
            .reset_into(seed_values, output)
            .map_err(PyValueError::new_err)
    }

    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn step_into<'py>(
        &mut self,
        unit_actions: Bound<'py, PyArray4<i64>>,
        market_actions: Bound<'py, PyArray4<i64>>,
        market_lengths: Bound<'py, PyArray2<i64>>,
        observations: Bound<'py, PyArray3<f32>>,
        action_masks: Bound<'py, PyArray3<u8>>,
        unit_active: Bound<'py, PyArray3<u8>>,
        rewards: Bound<'py, PyArray2<f64>>,
        dones: Bound<'py, PyArray2<u8>>,
        episode_ids: Bound<'py, PyArray1<u64>>,
        overflows: Bound<'py, PyArray2<u8>>,
    ) -> PyResult<()> {
        let n = self.core.num_envs();
        check_shape(
            "unit_actions",
            unit_actions.shape(),
            &[n, PLAYER_COUNT, self.core.max_units, ACTION_FIELD_COUNT],
        )?;
        check_shape(
            "market_actions",
            market_actions.shape(),
            &[n, PLAYER_COUNT, self.core.max_orders, ACTION_FIELD_COUNT],
        )?;
        check_shape("market_lengths", market_lengths.shape(), &[n, PLAYER_COUNT])?;
        check_shape(
            "observations",
            observations.shape(),
            &[n, PLAYER_COUNT, self.core.observation_spec.total],
        )?;
        check_shape(
            "action_masks",
            action_masks.shape(),
            &[n, PLAYER_COUNT, self.core.mask_spec.total],
        )?;
        check_shape(
            "unit_active",
            unit_active.shape(),
            &[n, PLAYER_COUNT, self.core.max_units],
        )?;
        check_shape("rewards", rewards.shape(), &[n, PLAYER_COUNT])?;
        check_shape("dones", dones.shape(), &[n, PLAYER_COUNT])?;
        check_shape("episode_ids", episode_ids.shape(), &[n])?;
        check_shape("overflows", overflows.shape(), &[n, PLAYER_COUNT])?;
        check_c_order("unit_actions", unit_actions.is_c_contiguous())?;
        check_c_order("market_actions", market_actions.is_c_contiguous())?;
        check_c_order("market_lengths", market_lengths.is_c_contiguous())?;
        check_c_order("observations", observations.is_c_contiguous())?;
        check_c_order("action_masks", action_masks.is_c_contiguous())?;
        check_c_order("unit_active", unit_active.is_c_contiguous())?;
        check_c_order("rewards", rewards.is_c_contiguous())?;
        check_c_order("dones", dones.is_c_contiguous())?;
        check_c_order("episode_ids", episode_ids.is_c_contiguous())?;
        check_c_order("overflows", overflows.is_c_contiguous())?;

        let unit_actions = borrow_array("unit_actions", unit_actions.try_readonly())?;
        let market_actions = borrow_array("market_actions", market_actions.try_readonly())?;
        let market_lengths = borrow_array("market_lengths", market_lengths.try_readonly())?;
        let mut observations = borrow_array("observations", observations.try_readwrite())?;
        let mut action_masks = borrow_array("action_masks", action_masks.try_readwrite())?;
        let mut unit_active = borrow_array("unit_active", unit_active.try_readwrite())?;
        let mut rewards = borrow_array("rewards", rewards.try_readwrite())?;
        let mut dones = borrow_array("dones", dones.try_readwrite())?;
        let mut episode_ids = borrow_array("episode_ids", episode_ids.try_readwrite())?;
        let mut overflows = borrow_array("overflows", overflows.try_readwrite())?;
        let unit_values = contiguous_read("unit_actions", unit_actions.as_slice())?;
        let market_values = contiguous_read("market_actions", market_actions.as_slice())?;
        let length_values = contiguous_read("market_lengths", market_lengths.as_slice())?;
        let output = OutputBuffers {
            observations: contiguous_write("observations", observations.as_slice_mut())?,
            action_masks: contiguous_write("action_masks", action_masks.as_slice_mut())?,
            unit_active: contiguous_write("unit_active", unit_active.as_slice_mut())?,
            rewards: contiguous_write("rewards", rewards.as_slice_mut())?,
            dones: contiguous_write("dones", dones.as_slice_mut())?,
            episode_ids: contiguous_write("episode_ids", episode_ids.as_slice_mut())?,
            overflows: contiguous_write("overflows", overflows.as_slice_mut())?,
        };
        self.core
            .step_into(unit_values, market_values, length_values, output)
            .map_err(PyValueError::new_err)
    }

    fn state_snapshot<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyDict>> {
        let slot = self.core.slots.get(index).ok_or_else(|| {
            PyIndexError::new_err(format!("environment index {index} is out of range"))
        })?;
        state_snapshot(py, &slot.sim.state, slot.sim.config.seed, slot.episode_id)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn v9_fingerprints<'py>(
        &self,
        seats: Bound<'py, PyArray1<i64>>,
        output: Bound<'py, PyArray2<u64>>,
    ) -> PyResult<()> {
        let n = self.core.num_envs();
        check_shape("seats", seats.shape(), &[n])?;
        check_shape("output", output.shape(), &[n, 6])?;
        check_c_order("seats", seats.is_c_contiguous())?;
        check_c_order("output", output.is_c_contiguous())?;
        let seats = borrow_array("seats", seats.try_readonly())?;
        let mut output = borrow_array("output", output.try_readwrite())?;
        self.core
            .v9_fingerprints(
                contiguous_read("seats", seats.as_slice())?,
                contiguous_write("output", output.as_slice_mut())?,
            )
            .map_err(PyValueError::new_err)
    }

    fn terminal_snapshot<'py>(
        &self,
        py: Python<'py>,
        index: usize,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let slot = self.core.slots.get(index).ok_or_else(|| {
            PyIndexError::new_err(format!("environment index {index} is out of range"))
        })?;
        slot.terminal
            .as_ref()
            .map(|terminal| state_snapshot(py, &terminal.state, terminal.seed, terminal.episode_id))
            .transpose()
    }
}

fn check_shape(name: &str, actual: &[usize], expected: &[usize]) -> PyResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} has shape {actual:?}; expected {expected:?}"
        )))
    }
}

fn check_c_order(name: &str, is_c_contiguous: bool) -> PyResult<()> {
    if is_c_contiguous {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be a C-contiguous NumPy array"
        )))
    }
}

fn borrow_array<T>(name: &str, result: Result<T, impl std::fmt::Display>) -> PyResult<T> {
    result.map_err(|error| {
        PyValueError::new_err(format!(
            "could not borrow {name}; output arrays must be writable and arrays must not overlap: {error}"
        ))
    })
}

fn contiguous_read<'a, T>(
    name: &str,
    result: Result<&'a [T], impl std::fmt::Display>,
) -> PyResult<&'a [T]> {
    result.map_err(|error| {
        PyValueError::new_err(format!(
            "{name} must be a C-contiguous NumPy array: {error}"
        ))
    })
}

fn contiguous_write<'a, T>(
    name: &str,
    result: Result<&'a mut [T], impl std::fmt::Display>,
) -> PyResult<&'a mut [T]> {
    result.map_err(|error| {
        PyValueError::new_err(format!(
            "{name} must be a writable C-contiguous NumPy array: {error}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{ITEM_CARROT, MARKET_BUY_SEED};

    struct TestBuffers {
        observations: Vec<f32>,
        action_masks: Vec<u8>,
        unit_active: Vec<u8>,
        rewards: Vec<f64>,
        dones: Vec<u8>,
        episode_ids: Vec<u64>,
        overflows: Vec<u8>,
    }

    impl TestBuffers {
        fn new(core: &VecEnvCore) -> Self {
            let n = core.num_envs();
            Self {
                observations: vec![f32::NAN; n * PLAYER_COUNT * core.observation_spec.total],
                action_masks: vec![0; n * PLAYER_COUNT * core.mask_spec.total],
                unit_active: vec![0; n * PLAYER_COUNT * core.max_units],
                rewards: vec![f64::NAN; n * PLAYER_COUNT],
                dones: vec![1; n * PLAYER_COUNT],
                episode_ids: vec![u64::MAX; n],
                overflows: vec![1; n * PLAYER_COUNT],
            }
        }

        fn output(&mut self) -> OutputBuffers<'_> {
            OutputBuffers {
                observations: &mut self.observations,
                action_masks: &mut self.action_masks,
                unit_active: &mut self.unit_active,
                rewards: &mut self.rewards,
                dones: &mut self.dones,
                episode_ids: &mut self.episode_ids,
                overflows: &mut self.overflows,
            }
        }
    }

    #[test]
    fn batch_matches_independent_sims() {
        let config = Config {
            weed_spawn_chance: 0.0,
            ..Config::default()
        };
        let mut vector = VecEnvCore::new(3, &config, 20, 0, false).unwrap();
        let mut direct = (0..3)
            .map(|index| {
                let mut config = config.clone();
                config.seed = 20 + index;
                Sim::new(config)
            })
            .collect::<Vec<_>>();
        let mut storage = TestBuffers::new(&vector);
        vector.reset_into(None, storage.output()).unwrap();

        let units = vec![0_i64; 3 * PLAYER_COUNT * vector.max_units * ACTION_FIELD_COUNT];
        let mut market = vec![0_i64; 3 * PLAYER_COUNT * vector.max_orders * ACTION_FIELD_COUNT];
        let mut lengths = vec![0_i64; 3 * PLAYER_COUNT];
        for env in 0..3 {
            let market_index = (env * PLAYER_COUNT * vector.max_orders) * ACTION_FIELD_COUNT;
            market[market_index] = MARKET_BUY_SEED;
            market[market_index + 1] = ITEM_CARROT;
            market[market_index + 2] = i64::try_from(env + 1).unwrap();
            lengths[env * PLAYER_COUNT] = 1;

            let action = Action {
                market: vec![kaggriculture_core::MarketOrder::BuySeed {
                    crop: kaggriculture_core::Crop::Carrot,
                    count: i64::try_from(env + 1).unwrap(),
                }],
                ..Action::default()
            };
            let pass = Action::pass();
            direct[env].step([&action, &pass]);
        }
        vector
            .step_into(&units, &market, &lengths, storage.output())
            .unwrap();

        for (slot, expected) in vector.slots.iter().zip(direct) {
            assert_eq!(slot.sim.state, expected.state);
        }
        // Poisoned output buffers were completely overwritten.
        assert!(storage.observations.iter().all(|value| value.is_finite()));
        assert!(storage.rewards.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn terminal_values_are_saved_before_auto_reset() {
        let config = Config {
            episode_steps: 2,
            weed_spawn_chance: 0.0,
            ..Config::default()
        };
        let mut vector = VecEnvCore::new(1, &config, 7, 0, true).unwrap();
        let mut storage = TestBuffers::new(&vector);
        vector.reset_into(None, storage.output()).unwrap();

        let units = vec![0_i64; PLAYER_COUNT * vector.max_units * ACTION_FIELD_COUNT];
        let mut market = vec![0_i64; PLAYER_COUNT * vector.max_orders * ACTION_FIELD_COUNT];
        market[0] = MARKET_BUY_SEED;
        market[1] = ITEM_CARROT;
        market[2] = 1;
        let lengths = [1_i64, 0];
        vector
            .step_into(&units, &market, &lengths, storage.output())
            .unwrap();

        assert_eq!(storage.rewards, [2_980.0, 3_000.0]);
        assert_eq!(storage.dones, [1, 1]);
        assert_eq!(vector.slots[0].sim.state.step, 0);
        assert_eq!(vector.slots[0].episode_id, 1);
        let terminal = vector.slots[0].terminal.as_ref().unwrap();
        assert_eq!(terminal.state.step, 1);
        assert_eq!(terminal.state.farms[0].private.seeds[1], 1);
        assert!((terminal.state.farms[0].money - 2_980.0).abs() < f64::EPSILON);
    }

    #[test]
    fn automatic_seed_schedule_is_disjoint_across_slots_and_episodes() {
        let config = Config {
            episode_steps: 2,
            weed_spawn_chance: 0.0,
            ..Config::default()
        };
        let mut vector = VecEnvCore::new(3, &config, 7, 0, true).unwrap();
        let mut storage = TestBuffers::new(&vector);
        vector.reset_into(None, storage.output()).unwrap();
        assert_eq!(
            vector
                .slots
                .iter()
                .map(|slot| slot.sim.config.seed)
                .collect::<Vec<_>>(),
            [7, 8, 9]
        );

        let units = vec![0_i64; 3 * PLAYER_COUNT * vector.max_units * ACTION_FIELD_COUNT];
        let market = vec![0_i64; 3 * PLAYER_COUNT * vector.max_orders * ACTION_FIELD_COUNT];
        let lengths = vec![0_i64; 3 * PLAYER_COUNT];
        vector
            .step_into(&units, &market, &lengths, storage.output())
            .unwrap();

        assert_eq!(
            vector
                .slots
                .iter()
                .map(|slot| slot.sim.config.seed)
                .collect::<Vec<_>>(),
            [10, 11, 12]
        );
        assert_eq!(
            vector
                .slots
                .iter()
                .map(|slot| slot.terminal.as_ref().unwrap().seed)
                .collect::<Vec<_>>(),
            [7, 8, 9]
        );
    }
}
