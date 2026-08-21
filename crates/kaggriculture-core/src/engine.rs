#![allow(clippy::cast_precision_loss)]

use crate::constants::{
    ANIMAL_DEFS, CROP_DEFS, LAND_ORDER, LAND_PRICES, MAX_SHOP_INSTANCES, shop_products,
};
use crate::rng::PyRandom;
use crate::state::{PlacedAnimal, Plant, animal_structure};
use crate::{
    Action, Config, Crop, Farm, Inventory, Item, Market, MarketOrder, Position, PrivateState,
    Product, Quadrant, Shop, State, Structure, Tile, Town, UnitAction,
};

/// A faithful, deterministic implementation of the Kaggriculture transition
/// function.  Invalid actions represented by the typed API become silent
/// no-ops where the Python environment would reject them.
#[derive(Clone, Debug)]
pub struct Sim {
    pub config: Config,
    pub state: State,
}

impl Default for Sim {
    fn default() -> Self {
        Self::new(Config::default())
    }
}

impl Sim {
    /// Constructs a fresh simulator from an already schema-validated config.
    ///
    /// # Panics
    ///
    /// Panics when values violate the Kaggle configuration schema's minimums.
    #[must_use]
    pub fn new(config: Config) -> Self {
        assert!(config.board_size >= 4, "board_size must be at least 4");
        assert!(
            config.episode_steps >= 1,
            "episode_steps must be at least 1"
        );
        assert!(config.turns_per_day > 0, "turns_per_day must be positive");
        assert!(
            config.max_market_orders_per_turn > 0,
            "max_market_orders_per_turn must be positive"
        );
        assert!(config.shed_capacity > 0, "shed_capacity must be positive");
        assert!(
            config.town_shop_unlock_interval > 0
                && config.town_shop_sell_interval > 0
                && config.town_center_sell_interval > 0,
            "town intervals must be positive"
        );
        assert!(
            config.farm_hand_cost_multiplier >= 0,
            "farm hand multiplier cannot be negative"
        );

        let state = Self::initial_state(&config);
        Self { config, state }
    }

    pub fn reset(&mut self) {
        self.state = Self::initial_state(&self.config);
    }

    #[must_use]
    pub fn reward(&self, player: usize) -> f64 {
        if self.state.done {
            self.state.farms[player].money
        } else {
            0.0
        }
    }

    /// Returns the player's current bank balance, including during an active
    /// episode. Unlike [`Self::reward`], this is not gated on termination.
    #[must_use]
    pub fn bank(&self, player: usize) -> f64 {
        self.state.farms[player].money
    }

    /// Estimates the current economic value of one farm.
    ///
    /// This is a training potential, not a game score. Cash is combined with
    /// owned inventory, planted assets, accumulated yields, and purchased
    /// land. Products use the current market quote; seeds and animals retain
    /// their acquisition cost until consumed or lost.
    #[must_use]
    pub fn economic_value(&self, player: usize) -> f64 {
        let farm = &self.state.farms[player];
        let item_value = |item: Item| -> f64 {
            if let Some(product) = item.as_product() {
                self.state.market.prices[product.index()]
            } else if let Some(animal) = item.as_animal() {
                animal.cost() as f64
            } else {
                0.0
            }
        };

        let mut value = farm.money;
        for item in Item::ALL {
            value += farm.private.shed[item.index()] as f64 * item_value(item);
        }
        for inventory in &farm.private.inventories {
            for (item, quantity) in inventory.iter() {
                value += quantity as f64 * item_value(item);
            }
        }
        for crop in Crop::ALL {
            value += farm.private.seeds[crop.index()] as f64 * crop.seed_cost() as f64;
        }

        for tile in &farm.tiles {
            match tile {
                Tile::Plant(plant) => {
                    value += plant.crop.seed_cost() as f64;
                    value += plant.yield_units as f64
                        * self.state.market.prices[plant.crop.product().index()];
                }
                Tile::Structure {
                    animal: Some(animal),
                    ..
                } => {
                    value += animal.animal.cost() as f64;
                    value += animal.yield_units as f64
                        * self.state.market.prices[animal.animal.product().index()];
                    if animal.fertilizer_available {
                        value += self.state.market.prices[Product::Fertilizer.index()];
                    }
                }
                _ => {}
            }
        }

        let extra_land = farm.unlocked_quadrants.len().saturating_sub(1);
        value + LAND_PRICES.iter().take(extra_land).sum::<i64>() as f64
    }

    /// Advances one acting transition.  Kaggle records 720 states for the
    /// default configuration, so this runs exactly 719 transitions.
    ///
    /// # Panics
    ///
    /// Panics only when an impractically large episode makes the in-game day
    /// exceed `i32::MAX`.
    pub fn step(&mut self, actions: [&Action; 2]) {
        if self.state.done {
            return;
        }

        let old_step = self.state.step;
        let day =
            i32::try_from(old_step / self.config.turns_per_day).expect("day counter fits in i32");

        self.apply_unit_actions(0, actions[0], day);
        self.apply_unit_actions(1, actions[1], day);
        self.process_market(actions);
        self.town_consume(old_step);
        self.decay_plants(old_step);
        if (old_step + 1) % self.config.turns_per_day == 0 {
            self.end_of_day(day);
        }

        self.state.step = old_step + 1;
        self.state.day = self.state.step / self.config.turns_per_day;
        self.state.hour = self.state.step % self.config.turns_per_day;
        if old_step >= self.config.episode_steps.saturating_sub(2) {
            self.state.done = true;
        }
    }

    fn initial_state(config: &Config) -> State {
        let farms = std::array::from_fn(|_| Self::new_farm(config));
        let inventory = std::array::from_fn(|index| config.market_params[index].initial_inventory);
        let prices = std::array::from_fn(|index| config.market_params[index].base);
        State {
            farms,
            market: Market { inventory, prices },
            town: Town::default(),
            step: 0,
            day: 0,
            hour: 0,
            done: false,
        }
    }

    fn new_farm(config: &Config) -> Farm {
        let board_size = config.board_size;
        let tiles = (0..board_size)
            .flat_map(|y| {
                (0..board_size).map(move |x| {
                    if quadrant_of(x, y, board_size) == Quadrant::Nw {
                        Tile::Empty
                    } else {
                        Tile::Locked
                    }
                })
            })
            .collect();
        Farm {
            money: config.starting_money as f64,
            tiles,
            farmer: default_spawn(board_size),
            hands: Vec::new(),
            unlocked_quadrants: vec![Quadrant::Nw],
            hires_today: 0,
            private: PrivateState::default(),
        }
    }

    fn apply_unit_actions(&mut self, player: usize, action: &Action, day: i32) {
        let mut plant_demand = [0_i64; Crop::COUNT];
        if let UnitAction::Plant(crop) = action.farmer {
            plant_demand[crop.index()] += 1;
        }
        for hand_action in &action.hands {
            if let UnitAction::Plant(crop) = hand_action {
                plant_demand[crop.index()] += 1;
            }
        }

        let seeds = self.state.farms[player].private.seeds;
        let blocked = std::array::from_fn::<_, { Crop::COUNT }, _>(|index| {
            plant_demand[index] > seeds[index]
        });

        let farmer_action = block_plant(action.farmer, blocked);
        self.apply_unit_action(player, 0, farmer_action, day);
        for (hand_index, hand_action) in action.hands.iter().copied().enumerate() {
            let hand_action = block_plant(hand_action, blocked);
            self.apply_unit_action(player, hand_index + 1, hand_action, day);
        }
    }

    #[allow(clippy::too_many_lines)]
    fn apply_unit_action(
        &mut self,
        player: usize,
        unit_index: usize,
        action: UnitAction,
        day: i32,
    ) {
        let board_size = self.config.board_size;
        let shed_capacity = self.config.shed_capacity;
        let farm = &mut self.state.farms[player];
        let Some(position) = unit_position(farm, unit_index) else {
            return;
        };

        let movement = match action {
            UnitAction::North => Some((0_isize, -1_isize)),
            UnitAction::South => Some((0, 1)),
            UnitAction::East => Some((1, 0)),
            UnitAction::West => Some((-1, 0)),
            _ => None,
        };
        if let Some((dx, dy)) = movement {
            let Some(x) = position.x.checked_add_signed(dx) else {
                return;
            };
            let Some(y) = position.y.checked_add_signed(dy) else {
                return;
            };
            if x < board_size && y < board_size {
                set_unit_position(farm, unit_index, Position { x, y });
            }
            return;
        }
        if action == UnitAction::Pass {
            return;
        }

        // Shed operations deliberately resolve before the locked-tile guard.
        match action {
            UnitAction::Drop => {
                if !is_shed_adjacent(position, board_size) {
                    return;
                }
                let private = &mut farm.private;
                let (order, order_len) = private.inventories[unit_index].order_snapshot();
                for raw_item in order[..order_len].iter().copied() {
                    let item = Item::ALL[usize::from(raw_item)];
                    let count = private.inventories[unit_index].get(item);
                    if count <= 0 {
                        continue;
                    }
                    let room = (shed_capacity - private.shed_total()).max(0);
                    let deposited = count.min(room);
                    private.shed[item.index()] += deposited;
                    let removed = private.inventories[unit_index].take(item, count);
                    debug_assert!(removed);
                }
                return;
            }
            UnitAction::Pickup { item, count } => {
                if !is_shed_adjacent(position, board_size) || count <= 0 {
                    return;
                }
                let private = &mut farm.private;
                let moved = count.min(private.shed[item.index()]);
                if moved > 0 {
                    private.shed[item.index()] -= moved;
                    private.inventories[unit_index].add(item, moved);
                }
                return;
            }
            UnitAction::Place { item, count } => {
                let tile_index = position.y * board_size + position.x;
                if let Some(animal) = item.as_animal() {
                    let matches_empty_structure = matches!(
                        &farm.tiles[tile_index],
                        Tile::Structure { kind, animal: None }
                            if *kind == animal_structure(animal)
                    );
                    if matches_empty_structure {
                        if farm.private.inventories[unit_index].take(item, 1) {
                            farm.tiles[tile_index] = Tile::Structure {
                                kind: animal_structure(animal),
                                animal: Some(PlacedAnimal::new(animal, day)),
                            };
                        }
                        return;
                    }
                }
                if !is_shed_adjacent(position, board_size) || count <= 0 {
                    return;
                }
                let private = &mut farm.private;
                let carried = private.inventories[unit_index].get(item);
                let room = (shed_capacity - private.shed_total()).max(0);
                let moved = count.min(carried).min(room);
                if moved > 0 {
                    let removed = private.inventories[unit_index].take(item, moved);
                    debug_assert!(removed);
                    private.shed[item.index()] += moved;
                }
                return;
            }
            _ => {}
        }

        let tile_index = position.y * board_size + position.x;
        if farm.tiles[tile_index] == Tile::Locked {
            return;
        }

        match action {
            UnitAction::Plant(crop) => {
                if farm.tiles[tile_index] != Tile::Empty || farm.private.seeds[crop.index()] <= 0 {
                    return;
                }
                farm.private.seeds[crop.index()] -= 1;
                farm.tiles[tile_index] =
                    Tile::Plant(Plant::new(crop, day, self.config.turns_per_day));
            }
            UnitAction::Water => {
                let Tile::Plant(plant) = &mut farm.tiles[tile_index] else {
                    return;
                };
                if plant.watered_today {
                    return;
                }
                plant.watered_today = true;
                let definition = CROP_DEFS[plant.crop.index()];
                if !definition.ongoing {
                    let age = day - plant.planted_day;
                    let window_start = (definition.max_yield_day + 1) / 2;
                    if (window_start..=definition.max_yield_day).contains(&age) {
                        let bonus = i64::from(plant.fertilized_until_day >= day) + 1;
                        plant.yield_units = definition.max_yield.min(plant.yield_units + bonus);
                    }
                }
            }
            UnitAction::Harvest => match &mut farm.tiles[tile_index] {
                Tile::Plant(plant) => {
                    if plant.yield_units <= 0 {
                        return;
                    }
                    let definition = CROP_DEFS[plant.crop.index()];
                    if day - plant.planted_day < definition.first_yield_day {
                        return;
                    }
                    let crop = plant.crop;
                    let units = plant.yield_units;
                    plant.yield_units = 0;
                    farm.private.inventories[unit_index].add(crop.item(), units);
                    if !definition.ongoing {
                        farm.tiles[tile_index] = Tile::Empty;
                    }
                }
                Tile::Structure {
                    animal: Some(animal),
                    ..
                } if animal.yield_units > 0 => {
                    let definition = ANIMAL_DEFS[animal.animal.index()];
                    let units = animal.yield_units;
                    animal.yield_units = 0;
                    farm.private.inventories[unit_index].add(definition.product.into(), units);
                }
                _ => {}
            },
            UnitAction::Fertilize => {
                if !matches!(farm.tiles[tile_index], Tile::Plant(_)) {
                    return;
                }
                if !farm.private.inventories[unit_index].take(Item::Fertilizer, 1) {
                    return;
                }
                if let Tile::Plant(plant) = &mut farm.tiles[tile_index] {
                    plant.fertilized_until_day = plant.fertilized_until_day.max(day + 2);
                }
            }
            UnitAction::Dig => match &farm.tiles[tile_index] {
                Tile::Empty
                | Tile::Structure {
                    animal: Some(_), ..
                } => {}
                _ => farm.tiles[tile_index] = Tile::Empty,
            },
            UnitAction::BuildCoop if farm.tiles[tile_index] == Tile::Empty => {
                farm.tiles[tile_index] = Tile::Structure {
                    kind: Structure::Coop,
                    animal: None,
                };
            }
            UnitAction::BuildPasture if farm.tiles[tile_index] == Tile::Empty => {
                farm.tiles[tile_index] = Tile::Structure {
                    kind: Structure::Pasture,
                    animal: None,
                };
            }
            UnitAction::Feed => {
                let Tile::Structure {
                    animal: Some(animal),
                    ..
                } = &mut farm.tiles[tile_index]
                else {
                    return;
                };
                if animal.fed_today {
                    return;
                }
                if farm.private.inventories[unit_index].take(Item::Wheat, 1) {
                    animal.fed_today = true;
                }
            }
            UnitAction::CollectFertilizer => {
                let Tile::Structure {
                    animal: Some(animal),
                    ..
                } = &mut farm.tiles[tile_index]
                else {
                    return;
                };
                if animal.fertilizer_available {
                    animal.fertilizer_available = false;
                    farm.private.inventories[unit_index].add(Item::Fertilizer, 1);
                }
            }
            UnitAction::Care => {
                let Tile::Structure {
                    animal: Some(animal),
                    ..
                } = &mut farm.tiles[tile_index]
                else {
                    return;
                };
                if !animal.cared_today {
                    animal.cared_today = true;
                }
            }
            _ => {}
        }
    }
}

fn block_plant(action: UnitAction, blocked: [bool; Crop::COUNT]) -> UnitAction {
    match action {
        UnitAction::Plant(crop) if blocked[crop.index()] => UnitAction::Pass,
        _ => action,
    }
}

fn quadrant_of(x: usize, y: usize, board_size: usize) -> Quadrant {
    let half = board_size / 2;
    match (y < half, x < half) {
        (true, true) => Quadrant::Nw,
        (true, false) => Quadrant::Ne,
        (false, true) => Quadrant::Sw,
        (false, false) => Quadrant::Se,
    }
}

fn shed_access_tiles(board_size: usize) -> [Position; 4] {
    let half = board_size / 2;
    [
        Position {
            x: half - 1,
            y: half - 1,
        },
        Position {
            x: half,
            y: half - 1,
        },
        Position {
            x: half - 1,
            y: half,
        },
        Position { x: half, y: half },
    ]
}

fn is_shed_adjacent(position: Position, board_size: usize) -> bool {
    shed_access_tiles(board_size).contains(&position)
}

fn default_spawn(board_size: usize) -> Position {
    shed_access_tiles(board_size)[0]
}

fn unit_position(farm: &Farm, unit_index: usize) -> Option<Position> {
    if unit_index == 0 {
        Some(farm.farmer)
    } else {
        farm.hands.get(unit_index - 1).copied()
    }
}

fn set_unit_position(farm: &mut Farm, unit_index: usize, position: Position) {
    if unit_index == 0 {
        farm.farmer = position;
    } else {
        farm.hands[unit_index - 1] = position;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ActiveOrderKind {
    BuySeed,
    BuyProduct,
    BuyAnimal,
    Sell,
}

#[derive(Clone, Copy, Debug)]
struct ActiveOrder {
    kind: ActiveOrderKind,
    item: Item,
    remaining: i64,
}

#[derive(Clone, Copy, Debug)]
struct Quote {
    kind: ActiveOrderKind,
    item: Item,
    price: i64,
}

impl Sim {
    fn process_market(&mut self, actions: [&Action; 2]) {
        let max_len = actions
            .iter()
            .map(|action| {
                action
                    .market
                    .len()
                    .min(self.config.max_market_orders_per_turn)
            })
            .max()
            .unwrap_or(0);

        for slot in 0..max_len {
            let mut orders: [Option<MarketOrder>; 2] = std::array::from_fn(|player| {
                actions[player]
                    .market
                    .get(slot)
                    .copied()
                    .filter(valid_market_quantity)
            });

            // HIRE and BUY_LAND execute once, in player order, before any
            // per-unit trades in the same slot.
            for (player, order) in orders.iter_mut().enumerate() {
                match *order {
                    Some(MarketOrder::Hire) => {
                        self.do_hire(player);
                        *order = None;
                    }
                    Some(MarketOrder::BuyLand) => {
                        self.do_buy_land(player);
                        *order = None;
                    }
                    _ => {}
                }
            }

            let mut active = orders.map(|order| order.and_then(active_order));
            // Python increments a guard first and aborts when it reaches
            // 100,000, allowing at most 99,999 units in one order slot.
            for _ in 0..99_999 {
                let mut quotes: [Option<Quote>; 2] = [None, None];
                for player in 0..2 {
                    let Some(order) = active[player] else {
                        continue;
                    };
                    if order.remaining <= 0 {
                        continue;
                    }
                    quotes[player] = self.quote(order);
                    if quotes[player].is_none() {
                        active[player] = None;
                    }
                }

                if quotes.iter().all(Option::is_none) {
                    break;
                }

                let mut committed_any = false;
                for player in 0..2 {
                    let Some(quote) = quotes[player] else {
                        continue;
                    };
                    if self.commit_unit(player, quote) {
                        if let Some(order) = &mut active[player] {
                            order.remaining -= 1;
                        }
                        committed_any = true;
                    } else {
                        active[player] = None;
                    }
                }
                if !committed_any {
                    break;
                }
            }
            self.refresh_prices();
        }
    }

    fn quote(&self, order: ActiveOrder) -> Option<Quote> {
        let price = match order.kind {
            ActiveOrderKind::Sell => {
                let product = order.item.as_product()?;
                self.market_price(product, self.state.market.inventory[product.index()])
            }
            ActiveOrderKind::BuyProduct => {
                let product = order.item.as_product()?;
                if !matches!(product, Product::Wheat | Product::Fertilizer) {
                    return None;
                }
                self.market_price(product, self.state.market.inventory[product.index()] - 1)
            }
            ActiveOrderKind::BuySeed => {
                let crop = order.item.as_crop()?;
                CROP_DEFS[crop.index()].seed_cost
            }
            ActiveOrderKind::BuyAnimal => {
                let animal = order.item.as_animal()?;
                ANIMAL_DEFS[animal.index()].cost
            }
        };
        Some(Quote {
            kind: order.kind,
            item: order.item,
            price,
        })
    }

    fn commit_unit(&mut self, player: usize, quote: Quote) -> bool {
        let farm = &mut self.state.farms[player];
        match quote.kind {
            ActiveOrderKind::Sell => {
                let Some(product) = quote.item.as_product() else {
                    return false;
                };
                if farm.private.shed[quote.item.index()] <= 0 {
                    return false;
                }
                farm.private.shed[quote.item.index()] -= 1;
                farm.money += quote.price as f64;
                if quote.price > 1 {
                    self.state.market.inventory[product.index()] += 1;
                }
                true
            }
            ActiveOrderKind::BuyProduct => {
                let Some(product) = quote.item.as_product() else {
                    return false;
                };
                if float_less_than_integer(farm.money, i128::from(quote.price))
                    || farm.private.shed_total() >= self.config.shed_capacity
                {
                    return false;
                }
                farm.money -= quote.price as f64;
                farm.private.shed[quote.item.index()] += 1;
                self.state.market.inventory[product.index()] -= 1;
                true
            }
            ActiveOrderKind::BuySeed => {
                let Some(crop) = quote.item.as_crop() else {
                    return false;
                };
                if float_less_than_integer(farm.money, i128::from(quote.price)) {
                    return false;
                }
                farm.money -= quote.price as f64;
                farm.private.seeds[crop.index()] += 1;
                true
            }
            ActiveOrderKind::BuyAnimal => {
                if quote.item.as_animal().is_none()
                    || float_less_than_integer(farm.money, i128::from(quote.price))
                    || farm.private.shed_total() >= self.config.shed_capacity
                {
                    return false;
                }
                farm.money -= quote.price as f64;
                farm.private.shed[quote.item.index()] += 1;
                true
            }
        }
    }

    fn do_hire(&mut self, player: usize) {
        let cost = i128::from(self.config.farm_hand_cost_multiplier)
            .saturating_mul(fibonacci(self.state.farms[player].hires_today));
        let farm = &mut self.state.farms[player];
        if float_less_than_integer(farm.money, cost) {
            return;
        }
        farm.money -= cost as f64;
        farm.hires_today += 1;

        let access = shed_access_tiles(self.config.board_size);
        let mut occupancy = [0_usize; 4];
        for position in std::iter::once(&farm.farmer).chain(farm.hands.iter()) {
            if let Some(index) = access.iter().position(|candidate| candidate == position) {
                occupancy[index] += 1;
            }
        }
        let best = (0..4).min_by_key(|&index| occupancy[index]).unwrap_or(0);
        farm.hands.push(access[best]);
        farm.private.inventories.push(Inventory::default());
    }

    fn do_buy_land(&mut self, player: usize) {
        let farm = &mut self.state.farms[player];
        let extra_unlocked = farm.unlocked_quadrants.len() - 1;
        let Some((&quadrant, &cost)) = LAND_ORDER
            .get(extra_unlocked)
            .zip(LAND_PRICES.get(extra_unlocked))
        else {
            return;
        };
        if float_less_than_integer(farm.money, i128::from(cost)) {
            return;
        }
        farm.money -= cost as f64;
        farm.unlocked_quadrants.push(quadrant);
        for y in 0..self.config.board_size {
            for x in 0..self.config.board_size {
                let index = y * self.config.board_size + x;
                if quadrant_of(x, y, self.config.board_size) == quadrant
                    && farm.tiles[index] == Tile::Locked
                {
                    farm.tiles[index] = Tile::Empty;
                }
            }
        }
    }

    fn town_consume(&mut self, step: u32) {
        if step % self.config.town_shop_sell_interval == 0 {
            for shop in &self.state.town.unlocked_shops {
                let multiplier = if shop_products(*shop).len() == 1 {
                    2
                } else {
                    1
                };
                for product in shop_products(*shop) {
                    self.state.market.inventory[product.index()] -= multiplier;
                }
            }
        }
        if step % self.config.town_center_sell_interval == 0 {
            for product in Product::ALL {
                if product != Product::Fertilizer {
                    self.state.market.inventory[product.index()] -= 1;
                }
            }
        }
        self.refresh_prices();
    }

    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn market_price(&self, product: Product, inventory: i64) -> i64 {
        let params = self.config.market_params[product.index()];
        let base = params.base;
        let price = if inventory < params.initial_inventory {
            let amplitude = params.scarcity_target * base
                / params
                    .scarcity_shape
                    .apply(params.threshold, params.threshold);
            base + amplitude
                * params.scarcity_shape.apply(
                    (params.initial_inventory - inventory) as f64,
                    params.threshold,
                )
        } else {
            let amplitude = params.glut_target * base
                / params.glut_shape.apply(params.threshold, params.threshold);
            base - amplitude
                * params.glut_shape.apply(
                    (inventory - params.initial_inventory) as f64,
                    params.threshold,
                )
        };
        (price.round_ties_even() as i64).max(1)
    }

    fn refresh_prices(&mut self) {
        for product in Product::ALL {
            self.state.market.prices[product.index()] =
                self.market_price(product, self.state.market.inventory[product.index()]) as f64;
        }
    }
}

fn valid_market_quantity(order: &MarketOrder) -> bool {
    match order {
        MarketOrder::Hire | MarketOrder::BuyLand => true,
        MarketOrder::BuySeed { count, .. }
        | MarketOrder::BuyProduct { count, .. }
        | MarketOrder::BuyAnimal { count, .. }
        | MarketOrder::Sell { count, .. } => *count > 0,
    }
}

fn active_order(order: MarketOrder) -> Option<ActiveOrder> {
    match order {
        MarketOrder::Hire | MarketOrder::BuyLand => None,
        MarketOrder::BuySeed { crop, count } => Some(ActiveOrder {
            kind: ActiveOrderKind::BuySeed,
            item: crop.into(),
            remaining: count,
        }),
        MarketOrder::BuyProduct { item, count } => Some(ActiveOrder {
            kind: ActiveOrderKind::BuyProduct,
            item,
            remaining: count,
        }),
        MarketOrder::BuyAnimal { animal, count } => Some(ActiveOrder {
            kind: ActiveOrderKind::BuyAnimal,
            item: animal.into(),
            remaining: count,
        }),
        MarketOrder::Sell { item, count } => Some(ActiveOrder {
            kind: ActiveOrderKind::Sell,
            item,
            remaining: count,
        }),
    }
}

fn fibonacci(index: usize) -> i128 {
    let (mut current, mut next) = (1_i128, 1_i128);
    for _ in 0..index {
        (current, next) = (next, current.saturating_add(next));
    }
    current
}

/// Mirrors Python's exact comparison between a binary float and an integer,
/// avoiding a lossy integer-to-float conversion at the 2^53 boundary.
#[allow(clippy::cast_possible_truncation)]
fn float_less_than_integer(value: f64, integer: i128) -> bool {
    if value.is_nan() || value == f64::INFINITY {
        return false;
    }
    if value == f64::NEG_INFINITY {
        return true;
    }
    (value.floor() as i128) < integer
}

impl Sim {
    fn decay_plants(&mut self, step: u32) {
        let step = i64::from(step);
        for farm in &mut self.state.farms {
            for tile in &mut farm.tiles {
                let Tile::Plant(plant) = tile else {
                    continue;
                };
                if plant.max_lifespan_step < 0 || step < plant.max_lifespan_step {
                    continue;
                }
                if (step - plant.max_lifespan_step) % 2 != 0 {
                    continue;
                }
                plant.yield_units -= 1;
                if plant.yield_units <= 0 {
                    *tile = Tile::Weed;
                }
            }
        }
    }

    fn end_of_day(&mut self, current_day: i32) {
        let random_key = (u128::from(self.config.seed) * 1_000_003_u128)
            ^ u128::try_from(current_day).expect("current day is nonnegative");
        let mut random = PyRandom::new(random_key);

        for farm in &mut self.state.farms {
            refresh_plants(farm, current_day, self.config.turns_per_day);
            refresh_animals(farm, current_day);
            spawn_weeds(farm, self.config.weed_spawn_chance, &mut random);
            drop_inventories(farm, self.config.shed_capacity);
            farm.farmer = default_spawn(self.config.board_size);
            farm.hands.clear();
            farm.hires_today = 0;
            farm.private.inventories.clear();
            farm.private.inventories.push(Inventory::default());
        }

        let next_day = current_day + 1;
        if next_day > 0
            && u32::try_from(next_day).expect("next day is nonnegative")
                % self.config.town_shop_unlock_interval
                == 0
            && self.state.town.unlocked_shops.len() < MAX_SHOP_INSTANCES
        {
            let selected = random.choice_index(Shop::SORTED.len());
            self.state.town.unlocked_shops.push(Shop::SORTED[selected]);
        }
    }
}

fn refresh_plants(farm: &mut Farm, current_day: i32, turns_per_day: u32) {
    let next_day = current_day + 1;
    for tile in &mut farm.tiles {
        let Tile::Plant(plant) = tile else {
            continue;
        };
        let was_watered = plant.watered_today;
        if was_watered {
            plant.consecutive_unwatered = 0;
        } else {
            plant.consecutive_unwatered += 1;
        }
        plant.watered_today = false;
        if plant.consecutive_unwatered >= 2 {
            *tile = Tile::Weed;
            continue;
        }

        let definition = CROP_DEFS[plant.crop.index()];
        if !definition.ongoing {
            continue;
        }
        let days_since_first = next_day - plant.planted_day - definition.first_yield_day;
        if days_since_first < 0 || days_since_first % definition.interval != 0 {
            continue;
        }
        let production_count = days_since_first / definition.interval + 1;
        if i64::from(production_count) > definition.max_yield {
            continue;
        }
        let fertilized = was_watered && plant.fertilized_until_day >= current_day;
        let produced = if fertilized { 2 } else { 1 };
        plant.yield_units = definition.max_yield.min(plant.yield_units + produced);
        if i64::from(production_count) == definition.max_yield {
            plant.max_lifespan_step = i64::from(next_day + 1) * i64::from(turns_per_day);
        }
    }
}

fn refresh_animals(farm: &mut Farm, current_day: i32) {
    let next_day = current_day + 1;
    for tile in &mut farm.tiles {
        let Tile::Structure {
            kind,
            animal: Some(placed),
        } = tile
        else {
            continue;
        };

        if placed.fed_today {
            placed.consecutive_unfed = 0;
        } else {
            placed.consecutive_unfed += 1;
        }
        if placed.consecutive_unfed >= 2 {
            let structure = *kind;
            *tile = Tile::Structure {
                kind: structure,
                animal: None,
            };
            continue;
        }

        let definition = ANIMAL_DEFS[placed.animal.index()];
        let days_since_first = next_day - placed.placed_day - definition.first_yield_day;
        if days_since_first >= 0 && days_since_first % definition.interval == 0 {
            let care_bonus = if placed.fed_today {
                placed.pending_care_bonus
            } else {
                0
            };
            placed.yield_units = definition.max_held.min(placed.yield_units + 1 + care_bonus);
            placed.pending_care_bonus = 0;
        }
        if placed.cared_today && placed.fed_today {
            placed.pending_care_bonus += 1;
        }
        placed.fertilizer_available = true;
        placed.fed_today = false;
        placed.cared_today = false;
    }
}

fn spawn_weeds(farm: &mut Farm, chance: f64, random: &mut PyRandom) {
    for tile in &mut farm.tiles {
        if *tile == Tile::Empty && random.random() < chance {
            *tile = Tile::Weed;
        }
    }
}

fn drop_inventories(farm: &mut Farm, capacity: i64) {
    for unit_index in 0..farm.private.inventories.len() {
        let (order, order_len) = farm.private.inventories[unit_index].order_snapshot();
        for raw_item in order[..order_len].iter().copied() {
            let item = Item::ALL[usize::from(raw_item)];
            let count = farm.private.inventories[unit_index].get(item);
            if count <= 0 {
                continue;
            }
            let room = (capacity - farm.private.shed_total()).max(0);
            let deposited = count.min(room);
            farm.private.shed[item.index()] += deposited;
            let removed = farm.private.inventories[unit_index].take(item, count);
            debug_assert!(removed);
        }
    }
}
