use crate::constants::{ANIMAL_DEFS, CROP_DEFS};
use crate::{Animal, Crop, Item, Product, Quadrant, Shop, Structure};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Linear,
    Square,
    SquareRoot,
    NaturalLog,
    Log10,
    Hinge,
}

impl Shape {
    #[allow(clippy::match_same_arms)]
    pub(crate) fn apply(self, x: f64, threshold: f64) -> f64 {
        let x = x.max(0.0);
        match self {
            Self::Linear => x,
            Self::Square => x * x,
            Self::SquareRoot => x.sqrt(),
            // Match Python's `math.log(1.0 + x)`, not the subtly different
            // `log1p(x)` implementation.
            Self::NaturalLog => (1.0 + x).ln(),
            Self::Log10 => (1.0 + x).log10(),
            Self::Hinge if threshold > 0.0 => {
                let u = x / threshold;
                u + 8.0 * (u - 1.0).max(0.0).powi(2)
            }
            Self::Hinge => x,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarketParams {
    pub base: f64,
    pub initial_inventory: i64,
    pub threshold: f64,
    pub scarcity_shape: Shape,
    pub scarcity_target: f64,
    pub glut_shape: Shape,
    pub glut_target: f64,
}

pub const DEFAULT_MARKET_PARAMS: [MarketParams; Product::COUNT] = [
    MarketParams {
        base: 25.0,
        initial_inventory: 10_000,
        threshold: 400.0,
        scarcity_shape: Shape::SquareRoot,
        scarcity_target: 0.80,
        glut_shape: Shape::NaturalLog,
        glut_target: 0.20,
    },
    MarketParams {
        base: 35.0,
        initial_inventory: 10_000,
        threshold: 450.0,
        scarcity_shape: Shape::Hinge,
        scarcity_target: 1.00,
        glut_shape: Shape::SquareRoot,
        glut_target: 0.70,
    },
    MarketParams {
        base: 60.0,
        initial_inventory: 10_000,
        threshold: 200.0,
        scarcity_shape: Shape::Hinge,
        scarcity_target: 0.40,
        glut_shape: Shape::SquareRoot,
        glut_target: 0.60,
    },
    MarketParams {
        base: 120.0,
        initial_inventory: 10_000,
        threshold: 100.0,
        scarcity_shape: Shape::SquareRoot,
        scarcity_target: 0.70,
        glut_shape: Shape::Linear,
        glut_target: 1.60,
    },
    MarketParams {
        base: 250.0,
        initial_inventory: 10_000,
        threshold: 300.0,
        scarcity_shape: Shape::NaturalLog,
        scarcity_target: 0.20,
        glut_shape: Shape::Square,
        glut_target: 3.60,
    },
    MarketParams {
        base: 50.0,
        initial_inventory: 10_000,
        threshold: 332.0,
        scarcity_shape: Shape::Hinge,
        scarcity_target: 0.40,
        glut_shape: Shape::NaturalLog,
        glut_target: 0.20,
    },
    MarketParams {
        base: 160.0,
        initial_inventory: 10_000,
        threshold: 122.0,
        scarcity_shape: Shape::SquareRoot,
        scarcity_target: 0.60,
        glut_shape: Shape::Linear,
        glut_target: 1.60,
    },
    MarketParams {
        base: 200.0,
        initial_inventory: 10_000,
        threshold: 105.0,
        scarcity_shape: Shape::NaturalLog,
        scarcity_target: 0.20,
        glut_shape: Shape::Square,
        glut_target: 3.20,
    },
    MarketParams {
        base: 100.0,
        initial_inventory: 10_000,
        threshold: 200.0,
        scarcity_shape: Shape::Linear,
        scarcity_target: 0.40,
        glut_shape: Shape::Linear,
        glut_target: 0.40,
    },
];

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub episode_steps: u32,
    pub board_size: usize,
    pub starting_money: i64,
    pub max_market_orders_per_turn: usize,
    pub turns_per_day: u32,
    pub shed_capacity: i64,
    pub weed_spawn_chance: f64,
    pub town_shop_unlock_interval: u32,
    pub town_shop_sell_interval: u32,
    pub town_center_sell_interval: u32,
    pub farm_hand_cost_multiplier: i64,
    pub seed: u64,
    pub market_params: [MarketParams; Product::COUNT],
}

impl Default for Config {
    fn default() -> Self {
        Self {
            episode_steps: 720,
            board_size: 10,
            starting_money: 3_000,
            max_market_orders_per_turn: 10,
            turns_per_day: 24,
            shed_capacity: 100,
            weed_spawn_chance: 0.005,
            town_shop_unlock_interval: 3,
            town_shop_sell_interval: 4,
            town_center_sell_interval: 24,
            farm_hand_cost_multiplier: 1,
            seed: 0,
            market_params: DEFAULT_MARKET_PARAMS,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Position {
    pub x: usize,
    pub y: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Inventory {
    counts: [i64; Item::COUNT],
    insertion_order: [u8; Item::COUNT],
    order_len: u8,
}

impl Default for Inventory {
    fn default() -> Self {
        Self {
            counts: [0; Item::COUNT],
            insertion_order: [0; Item::COUNT],
            order_len: 0,
        }
    }
}

impl Inventory {
    #[must_use]
    pub const fn get(&self, item: Item) -> i64 {
        self.counts[item.index()]
    }

    #[must_use]
    pub fn total(&self) -> i64 {
        self.counts.iter().sum()
    }

    pub fn iter(&self) -> impl Iterator<Item = (Item, i64)> + '_ {
        self.insertion_order[..usize::from(self.order_len)]
            .iter()
            .map(|&index| {
                let item = Item::ALL[usize::from(index)];
                (item, self.get(item))
            })
    }

    pub(crate) fn add(&mut self, item: Item, count: i64) {
        if count <= 0 {
            return;
        }
        if self.get(item) == 0 {
            self.insertion_order[usize::from(self.order_len)] = item as u8;
            self.order_len += 1;
        }
        self.counts[item.index()] += count;
    }

    pub(crate) fn take(&mut self, item: Item, count: i64) -> bool {
        if count < 0 || self.get(item) < count {
            return false;
        }
        self.counts[item.index()] -= count;
        if self.get(item) == 0 {
            self.erase(item);
        }
        true
    }

    pub(crate) const fn order_snapshot(&self) -> ([u8; Item::COUNT], usize) {
        (self.insertion_order, self.order_len as usize)
    }

    fn erase(&mut self, item: Item) {
        let len = usize::from(self.order_len);
        if let Some(index) = self.insertion_order[..len]
            .iter()
            .position(|&candidate| candidate == item as u8)
        {
            self.insertion_order.copy_within(index + 1..len, index);
            self.order_len -= 1;
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Plant {
    pub crop: Crop,
    pub planted_day: i32,
    pub watered_today: bool,
    pub consecutive_unwatered: i32,
    pub yield_units: i64,
    pub max_lifespan_step: i64,
    pub fertilized_until_day: i32,
}

impl Plant {
    pub(crate) fn new(crop: Crop, day: i32, turns_per_day: u32) -> Self {
        let definition = CROP_DEFS[crop.index()];
        Self {
            crop,
            planted_day: day,
            watered_today: false,
            consecutive_unwatered: 1,
            yield_units: i64::from(!definition.ongoing),
            max_lifespan_step: if definition.ongoing {
                -1
            } else {
                i64::from(day + definition.max_yield_day + 1) * i64::from(turns_per_day)
            },
            fertilized_until_day: -1,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacedAnimal {
    pub animal: Animal,
    pub placed_day: i32,
    pub yield_units: i64,
    pub consecutive_unfed: i32,
    pub fed_today: bool,
    pub cared_today: bool,
    pub fertilizer_available: bool,
    pub pending_care_bonus: i64,
}

impl PlacedAnimal {
    pub(crate) fn new(animal: Animal, day: i32) -> Self {
        Self {
            animal,
            placed_day: day,
            yield_units: 0,
            consecutive_unfed: 0,
            fed_today: false,
            cared_today: false,
            fertilizer_available: false,
            pending_care_bonus: 0,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Tile {
    Empty,
    Locked,
    Weed,
    Plant(Plant),
    Structure {
        kind: Structure,
        animal: Option<PlacedAnimal>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrivateState {
    pub shed: [i64; Item::COUNT],
    pub seeds: [i64; Crop::COUNT],
    pub inventories: Vec<Inventory>,
}

impl Default for PrivateState {
    fn default() -> Self {
        Self {
            shed: [0; Item::COUNT],
            seeds: [0; Crop::COUNT],
            inventories: vec![Inventory::default()],
        }
    }
}

impl PrivateState {
    #[must_use]
    pub fn shed_total(&self) -> i64 {
        self.shed.iter().sum()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Farm {
    pub money: f64,
    pub tiles: Vec<Tile>,
    pub farmer: Position,
    pub hands: Vec<Position>,
    pub unlocked_quadrants: Vec<Quadrant>,
    pub hires_today: usize,
    pub private: PrivateState,
}

impl Farm {
    #[must_use]
    pub fn tile(&self, board_size: usize, position: Position) -> &Tile {
        &self.tiles[position.y * board_size + position.x]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Market {
    pub inventory: [i64; Product::COUNT],
    pub prices: [f64; Product::COUNT],
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Town {
    pub unlocked_shops: Vec<Shop>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub farms: [Farm; 2],
    pub market: Market,
    pub town: Town,
    pub step: u32,
    pub day: u32,
    pub hour: u32,
    pub done: bool,
}

pub(crate) fn animal_structure(animal: Animal) -> Structure {
    ANIMAL_DEFS[animal.index()].structure
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_preserves_python_dict_insertion_order() {
        let mut inventory = Inventory::default();
        inventory.add(Item::Milk, 2);
        inventory.add(Item::Wheat, 1);
        inventory.add(Item::Milk, 3);
        assert_eq!(
            inventory.iter().collect::<Vec<_>>(),
            vec![(Item::Milk, 5), (Item::Wheat, 1)]
        );

        assert!(inventory.take(Item::Milk, 5));
        inventory.add(Item::Milk, 1);
        assert_eq!(
            inventory.iter().collect::<Vec<_>>(),
            vec![(Item::Wheat, 1), (Item::Milk, 1)]
        );
    }
}
