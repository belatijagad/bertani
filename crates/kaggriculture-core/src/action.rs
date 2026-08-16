use crate::{Animal, Crop, Item};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum UnitAction {
    #[default]
    Pass,
    North,
    South,
    East,
    West,
    Pickup {
        item: Item,
        count: i64,
    },
    Drop,
    Place {
        item: Item,
        count: i64,
    },
    Plant(Crop),
    Water,
    Harvest,
    Fertilize,
    Dig,
    BuildCoop,
    BuildPasture,
    Feed,
    CollectFertilizer,
    Care,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MarketOrder {
    Hire,
    BuyLand,
    BuySeed { crop: Crop, count: i64 },
    BuyProduct { item: Item, count: i64 },
    BuyAnimal { animal: Animal, count: i64 },
    Sell { item: Item, count: i64 },
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Action {
    pub farmer: UnitAction,
    pub hands: Vec<UnitAction>,
    pub market: Vec<MarketOrder>,
}

impl Action {
    #[must_use]
    pub fn pass() -> Self {
        Self::default()
    }
}
