//! Stable identifiers and static game data.

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum Product {
    Wheat = 0,
    Carrot,
    Tomato,
    Strawberry,
    Melon,
    Egg,
    Milk,
    Wool,
    Fertilizer,
}

impl Product {
    pub const ALL: [Self; 9] = [
        Self::Wheat,
        Self::Carrot,
        Self::Tomato,
        Self::Strawberry,
        Self::Melon,
        Self::Egg,
        Self::Milk,
        Self::Wool,
        Self::Fertilizer,
    ];
    pub const COUNT: usize = Self::ALL.len();

    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Wheat => "WHEAT",
            Self::Carrot => "CARROT",
            Self::Tomato => "TOMATO",
            Self::Strawberry => "STRAWBERRY",
            Self::Melon => "MELON",
            Self::Egg => "EGG",
            Self::Milk => "MILK",
            Self::Wool => "WOOL",
            Self::Fertilizer => "FERTILIZER",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum Crop {
    Wheat = 0,
    Carrot,
    Tomato,
    Strawberry,
    Melon,
}

impl Crop {
    pub const ALL: [Self; 5] = [
        Self::Wheat,
        Self::Carrot,
        Self::Tomato,
        Self::Strawberry,
        Self::Melon,
    ];
    pub const COUNT: usize = Self::ALL.len();

    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[must_use]
    pub const fn product(self) -> Product {
        Product::ALL[self.index()]
    }

    #[must_use]
    pub const fn item(self) -> Item {
        Item::ALL[self.index()]
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.product().as_str()
    }

    #[must_use]
    pub const fn seed_cost(self) -> i64 {
        CROP_DEFS[self.index()].seed_cost
    }

    #[must_use]
    pub const fn first_yield_day(self) -> i32 {
        CROP_DEFS[self.index()].first_yield_day
    }

    #[must_use]
    pub const fn max_yield_day(self) -> i32 {
        CROP_DEFS[self.index()].max_yield_day
    }

    #[must_use]
    pub const fn interval(self) -> i32 {
        CROP_DEFS[self.index()].interval
    }

    #[must_use]
    pub const fn max_yield(self) -> i64 {
        CROP_DEFS[self.index()].max_yield
    }

    #[must_use]
    pub const fn ongoing(self) -> bool {
        CROP_DEFS[self.index()].ongoing
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum Animal {
    Goose = 0,
    Cow,
    Sheep,
}

impl Animal {
    pub const ALL: [Self; 3] = [Self::Goose, Self::Cow, Self::Sheep];
    pub const COUNT: usize = Self::ALL.len();

    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[must_use]
    pub const fn item(self) -> Item {
        Item::ALL[Product::COUNT + self.index()]
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Goose => "GOOSE",
            Self::Cow => "COW",
            Self::Sheep => "SHEEP",
        }
    }

    #[must_use]
    pub const fn cost(self) -> i64 {
        ANIMAL_DEFS[self.index()].cost
    }

    #[must_use]
    pub const fn structure(self) -> Structure {
        ANIMAL_DEFS[self.index()].structure
    }

    #[must_use]
    pub const fn first_yield_day(self) -> i32 {
        ANIMAL_DEFS[self.index()].first_yield_day
    }

    #[must_use]
    pub const fn interval(self) -> i32 {
        ANIMAL_DEFS[self.index()].interval
    }

    #[must_use]
    pub const fn max_held(self) -> i64 {
        ANIMAL_DEFS[self.index()].max_held
    }

    #[must_use]
    pub const fn product(self) -> Product {
        ANIMAL_DEFS[self.index()].product
    }
}

/// Every item that can be held in the shed or by a unit.  The first nine
/// values deliberately have the same numeric IDs as [`Product`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum Item {
    Wheat = 0,
    Carrot,
    Tomato,
    Strawberry,
    Melon,
    Egg,
    Milk,
    Wool,
    Fertilizer,
    Goose,
    Cow,
    Sheep,
}

impl Item {
    pub const ALL: [Self; 12] = [
        Self::Wheat,
        Self::Carrot,
        Self::Tomato,
        Self::Strawberry,
        Self::Melon,
        Self::Egg,
        Self::Milk,
        Self::Wool,
        Self::Fertilizer,
        Self::Goose,
        Self::Cow,
        Self::Sheep,
    ];
    pub const COUNT: usize = Self::ALL.len();

    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[must_use]
    pub const fn as_product(self) -> Option<Product> {
        if self.index() < Product::COUNT {
            Some(Product::ALL[self.index()])
        } else {
            None
        }
    }

    #[must_use]
    pub const fn as_crop(self) -> Option<Crop> {
        if self.index() < Crop::COUNT {
            Some(Crop::ALL[self.index()])
        } else {
            None
        }
    }

    #[must_use]
    pub const fn as_animal(self) -> Option<Animal> {
        let index = self.index();
        if index >= Product::COUNT && index < Item::COUNT {
            Some(Animal::ALL[index - Product::COUNT])
        } else {
            None
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self.as_product() {
            Some(product) => product.as_str(),
            None => Animal::ALL[self.index() - Product::COUNT].as_str(),
        }
    }
}

impl From<Product> for Item {
    fn from(value: Product) -> Self {
        Self::ALL[value.index()]
    }
}

impl From<Crop> for Item {
    fn from(value: Crop) -> Self {
        value.item()
    }
}

impl From<Animal> for Item {
    fn from(value: Animal) -> Self {
        value.item()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Structure {
    Coop,
    Pasture,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum Quadrant {
    Nw = 0,
    Ne,
    Sw,
    Se,
}

impl Quadrant {
    pub const ALL: [Self; 4] = [Self::Nw, Self::Ne, Self::Sw, Self::Se];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nw => "NW",
            Self::Ne => "NE",
            Self::Sw => "SW",
            Self::Se => "SE",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum Shop {
    Bakery = 0,
    BrunchSpot,
    FarmersMarket,
    IceCreamShop,
    PetCafe,
    PizzaShop,
    SmoothieShop,
    YarnStore,
}

impl Shop {
    /// The exact order used by `random.choice(sorted(SHOPS))` in Python.
    pub const SORTED: [Self; 8] = [
        Self::Bakery,
        Self::BrunchSpot,
        Self::FarmersMarket,
        Self::IceCreamShop,
        Self::PetCafe,
        Self::PizzaShop,
        Self::SmoothieShop,
        Self::YarnStore,
    ];

    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bakery => "BAKERY",
            Self::BrunchSpot => "BRUNCH_SPOT",
            Self::FarmersMarket => "FARMERS_MARKET",
            Self::IceCreamShop => "ICE_CREAM_SHOP",
            Self::PetCafe => "PET_CAFE",
            Self::PizzaShop => "PIZZA_SHOP",
            Self::SmoothieShop => "SMOOTHIE_SHOP",
            Self::YarnStore => "YARN_STORE",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CropDef {
    pub seed_cost: i64,
    pub first_yield_day: i32,
    pub max_yield_day: i32,
    pub interval: i32,
    pub max_yield: i64,
    pub ongoing: bool,
}

pub(crate) const CROP_DEFS: [CropDef; Crop::COUNT] = [
    CropDef {
        seed_cost: 10,
        first_yield_day: 2,
        max_yield_day: 4,
        interval: 0,
        max_yield: 6,
        ongoing: false,
    },
    CropDef {
        seed_cost: 20,
        first_yield_day: 2,
        max_yield_day: 3,
        interval: 0,
        max_yield: 4,
        ongoing: false,
    },
    CropDef {
        seed_cost: 50,
        first_yield_day: 8,
        max_yield_day: 8,
        interval: 1,
        max_yield: 4,
        ongoing: true,
    },
    CropDef {
        seed_cost: 100,
        first_yield_day: 10,
        max_yield_day: 10,
        interval: 2,
        max_yield: 4,
        ongoing: true,
    },
    CropDef {
        seed_cost: 80,
        first_yield_day: 10,
        max_yield_day: 12,
        interval: 0,
        max_yield: 6,
        ongoing: false,
    },
];

#[derive(Clone, Copy, Debug)]
pub(crate) struct AnimalDef {
    pub cost: i64,
    pub structure: Structure,
    pub first_yield_day: i32,
    pub interval: i32,
    pub max_held: i64,
    pub product: Product,
}

pub(crate) const ANIMAL_DEFS: [AnimalDef; Animal::COUNT] = [
    AnimalDef {
        cost: 300,
        structure: Structure::Coop,
        first_yield_day: 4,
        interval: 1,
        max_held: 4,
        product: Product::Egg,
    },
    AnimalDef {
        cost: 400,
        structure: Structure::Pasture,
        first_yield_day: 8,
        interval: 2,
        max_held: 6,
        product: Product::Milk,
    },
    AnimalDef {
        cost: 500,
        structure: Structure::Pasture,
        first_yield_day: 6,
        interval: 3,
        max_held: 6,
        product: Product::Wool,
    },
];

pub(crate) const LAND_ORDER: [Quadrant; 3] = [Quadrant::Ne, Quadrant::Sw, Quadrant::Se];
pub(crate) const LAND_PRICES: [i64; 3] = [1_000, 2_000, 4_000];
pub(crate) const MAX_SHOP_INSTANCES: usize = 8;
