#![allow(
    clippy::float_cmp,
    clippy::needless_pass_by_value,
    clippy::too_many_lines
)]

use kaggriculture_core::{
    Action, Animal, Config, Crop, Item, MarketOrder, MarketParams, Position, Product, Quadrant,
    Shape, Sim, Structure, Tile, UnitAction,
};

fn step(sim: &mut Sim, player_zero: Action, player_one: Action) {
    sim.step([&player_zero, &player_one]);
}

fn pass(sim: &mut Sim) {
    step(sim, Action::pass(), Action::pass());
}

fn pass_turns(sim: &mut Sim, count: usize) {
    for _ in 0..count {
        pass(sim);
    }
}

fn deterministic_config(turns_per_day: u32) -> Config {
    Config {
        turns_per_day,
        weed_spawn_chance: 0.0,
        seed: 7,
        ..Config::default()
    }
}

fn farmer_action(action: UnitAction) -> Action {
    Action {
        farmer: action,
        ..Action::pass()
    }
}

fn tile_at_farmer(sim: &Sim, player: usize) -> &Tile {
    let farm = &sim.state.farms[player];
    farm.tile(sim.config.board_size, farm.farmer)
}

#[test]
fn initial_layout_and_clock_follow_the_reference_geometry() {
    let mut sim = Sim::new(deterministic_config(3));

    assert_eq!((sim.state.step, sim.state.day, sim.state.hour), (0, 0, 0));
    assert!(!sim.state.done);
    assert!(sim.state.town.unlocked_shops.is_empty());

    for farm in &sim.state.farms {
        assert_eq!(farm.money, 3_000.0);
        assert_eq!(farm.farmer, Position { x: 4, y: 4 });
        assert!(farm.hands.is_empty());
        assert_eq!(farm.unlocked_quadrants, vec![Quadrant::Nw]);
        assert_eq!(farm.private.inventories.len(), 1);
        assert_eq!(farm.private.shed_total(), 0);

        for y in 0..10 {
            for x in 0..10 {
                let expected = if x < 5 && y < 5 {
                    Tile::Empty
                } else {
                    Tile::Locked
                };
                assert_eq!(farm.tile(10, Position { x, y }), &expected);
            }
        }
    }

    for product in Product::ALL {
        let params = sim.config.market_params[product.index()];
        assert_eq!(sim.state.market.inventory[product.index()], 10_000);
        assert_eq!(sim.state.market.prices[product.index()], params.base);
    }

    pass(&mut sim);
    assert_eq!((sim.state.step, sim.state.day, sim.state.hour), (1, 0, 1));
    pass(&mut sim);
    assert_eq!((sim.state.step, sim.state.day, sim.state.hour), (2, 0, 2));
    pass(&mut sim);
    assert_eq!((sim.state.step, sim.state.day, sim.state.hour), (3, 1, 0));
}

#[test]
fn economic_value_preserves_owned_assets_at_acquisition_cost() {
    let mut sim = Sim::new(deterministic_config(24));
    let initial = sim.economic_value(0);
    let buy_seed = Action {
        market: vec![MarketOrder::BuySeed {
            crop: Crop::Carrot,
            count: 1,
        }],
        ..Action::pass()
    };

    step(&mut sim, buy_seed, Action::pass());

    assert_eq!(sim.state.farms[0].money, 2_980.0);
    assert_eq!(sim.economic_value(0), initial);
    assert_eq!(sim.economic_value(1), initial);
}

#[test]
fn market_buys_quote_post_trade_inventory_in_player_lockstep() {
    let mut config = deterministic_config(24);
    config.market_params[Product::Fertilizer.index()] = MarketParams {
        base: 100.0,
        initial_inventory: 10,
        threshold: 1.0,
        scarcity_shape: Shape::Linear,
        scarcity_target: 1.0,
        glut_shape: Shape::Linear,
        glut_target: 1.0,
    };
    let mut sim = Sim::new(config);
    let buy_two = Action {
        market: vec![MarketOrder::BuyProduct {
            item: Item::Fertilizer,
            count: 2,
        }],
        ..Action::pass()
    };

    step(&mut sim, buy_two.clone(), buy_two);

    // Both players see the same quote in each unit round: 200, then 400.
    // A sequential implementation would charge one seat 200 + 400 and the
    // other 300 + 500. Quoting the current inventory rather than inventory-1
    // would make the first quote 100 instead of 200.
    for farm in &sim.state.farms {
        assert_eq!(farm.money, 2_400.0);
        assert_eq!(farm.private.shed[Item::Fertilizer.index()], 2);
    }
    assert_eq!(sim.state.market.inventory[Product::Fertilizer.index()], 6);
    assert_eq!(sim.state.market.prices[Product::Fertilizer.index()], 500.0);
}

#[test]
fn plant_demand_is_atomic_and_counts_actions_for_nonexistent_hands() {
    let mut sim = Sim::new(deterministic_config(24));
    let buy_one = Action {
        market: vec![MarketOrder::BuySeed {
            crop: Crop::Wheat,
            count: 1,
        }],
        ..Action::pass()
    };
    step(&mut sim, buy_one, Action::pass());
    assert_eq!(sim.state.farms[0].private.seeds[Crop::Wheat.index()], 1);

    let over_demand = Action {
        farmer: UnitAction::Plant(Crop::Wheat),
        // This hand does not exist, but the reference still includes its plant
        // request in the all-or-nothing seed demand check.
        hands: vec![UnitAction::Plant(Crop::Wheat)],
        market: vec![MarketOrder::BuySeed {
            crop: Crop::Wheat,
            count: 1,
        }],
    };
    step(&mut sim, over_demand.clone(), Action::pass());

    assert_eq!(tile_at_farmer(&sim, 0), &Tile::Empty);
    assert_eq!(sim.state.farms[0].private.seeds[Crop::Wheat.index()], 2);

    let mut exact_demand = over_demand;
    exact_demand.market.clear();
    step(&mut sim, exact_demand, Action::pass());
    assert!(matches!(
        tile_at_farmer(&sim, 0),
        Tile::Plant(plant) if plant.crop == Crop::Wheat
    ));
    assert_eq!(sim.state.farms[0].private.seeds[Crop::Wheat.index()], 1);
}

#[test]
fn locked_tiles_allow_movement_and_center_shed_operations_only() {
    let mut sim = Sim::new(deterministic_config(24));
    let move_and_buy = Action {
        farmer: UnitAction::East,
        market: vec![MarketOrder::BuyProduct {
            item: Item::Wheat,
            count: 1,
        }],
        ..Action::pass()
    };
    step(&mut sim, move_and_buy, Action::pass());

    let locked_center = Position { x: 5, y: 4 };
    assert_eq!(sim.state.farms[0].farmer, locked_center);
    assert_eq!(
        sim.state.farms[0].tile(sim.config.board_size, locked_center),
        &Tile::Locked
    );
    assert_eq!(sim.state.farms[0].private.shed[Item::Wheat.index()], 1);

    step(
        &mut sim,
        farmer_action(UnitAction::Pickup {
            item: Item::Wheat,
            count: 1,
        }),
        Action::pass(),
    );
    assert_eq!(sim.state.farms[0].private.shed[Item::Wheat.index()], 0);
    assert_eq!(
        sim.state.farms[0].private.inventories[0].get(Item::Wheat),
        1
    );

    step(
        &mut sim,
        farmer_action(UnitAction::BuildCoop),
        Action::pass(),
    );
    assert_eq!(
        sim.state.farms[0].tile(sim.config.board_size, locked_center),
        &Tile::Locked
    );

    step(&mut sim, farmer_action(UnitAction::Drop), Action::pass());
    assert_eq!(sim.state.farms[0].private.shed[Item::Wheat.index()], 1);
    assert_eq!(sim.state.farms[0].private.inventories[0].total(), 0);
}

#[test]
fn hires_use_fibonacci_costs_and_land_unlocks_in_fixed_order() {
    let mut config = deterministic_config(24);
    config.starting_money = 10_000;
    config.farm_hand_cost_multiplier = 10;
    let mut sim = Sim::new(config);
    let develop = Action {
        market: vec![
            MarketOrder::Hire,
            MarketOrder::Hire,
            MarketOrder::Hire,
            MarketOrder::BuyLand,
            MarketOrder::BuyLand,
            MarketOrder::BuyLand,
            MarketOrder::BuyLand,
        ],
        ..Action::pass()
    };

    step(&mut sim, develop, Action::pass());

    let farm = &sim.state.farms[0];
    assert_eq!(farm.money, 2_960.0); // hires 10 + 10 + 20; land 1k + 2k + 4k
    assert_eq!(farm.hires_today, 3);
    assert_eq!(
        farm.hands,
        vec![
            Position { x: 5, y: 4 },
            Position { x: 4, y: 5 },
            Position { x: 5, y: 5 },
        ]
    );
    assert_eq!(farm.private.inventories.len(), 4);
    assert_eq!(
        farm.unlocked_quadrants,
        vec![Quadrant::Nw, Quadrant::Ne, Quadrant::Sw, Quadrant::Se]
    );
    assert!(farm.tiles.iter().all(|tile| tile == &Tile::Empty));
}

#[test]
fn hire_affordability_keeps_python_float_integer_comparison_exact() {
    let mut sim = Sim::new(Config {
        episode_steps: 2,
        starting_money: 1_i64 << 53,
        farm_hand_cost_multiplier: (1_i64 << 53) + 1,
        weed_spawn_chance: 0.0,
        ..Config::default()
    });
    step(
        &mut sim,
        Action {
            market: vec![MarketOrder::Hire],
            ..Action::pass()
        },
        Action::pass(),
    );
    assert_eq!(sim.state.farms[0].money, 9_007_199_254_740_992.0);
    assert_eq!(sim.state.farms[0].hires_today, 0);
    assert!(sim.state.farms[0].hands.is_empty());
}

#[test]
fn wheat_matures_with_water_while_an_unwatered_new_plant_dies() {
    let mut sim = Sim::new(deterministic_config(4));
    let buy_seed = Action {
        market: vec![MarketOrder::BuySeed {
            crop: Crop::Wheat,
            count: 1,
        }],
        ..Action::pass()
    };
    step(&mut sim, buy_seed.clone(), buy_seed);
    step(
        &mut sim,
        farmer_action(UnitAction::Plant(Crop::Wheat)),
        farmer_action(UnitAction::Plant(Crop::Wheat)),
    );
    step(&mut sim, farmer_action(UnitAction::Water), Action::pass());
    pass(&mut sim); // End day zero.

    assert!(matches!(
        tile_at_farmer(&sim, 0),
        Tile::Plant(plant)
            if plant.crop == Crop::Wheat
                && !plant.watered_today
                && plant.consecutive_unwatered == 0
                && plant.yield_units == 1
    ));
    assert_eq!(tile_at_farmer(&sim, 1), &Tile::Weed);

    // Day one: an early harvest is a no-op, but the plant must still be watered.
    step(&mut sim, farmer_action(UnitAction::Harvest), Action::pass());
    assert!(matches!(tile_at_farmer(&sim, 0), Tile::Plant(_)));
    step(&mut sim, farmer_action(UnitAction::Water), Action::pass());
    pass_turns(&mut sim, 2);

    // Day two is the first-yield day and the start of wheat's watering bonus
    // window, so watering raises the harvest from one unit to two.
    step(&mut sim, farmer_action(UnitAction::Water), Action::pass());
    assert!(matches!(
        tile_at_farmer(&sim, 0),
        Tile::Plant(plant) if plant.watered_today && plant.yield_units == 2
    ));
    step(&mut sim, farmer_action(UnitAction::Harvest), Action::pass());
    assert_eq!(tile_at_farmer(&sim, 0), &Tile::Empty);
    assert_eq!(
        sim.state.farms[0].private.inventories[0].get(Item::Wheat),
        2
    );
}

#[test]
fn goose_can_be_placed_fed_cared_harvested_and_then_escape() {
    let mut sim = Sim::new(deterministic_config(5));
    let stock_and_hire = Action {
        market: vec![
            MarketOrder::Hire,
            MarketOrder::BuyAnimal {
                animal: Animal::Goose,
                count: 1,
            },
            MarketOrder::BuyProduct {
                item: Item::Wheat,
                count: 10,
            },
        ],
        ..Action::pass()
    };
    step(&mut sim, stock_and_hire, Action::pass());

    // Bring the new hand onto the farmer's tile so the two units can place and
    // feed the goose in one turn.
    step(
        &mut sim,
        Action {
            farmer: UnitAction::BuildCoop,
            hands: vec![UnitAction::West],
            market: Vec::new(),
        },
        Action::pass(),
    );
    step(
        &mut sim,
        Action {
            farmer: UnitAction::Pickup {
                item: Item::Goose,
                count: 1,
            },
            hands: vec![UnitAction::Pickup {
                item: Item::Wheat,
                count: 1,
            }],
            market: Vec::new(),
        },
        Action::pass(),
    );
    step(
        &mut sim,
        Action {
            farmer: UnitAction::Place {
                item: Item::Goose,
                count: 1,
            },
            hands: vec![UnitAction::Feed],
            market: Vec::new(),
        },
        Action::pass(),
    );
    step(
        &mut sim,
        Action {
            farmer: UnitAction::Care,
            hands: vec![UnitAction::Pass],
            market: Vec::new(),
        },
        Action::pass(),
    ); // End day zero.

    let animal = match tile_at_farmer(&sim, 0) {
        Tile::Structure {
            kind: Structure::Coop,
            animal: Some(animal),
        } => animal,
        tile => panic!("expected a housed goose, got {tile:?}"),
    };
    assert_eq!(animal.animal, Animal::Goose);
    assert_eq!(animal.pending_care_bonus, 1);
    assert_eq!(animal.consecutive_unfed, 0);
    assert!(animal.fertilizer_available);

    // Feed, collect fertilizer, and care on days one through three. The first
    // goose production occurs at the end of day three and cashes in all three
    // previously banked care bonuses, reaching its held-output cap of four.
    for _ in 0..3 {
        step(
            &mut sim,
            farmer_action(UnitAction::Pickup {
                item: Item::Wheat,
                count: 1,
            }),
            Action::pass(),
        );
        step(&mut sim, farmer_action(UnitAction::Feed), Action::pass());
        step(
            &mut sim,
            farmer_action(UnitAction::CollectFertilizer),
            Action::pass(),
        );
        step(&mut sim, farmer_action(UnitAction::Care), Action::pass());
        pass(&mut sim);
    }

    let animal = match tile_at_farmer(&sim, 0) {
        Tile::Structure {
            kind: Structure::Coop,
            animal: Some(animal),
        } => animal,
        tile => panic!("expected a productive goose, got {tile:?}"),
    };
    assert_eq!(animal.yield_units, 4);
    assert_eq!(animal.pending_care_bonus, 1);
    assert_eq!(sim.state.farms[0].private.shed[Item::Fertilizer.index()], 3);

    step(&mut sim, farmer_action(UnitAction::Harvest), Action::pass());
    assert_eq!(sim.state.farms[0].private.inventories[0].get(Item::Egg), 4);

    // Missing feed at two consecutive end-of-day refreshes removes the animal,
    // while leaving its coop behind.
    pass_turns(&mut sim, 4);
    assert!(matches!(
        tile_at_farmer(&sim, 0),
        Tile::Structure {
            kind: Structure::Coop,
            animal: Some(animal),
        } if animal.consecutive_unfed == 1
    ));
    pass_turns(&mut sim, 5);
    assert_eq!(
        tile_at_farmer(&sim, 0),
        &Tile::Structure {
            kind: Structure::Coop,
            animal: None,
        }
    );
}

#[test]
fn default_episode_has_exactly_719_acting_transitions() {
    let mut sim = Sim::new(deterministic_config(24));

    pass_turns(&mut sim, 718);
    assert_eq!(
        (sim.state.step, sim.state.day, sim.state.hour),
        (718, 29, 22)
    );
    assert!(!sim.state.done);

    pass(&mut sim);
    assert_eq!(
        (sim.state.step, sim.state.day, sim.state.hour),
        (719, 29, 23)
    );
    assert!(sim.state.done);

    let terminal = sim.state.clone();
    step(
        &mut sim,
        Action {
            farmer: UnitAction::North,
            market: vec![MarketOrder::BuyLand],
            ..Action::pass()
        },
        Action::pass(),
    );
    assert_eq!(sim.state, terminal);
}

#[test]
fn one_step_episode_still_executes_one_transition() {
    let mut sim = Sim::new(Config {
        episode_steps: 1,
        weed_spawn_chance: 0.0,
        ..Config::default()
    });
    assert!(!sim.state.done);
    pass(&mut sim);
    assert_eq!(sim.state.step, 1);
    assert!(sim.state.done);
}

#[test]
fn default_hinge_prices_match_python_goldens() {
    let sim = Sim::default();
    let cases = [
        (Product::Carrot, 10_000, 35),
        (Product::Carrot, 9_550, 70),
        (Product::Carrot, 9_325, 158),
        (Product::Carrot, 9_100, 385),
        (Product::Tomato, 10_000, 60),
        (Product::Tomato, 9_800, 84),
        (Product::Tomato, 9_700, 144),
        (Product::Tomato, 9_600, 300),
        (Product::Egg, 10_000, 50),
        (Product::Egg, 9_668, 70),
        (Product::Egg, 9_502, 120),
        (Product::Egg, 9_336, 250),
    ];

    for (product, inventory, expected) in cases {
        assert_eq!(
            sim.market_price(product, inventory),
            expected,
            "{product:?} at inventory {inventory}"
        );
    }
}
