use std::io::Read;

use flate2::read::GzDecoder;
use kaggriculture_core::{
    Action, Animal, Config, Crop, Farm, Item, MarketOrder, Position, PrivateState, Product, Sim,
    Structure, Tile, UnitAction,
};
use serde_json::{Value, json};

const REFERENCE_VERSION: &str = "1.32.7";
const REFERENCE_SOURCE: &str = "kaggle_environments/envs/kaggriculture/kaggriculture.py";
const REFERENCE_SHA256: &str = "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e";
const TRACE_BYTES: &[u8] = include_bytes!("fixtures/starter_vs_pass_seed11.jsonl.gz");

#[test]
fn starter_vs_pass_seed_11_matches_all_reference_states() {
    let mut decoder = GzDecoder::new(TRACE_BYTES);
    let mut trace = String::new();
    decoder
        .read_to_string(&mut trace)
        .expect("checked-in reference trace must be valid gzip UTF-8");
    let mut lines = trace.lines();

    let header: Value = serde_json::from_str(
        lines
            .next()
            .expect("reference trace must start with a metadata record"),
    )
    .expect("reference metadata must be valid JSON");
    assert_reference_header(&header);

    let records: Vec<Value> = lines
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str(line)
                .unwrap_or_else(|error| panic!("reference state {index} is invalid JSON: {error}"))
        })
        .collect();
    assert_eq!(records.len(), 720, "fixture must contain all 720 states");

    let config = config_from_header(&header);
    let mut sim = Sim::new(config);
    assert_state_matches(&records[0]["state"], &sim, 0);

    // Kaggle stores the action for transition t -> t+1 on env.steps[t+1].
    for transition in 0..719 {
        let next_record = &records[transition + 1];
        let raw_actions = next_record["actions"]
            .as_array()
            .expect("each record must contain the two recorded actions");
        assert_eq!(
            raw_actions.len(),
            2,
            "state {} must contain exactly two actions",
            transition + 1
        );
        let actions = [
            decode_action(&raw_actions[0]),
            decode_action(&raw_actions[1]),
        ];
        sim.step([&actions[0], &actions[1]]);
        assert_state_matches(&next_record["state"], &sim, transition + 1);
    }

    assert!(sim.state.done);
    assert_eq!(sim.state.step, 719);
}

fn assert_reference_header(header: &Value) {
    assert_eq!(header["format"], "kaggriculture-reference-trace-v1");
    assert_eq!(header["kaggle_environments_version"], REFERENCE_VERSION);
    assert_eq!(header["source"], REFERENCE_SOURCE);
    assert_eq!(header["source_sha256"], REFERENCE_SHA256);
    assert_eq!(header["seed"], 11);
    assert_eq!(header["players"], json!(["starter", "pass"]));
    assert_eq!(header["state_count"], 720);
    assert_eq!(header["transition_count"], 719);
    assert_eq!(
        header["product_order"],
        json!([
            "WHEAT",
            "CARROT",
            "TOMATO",
            "STRAWBERRY",
            "MELON",
            "EGG",
            "MILK",
            "WOOL",
            "FERTILIZER"
        ])
    );
    assert_eq!(
        header["item_order"],
        json!([
            "WHEAT",
            "CARROT",
            "TOMATO",
            "STRAWBERRY",
            "MELON",
            "EGG",
            "MILK",
            "WOOL",
            "FERTILIZER",
            "GOOSE",
            "COW",
            "SHEEP"
        ])
    );
    assert_eq!(
        header["crop_order"],
        json!(["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"])
    );
}

fn config_from_header(header: &Value) -> Config {
    let raw = &header["configuration"];
    Config {
        episode_steps: json_u32(&raw["episodeSteps"], "episodeSteps"),
        board_size: json_usize(&raw["boardSize"], "boardSize"),
        starting_money: json_i64(&raw["startingMoney"], "startingMoney"),
        max_market_orders_per_turn: json_usize(
            &raw["maxMarketOrdersPerTurn"],
            "maxMarketOrdersPerTurn",
        ),
        turns_per_day: json_u32(&raw["turnsPerDay"], "turnsPerDay"),
        shed_capacity: json_i64(&raw["shedCapacity"], "shedCapacity"),
        weed_spawn_chance: raw["weedSpawnChance"]
            .as_f64()
            .expect("weedSpawnChance must be numeric"),
        town_shop_unlock_interval: json_u32(
            &raw["townShopUnlockInterval"],
            "townShopUnlockInterval",
        ),
        town_shop_sell_interval: json_u32(&raw["townShopSellInterval"], "townShopSellInterval"),
        town_center_sell_interval: json_u32(
            &raw["townCenterSellInterval"],
            "townCenterSellInterval",
        ),
        farm_hand_cost_multiplier: json_i64(&raw["farmHandCostMult"], "farmHandCostMult"),
        seed: json_u64(&raw["seed"], "seed"),
        ..Config::default()
    }
}

fn json_u64(value: &Value, field: &str) -> u64 {
    value
        .as_u64()
        .unwrap_or_else(|| panic!("{field} must be an unsigned integer"))
}

fn json_u32(value: &Value, field: &str) -> u32 {
    u32::try_from(json_u64(value, field)).unwrap_or_else(|_| panic!("{field} must fit in a u32"))
}

fn json_usize(value: &Value, field: &str) -> usize {
    usize::try_from(json_u64(value, field))
        .unwrap_or_else(|_| panic!("{field} must fit in a usize"))
}

fn json_i64(value: &Value, field: &str) -> i64 {
    value
        .as_i64()
        .unwrap_or_else(|| panic!("{field} must be an integer"))
}

fn assert_state_matches(reference: &Value, sim: &Sim, step: usize) {
    let rust_state = normalize_sim(sim);
    assert_json_eq(reference, &rust_state, &format!("state[{step}]"));
}

fn assert_json_eq(reference: &Value, actual: &Value, path: &str) {
    match (reference, actual) {
        (Value::Array(reference), Value::Array(actual)) => {
            assert_eq!(
                reference.len(),
                actual.len(),
                "length mismatch at {path}: reference={}, rust={}",
                reference.len(),
                actual.len()
            );
            for (index, (reference, actual)) in reference.iter().zip(actual).enumerate() {
                assert_json_eq(reference, actual, &format!("{path}[{index}]"));
            }
        }
        (Value::Object(reference), Value::Object(actual)) => {
            for (key, reference_value) in reference {
                let actual_value = actual.get(key).unwrap_or_else(|| {
                    panic!("missing Rust field at {path}.{key}");
                });
                assert_json_eq(reference_value, actual_value, &format!("{path}.{key}"));
            }
            for key in actual.keys() {
                assert!(
                    reference.contains_key(key),
                    "unexpected Rust field at {path}.{key}"
                );
            }
        }
        _ => assert_eq!(
            reference, actual,
            "value mismatch at {path}: reference={reference}, rust={actual}"
        ),
    }
}

fn normalize_sim(sim: &Sim) -> Value {
    let done = sim.state.done;
    let agents = sim
        .state
        .farms
        .iter()
        .enumerate()
        .map(|(player, farm)| {
            json!({
                "player": player,
                "status": if done { "DONE" } else { "ACTIVE" },
                "reward": if done { farm.money } else { 0.0 },
            })
        })
        .collect::<Vec<_>>();
    let farms = sim
        .state
        .farms
        .iter()
        .map(|farm| normalize_farm(farm, sim.config.board_size))
        .collect::<Vec<_>>();
    let privates = sim
        .state
        .farms
        .iter()
        .map(|farm| normalize_private(&farm.private))
        .collect::<Vec<_>>();

    json!({
        "step": sim.state.step,
        "day": sim.state.day,
        "hour": sim.state.hour,
        "agents": agents,
        "farms": farms,
        "privates": privates,
        "market": {
            "inventory": sim.state.market.inventory,
            "prices": sim.state.market.prices,
        },
        "town": {
            "unlocked_shops": sim.state.town.unlocked_shops
                .iter()
                .map(|shop| shop.as_str())
                .collect::<Vec<_>>(),
        },
    })
}

fn normalize_farm(farm: &Farm, board_size: usize) -> Value {
    let tiles = farm
        .tiles
        .chunks(board_size)
        .map(|row| row.iter().map(normalize_tile).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    json!({
        "money": farm.money,
        "tiles": tiles,
        "farmer": normalize_position(farm.farmer),
        "hands": farm.hands.iter().copied().map(normalize_position).collect::<Vec<_>>(),
        "unlocked_quadrants": farm.unlocked_quadrants
            .iter()
            .map(|quadrant| quadrant.as_str())
            .collect::<Vec<_>>(),
        "hires_today": farm.hires_today,
    })
}

fn normalize_position(position: Position) -> Value {
    json!([position.x, position.y])
}

fn normalize_private(private: &PrivateState) -> Value {
    let inventories = private
        .inventories
        .iter()
        .map(|inventory| {
            inventory
                .iter()
                .map(|(item, count)| json!([item.as_str(), count]))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    json!({
        "shed": private.shed,
        "seeds": private.seeds,
        "inventories": inventories,
    })
}

fn normalize_tile(tile: &Tile) -> Value {
    match tile {
        Tile::Empty => Value::Null,
        Tile::Locked => Value::String("LOCKED".to_owned()),
        Tile::Weed => json!({"kind": "WEED"}),
        Tile::Plant(plant) => json!({
            "kind": "PLANT",
            "crop": plant.crop.as_str(),
            "planted_day": plant.planted_day,
            "watered_today": plant.watered_today,
            "consecutive_unwatered": plant.consecutive_unwatered,
            "yield_units": plant.yield_units,
            "max_lifespan_step": plant.max_lifespan_step,
            "fertilized_until_day": plant.fertilized_until_day,
        }),
        Tile::Structure { kind, animal: None } => json!({
            "kind": structure_name(*kind),
        }),
        Tile::Structure {
            kind,
            animal: Some(animal),
        } => json!({
            "kind": structure_name(*kind),
            "animal": animal.animal.as_str(),
            "placed_day": animal.placed_day,
            "yield_units": animal.yield_units,
            "consecutive_unfed": animal.consecutive_unfed,
            "fed_today": animal.fed_today,
            "cared_today": animal.cared_today,
            "fertilizer_available": animal.fertilizer_available,
            "pending_care_bonus": animal.pending_care_bonus,
        }),
    }
}

fn structure_name(structure: Structure) -> &'static str {
    match structure {
        Structure::Coop => "COOP",
        Structure::Pasture => "PASTURE",
    }
}

fn decode_action(value: &Value) -> Action {
    let Some(action) = value.as_object() else {
        return Action::pass();
    };
    Action {
        farmer: action
            .get("farmer")
            .map_or(UnitAction::Pass, decode_unit_action),
        hands: action
            .get("hands")
            .and_then(Value::as_array)
            .map(|hands| hands.iter().map(decode_unit_action).collect())
            .unwrap_or_default(),
        market: action
            .get("market")
            .and_then(Value::as_array)
            .map(|orders| orders.iter().map(decode_market_order).collect())
            .unwrap_or_default(),
    }
}

fn decode_unit_action(value: &Value) -> UnitAction {
    let Some(parts) = value.as_array() else {
        return UnitAction::Pass;
    };
    let Some(op) = parts.first().and_then(Value::as_str) else {
        return UnitAction::Pass;
    };
    match op {
        "PASS" => UnitAction::default(),
        "NORTH" => UnitAction::North,
        "SOUTH" => UnitAction::South,
        "EAST" => UnitAction::East,
        "WEST" => UnitAction::West,
        "PICKUP" => parse_item(parts.get(1)).map_or(UnitAction::Pass, |item| UnitAction::Pickup {
            item,
            count: parts.get(2).and_then(Value::as_i64).unwrap_or(1),
        }),
        "DROP" => UnitAction::Drop,
        "PLACE" => parse_item(parts.get(1)).map_or(UnitAction::Pass, |item| UnitAction::Place {
            item,
            count: parts.get(2).and_then(Value::as_i64).unwrap_or(1),
        }),
        "PLANT" => parse_crop(parts.get(1)).map_or(UnitAction::Pass, UnitAction::Plant),
        "WATER" => UnitAction::Water,
        "HARVEST" => UnitAction::Harvest,
        "FERTILIZE" => UnitAction::Fertilize,
        "DIG" => UnitAction::Dig,
        "BUILD_COOP" => UnitAction::BuildCoop,
        "BUILD_PASTURE" => UnitAction::BuildPasture,
        "FEED" => UnitAction::Feed,
        "COLLECT_FERTILIZER" => UnitAction::CollectFertilizer,
        "CARE" => UnitAction::Care,
        _ => UnitAction::Pass,
    }
}

fn decode_market_order(value: &Value) -> MarketOrder {
    let no_op = || MarketOrder::Sell {
        item: Item::Wheat,
        count: 0,
    };
    let Some(parts) = value.as_array() else {
        return no_op();
    };
    let Some(op) = parts.first().and_then(Value::as_str) else {
        return no_op();
    };
    match op {
        "HIRE" => MarketOrder::Hire,
        "BUY_LAND" => MarketOrder::BuyLand,
        "BUY_SEED" => parse_crop(parts.get(1))
            .zip(parts.get(2).and_then(Value::as_i64))
            .map_or_else(no_op, |(crop, count)| MarketOrder::BuySeed { crop, count }),
        "BUY_PRODUCT" => parse_item(parts.get(1))
            .zip(parts.get(2).and_then(Value::as_i64))
            .map_or_else(no_op, |(item, count)| MarketOrder::BuyProduct {
                item,
                count,
            }),
        "BUY_ANIMAL" => parse_animal(parts.get(1))
            .zip(parts.get(2).and_then(Value::as_i64))
            .map_or_else(no_op, |(animal, count)| MarketOrder::BuyAnimal {
                animal,
                count,
            }),
        "SELL" => parse_item(parts.get(1))
            .zip(parts.get(2).and_then(Value::as_i64))
            .map_or_else(no_op, |(item, count)| MarketOrder::Sell { item, count }),
        _ => no_op(),
    }
}

fn parse_product(value: Option<&Value>) -> Option<Product> {
    match value?.as_str()? {
        "WHEAT" => Some(Product::Wheat),
        "CARROT" => Some(Product::Carrot),
        "TOMATO" => Some(Product::Tomato),
        "STRAWBERRY" => Some(Product::Strawberry),
        "MELON" => Some(Product::Melon),
        "EGG" => Some(Product::Egg),
        "MILK" => Some(Product::Milk),
        "WOOL" => Some(Product::Wool),
        "FERTILIZER" => Some(Product::Fertilizer),
        _ => None,
    }
}

fn parse_crop(value: Option<&Value>) -> Option<Crop> {
    match value?.as_str()? {
        "WHEAT" => Some(Crop::Wheat),
        "CARROT" => Some(Crop::Carrot),
        "TOMATO" => Some(Crop::Tomato),
        "STRAWBERRY" => Some(Crop::Strawberry),
        "MELON" => Some(Crop::Melon),
        _ => None,
    }
}

fn parse_animal(value: Option<&Value>) -> Option<Animal> {
    match value?.as_str()? {
        "GOOSE" => Some(Animal::Goose),
        "COW" => Some(Animal::Cow),
        "SHEEP" => Some(Animal::Sheep),
        _ => None,
    }
}

fn parse_item(value: Option<&Value>) -> Option<Item> {
    if let Some(product) = parse_product(value) {
        return Some(product.into());
    }
    parse_animal(value).map(Into::into)
}
