//! JSON-compatible debug snapshots for Python callers.

use kaggriculture_core::{Farm, Inventory, Item, Position, State, Structure, Tile};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Convert a simulator state into ordinary Python containers and scalars.
///
/// The result intentionally contains no extension types or `NumPy` values, so it
/// can be passed directly to `json.dumps`. Enum-valued fields use their stable
/// numeric IDs. Tile kinds remain short strings because they form a tagged
/// union rather than a single enum in the core model.
pub(crate) fn state_snapshot<'py>(
    py: Python<'py>,
    state: &State,
    seed: u64,
    episode_id: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let snapshot = PyDict::new(py);
    snapshot.set_item("seed", seed)?;
    snapshot.set_item("episode_id", episode_id)?;
    snapshot.set_item("step", state.step)?;
    snapshot.set_item("day", state.day)?;
    snapshot.set_item("hour", state.hour)?;
    snapshot.set_item("done", state.done)?;

    let market = PyDict::new(py);
    market.set_item("inventory", state.market.inventory.to_vec())?;
    market.set_item("prices", state.market.prices.to_vec())?;
    snapshot.set_item("market", market)?;

    let town = PyDict::new(py);
    town.set_item(
        "unlocked_shops",
        state
            .town
            .unlocked_shops
            .iter()
            .map(|shop| shop.index())
            .collect::<Vec<_>>(),
    )?;
    snapshot.set_item("town", town)?;

    let farms = PyList::empty(py);
    for farm in &state.farms {
        farms.append(farm_snapshot(py, farm)?)?;
    }
    snapshot.set_item("farms", farms)?;

    Ok(snapshot)
}

fn farm_snapshot<'py>(py: Python<'py>, farm: &Farm) -> PyResult<Bound<'py, PyDict>> {
    let snapshot = PyDict::new(py);
    snapshot.set_item("money", farm.money)?;
    snapshot.set_item("farmer", position_snapshot(farm.farmer))?;
    snapshot.set_item(
        "hands",
        farm.hands
            .iter()
            .copied()
            .map(position_snapshot)
            .collect::<Vec<_>>(),
    )?;
    snapshot.set_item(
        "unlocked_quadrants",
        farm.unlocked_quadrants
            .iter()
            .map(|quadrant| usize::from(*quadrant as u8))
            .collect::<Vec<_>>(),
    )?;
    snapshot.set_item("hires_today", farm.hires_today)?;
    snapshot.set_item("shed", farm.private.shed.to_vec())?;
    snapshot.set_item("seeds", farm.private.seeds.to_vec())?;

    let inventories = PyList::empty(py);
    for inventory in &farm.private.inventories {
        inventories.append(inventory_snapshot(py, inventory)?)?;
    }
    snapshot.set_item("inventories", inventories)?;

    let board_size = farm.tiles.len().isqrt();
    debug_assert_eq!(board_size * board_size, farm.tiles.len());
    let rows = PyList::empty(py);
    for row in farm.tiles.chunks(board_size) {
        let tiles = PyList::empty(py);
        for tile in row {
            tiles.append(tile_snapshot(py, tile)?)?;
        }
        rows.append(tiles)?;
    }
    snapshot.set_item("tiles", rows)?;

    Ok(snapshot)
}

fn inventory_snapshot<'py>(py: Python<'py>, inventory: &Inventory) -> PyResult<Bound<'py, PyDict>> {
    let snapshot = PyDict::new(py);
    snapshot.set_item(
        "counts",
        Item::ALL
            .iter()
            .map(|&item| inventory.get(item))
            .collect::<Vec<_>>(),
    )?;
    snapshot.set_item(
        "insertion_order",
        inventory
            .iter()
            .map(|(item, _)| item.index())
            .collect::<Vec<_>>(),
    )?;
    Ok(snapshot)
}

fn tile_snapshot<'py>(py: Python<'py>, tile: &Tile) -> PyResult<Bound<'py, PyDict>> {
    let snapshot = PyDict::new(py);
    match tile {
        Tile::Empty => snapshot.set_item("kind", "EMPTY")?,
        Tile::Locked => snapshot.set_item("kind", "LOCKED")?,
        Tile::Weed => snapshot.set_item("kind", "WEED")?,
        Tile::Plant(plant) => {
            snapshot.set_item("kind", "PLANT")?;
            snapshot.set_item("crop", plant.crop.index())?;
            snapshot.set_item("planted_day", plant.planted_day)?;
            snapshot.set_item("watered_today", plant.watered_today)?;
            snapshot.set_item("consecutive_unwatered", plant.consecutive_unwatered)?;
            snapshot.set_item("yield_units", plant.yield_units)?;
            snapshot.set_item("max_lifespan_step", plant.max_lifespan_step)?;
            snapshot.set_item("fertilized_until_day", plant.fertilized_until_day)?;
        }
        Tile::Structure { kind, animal } => {
            snapshot.set_item("kind", structure_kind(*kind))?;
            if let Some(animal) = animal {
                snapshot.set_item("animal", animal.animal.index())?;
                snapshot.set_item("placed_day", animal.placed_day)?;
                snapshot.set_item("yield_units", animal.yield_units)?;
                snapshot.set_item("consecutive_unfed", animal.consecutive_unfed)?;
                snapshot.set_item("fed_today", animal.fed_today)?;
                snapshot.set_item("cared_today", animal.cared_today)?;
                snapshot.set_item("fertilizer_available", animal.fertilizer_available)?;
                snapshot.set_item("pending_care_bonus", animal.pending_care_bonus)?;
            }
        }
    }
    Ok(snapshot)
}

const fn position_snapshot(position: Position) -> [usize; 2] {
    [position.x, position.y]
}

const fn structure_kind(structure: Structure) -> &'static str {
    match structure {
        Structure::Coop => "COOP",
        Structure::Pasture => "PASTURE",
    }
}
