//! A deterministic Rust rules engine for Kaggle's Kaggriculture environment.
//!
//! The implementation follows `kaggle-environments==1.32.7`.  The public core
//! API is intentionally independent of Python: construct a [`Sim`], then call
//! [`Sim::step`] with one [`Action`] per player.

mod action;
mod constants;
mod engine;
mod rng;
mod state;

pub use action::{Action, MarketOrder, UnitAction};
pub use constants::{Animal, Crop, Item, Product, Quadrant, Shop, Structure};
pub use engine::Sim;
pub use state::{
    Config, DEFAULT_MARKET_PARAMS, Farm, Inventory, Market, MarketParams, PlacedAnimal, Plant,
    Position, PrivateState, Shape, State, Tile, Town,
};
