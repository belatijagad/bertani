//! Python extension and fixed-shape RL interface for Kaggriculture.

mod action;
mod encoding;
mod maintenance_tasks;
mod market_rule;
mod production_tasks;
mod route_scheduler;
mod snapshot;
mod vec_env;

use pyo3::prelude::*;

#[pymodule]
mod _rust {
    #[pymodule_export]
    use super::vec_env::NativeVecEnv;

    #[pymodule_export]
    use super::maintenance_tasks::propose_maintenance_tasks;

    #[pymodule_export]
    use super::market_rule::propose_rule_market;

    #[pymodule_export]
    use super::production_tasks::propose_production_tasks;

    #[pymodule_export]
    use super::route_scheduler::schedule_routes;

    #[pymodule_export]
    const UNIT_ACTION_COUNT: usize = super::action::UNIT_ACTION_COUNT;

    #[pymodule_export]
    const MARKET_ACTION_COUNT: usize = super::action::MARKET_ACTION_COUNT;

    #[pymodule_export]
    const ITEM_COUNT: usize = super::action::ITEM_COUNT;

    #[pymodule_export]
    const BUILD_PROFILE: &str = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };

    #[pymodule_export]
    const RL_API_VERSION: usize = 1;

    #[pymodule_export]
    const ITEM_WHEAT: i64 = super::action::ITEM_WHEAT;
    #[pymodule_export]
    const ITEM_CARROT: i64 = super::action::ITEM_CARROT;
    #[pymodule_export]
    const ITEM_TOMATO: i64 = super::action::ITEM_TOMATO;
    #[pymodule_export]
    const ITEM_STRAWBERRY: i64 = super::action::ITEM_STRAWBERRY;
    #[pymodule_export]
    const ITEM_MELON: i64 = super::action::ITEM_MELON;
    #[pymodule_export]
    const ITEM_EGG: i64 = super::action::ITEM_EGG;
    #[pymodule_export]
    const ITEM_MILK: i64 = super::action::ITEM_MILK;
    #[pymodule_export]
    const ITEM_WOOL: i64 = super::action::ITEM_WOOL;
    #[pymodule_export]
    const ITEM_FERTILIZER: i64 = super::action::ITEM_FERTILIZER;
    #[pymodule_export]
    const ITEM_GOOSE: i64 = super::action::ITEM_GOOSE;
    #[pymodule_export]
    const ITEM_COW: i64 = super::action::ITEM_COW;
    #[pymodule_export]
    const ITEM_SHEEP: i64 = super::action::ITEM_SHEEP;
}
