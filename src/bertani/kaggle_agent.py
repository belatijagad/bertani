"""Kaggle ``agent(obs, config)`` adapter for :class:`VectorRulePolicy`.

The policy is batch-first in local training. This module reconstructs its
public observation views for a one-environment batch, then serializes the
typed action rows back to Kaggriculture's submission format. It deliberately
does not depend on the Rust simulator extension.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .rule_based import RuleConfig, VectorRulePolicy
from .vec_env import Batch, Item, MarketOp, MaskViews, ObservationViews, UnitOp


ITEM_NAMES = tuple(item.name for item in Item)
CROP_NAMES = ITEM_NAMES[:5]
ANIMAL_NAMES = ITEM_NAMES[9:12]
PRODUCT_BASE_PRICES = (25, 35, 60, 120, 250, 50, 160, 200, 100)
PRODUCT_THRESHOLDS = (400, 450, 200, 100, 300, 332, 122, 105, 200)
QUADRANTS = ("NW", "NE", "SW", "SE")
SHOP_NAMES = (
    "BAKERY",
    "BRUNCH_SPOT",
    "FARMERS_MARKET",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "PIZZA_SHOP",
    "SMOOTHIE_SHOP",
    "YARN_STORE",
)
SHOP_PRODUCT_INDICES = (
    (5, 0),
    (5, 0, 3),
    (0, 1, 2, 3),
    (3, 6, 0),
    (1,),
    (6, 2, 0),
    (3, 6),
    (7,),
)
PolicyFactory = Callable[[RuleConfig], VectorRulePolicy]


def _get(value: object, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _config_value(configuration: object, key: str, default: Any) -> Any:
    value = _get(configuration, key, default)
    return default if value is None else value


def _rule_config(configuration: object) -> RuleConfig:
    return RuleConfig(
        episode_steps=int(_config_value(configuration, "episodeSteps", 720)),
        turns_per_day=int(_config_value(configuration, "turnsPerDay", 24)),
        starting_money=int(_config_value(configuration, "startingMoney", 3_000)),
        shed_capacity=int(_config_value(configuration, "shedCapacity", 100)),
        town_shop_unlock_interval=int(
            _config_value(configuration, "townShopUnlockInterval", 3)
        ),
        town_shop_sell_interval=int(
            _config_value(configuration, "townShopSellInterval", 4)
        ),
        town_center_sell_interval=int(
            _config_value(configuration, "townCenterSellInterval", 24)
        ),
    )


def _encode_tile(
    tile: object,
    output: np.ndarray[Any, np.dtype[np.float32]],
    *,
    day: int,
    step: int,
    episode_steps: int,
    turns_per_day: int,
) -> None:
    if tile is None:
        output[0] = 1.0
        return
    if tile == "LOCKED":
        output[1] = 1.0
        return
    kind = str(_get(tile, "kind", ""))
    if kind == "WEED":
        output[2] = 1.0
        return
    numeric = 14
    if kind == "PLANT":
        output[3] = 1.0
        crop = str(_get(tile, "crop", ""))
        if crop in CROP_NAMES:
            output[9 + CROP_NAMES.index(crop)] = 1.0
        age = max(0, day - int(_get(tile, "planted_day", day)))
        output[numeric] = age / max(1, (episode_steps + turns_per_day - 1) // turns_per_day)
        output[numeric + 1] = bool(_get(tile, "watered_today", False))
        output[numeric + 3] = int(_get(tile, "consecutive_unwatered", 0)) / 2.0
        output[numeric + 4] = int(_get(tile, "yield_units", 0)) / 6.0
        fertilized_until = int(_get(tile, "fertilized_until_day", -1))
        output[numeric + 5] = max(0, fertilized_until - day + 1) / 3.0
        lifespan = int(_get(tile, "max_lifespan_step", -1))
        if lifespan >= 0:
            output[numeric + 8] = max(0, lifespan - step) / max(1, episode_steps)
        first_yield = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10}.get(crop, 0)
        output[numeric + 9] = age >= first_yield and int(_get(tile, "yield_units", 0)) > 0
        return
    if kind in {"COOP", "PASTURE"}:
        animal = _get(tile, "animal")
        if animal is None:
            output[4 if kind == "COOP" else 5] = 1.0
            return
        animal_name = str(animal)
        if animal_name in ANIMAL_NAMES:
            output[6 + ANIMAL_NAMES.index(animal_name)] = 1.0
        age = max(0, day - int(_get(tile, "placed_day", day)))
        output[numeric] = age / max(1, (episode_steps + turns_per_day - 1) // turns_per_day)
        output[numeric + 1] = bool(_get(tile, "fed_today", False))
        output[numeric + 2] = bool(_get(tile, "cared_today", False))
        output[numeric + 3] = int(_get(tile, "consecutive_unfed", 0)) / 2.0
        output[numeric + 4] = int(_get(tile, "yield_units", 0)) / 6.0
        output[numeric + 6] = bool(_get(tile, "fertilizer_available", False))
        output[numeric + 7] = int(_get(tile, "pending_care_bonus", 0)) / 6.0
        output[numeric + 9] = int(_get(tile, "yield_units", 0)) > 0


def observation_batch(obs: object, config: RuleConfig) -> Batch:
    """Encode one Kaggle observation into the policy's named batch views."""
    seat = 1 if int(_get(obs, "player", 0)) == 1 else 0
    farms_raw = list(_get(obs, "farms", []) or [])
    own_farm = farms_raw[seat]
    hands = list(_get(own_farm, "hands", []) or [])
    max_units = 1 + len(hands)
    board_size = len(_get(own_farm, "tiles", []) or []) or 10
    step = int(_get(obs, "step", 0))
    day = int(_get(obs, "day", step // config.turns_per_day))

    global_features = np.zeros((1, 2, 42), dtype=np.float32)
    farms = np.zeros((1, 2, 2, 9), dtype=np.float32)
    tiles = np.zeros((1, 2, 2, board_size, board_size, 24), dtype=np.float32)
    units = np.zeros((1, 2, 2, max_units, 29), dtype=np.float32)
    private = np.zeros((1, 2, 17), dtype=np.float32)
    active_units = np.zeros((1, 2, max_units), dtype=np.bool_)

    last_step = max(1, config.episode_steps - 1)
    last_day = max(1, (config.episode_steps - 1) // config.turns_per_day)
    hour = step % config.turns_per_day
    clock = (
        step / last_step,
        day / last_day,
        hour / max(1, config.turns_per_day - 1),
        max(0, config.episode_steps - 1 - step) / last_step,
    )
    global_features[0, seat, :4] = clock
    market = _get(obs, "market", {}) or {}
    market_prices = _get(market, "prices", {}) or {}
    market_inventory = _get(market, "inventory", {}) or {}
    for index, (product, base, threshold) in enumerate(
        zip(ITEM_NAMES[:9], PRODUCT_BASE_PRICES, PRODUCT_THRESHOLDS)
    ):
        global_features[0, seat, 4 + 2 * index] = (
            float(_get(market_inventory, product, 10_000)) - 10_000
        ) / threshold
        global_features[0, seat, 5 + 2 * index] = float(
            _get(market_prices, product, base)
        ) / base
    unlocked_shops = list(
        _get(_get(obs, "town", {}), "unlocked_shops", []) or []
    )
    for index, shop in enumerate(SHOP_NAMES):
        global_features[0, seat, 22 + index] = unlocked_shops.count(shop) / 8.0

    shop_demand = np.zeros(9, dtype=np.float32)
    for shop, products in zip(SHOP_NAMES, SHOP_PRODUCT_INDICES):
        multiplier = 2 if len(products) == 1 else 1
        count = unlocked_shops.count(shop)
        for product_index in products:
            shop_demand[product_index] += count * multiplier
    global_features[0, seat, 30:39] = shop_demand / 16.0

    def turns_until_tick(interval: int) -> float:
        remainder = step % interval
        turns = 0 if remainder == 0 else interval - remainder
        return turns / interval

    global_features[0, seat, 39] = turns_until_tick(
        config.town_shop_sell_interval
    )
    global_features[0, seat, 40] = turns_until_tick(
        config.town_center_sell_interval
    )
    unlock_interval = config.town_shop_unlock_interval
    days_until_boundary = unlock_interval - day % unlock_interval
    turns_until_unlock = (
        days_until_boundary * config.turns_per_day - hour - 1
    )
    global_features[0, seat, 41] = turns_until_unlock / (
        unlock_interval * config.turns_per_day
    )

    for relative, farm_index in enumerate((seat, 1 - seat)):
        if farm_index >= len(farms_raw):
            continue
        farm = farms_raw[farm_index]
        farms[0, seat, relative, 0] = float(_get(farm, "money", 0.0)) / max(1, config.starting_money)
        position = _get(farm, "farmer", (0, 0))
        farms[0, seat, relative, 1:3] = np.asarray(position, dtype=np.float32) / max(1, board_size - 1)
        farm_hands = list(_get(farm, "hands", []) or [])
        farms[0, seat, relative, 3] = len(farm_hands) / max(1, max_units - 1)
        unlocked = set(_get(farm, "unlocked_quadrants", []) or [])
        farms[0, seat, relative, 4:8] = [quadrant in unlocked for quadrant in QUADRANTS]
        farms[0, seat, relative, 8] = int(_get(farm, "hires_today", 0)) / max_units
        for y, row in enumerate(_get(farm, "tiles", []) or []):
            for x, tile in enumerate(row):
                _encode_tile(
                    tile,
                    tiles[0, seat, relative, y, x],
                    day=day,
                    step=step,
                    episode_steps=config.episode_steps,
                    turns_per_day=config.turns_per_day,
                )
        positions = [_get(farm, "farmer", (0, 0)), *farm_hands]
        for unit, unit_position in enumerate(positions[:max_units]):
            units[0, seat, relative, unit, 0] = 1.0
            units[0, seat, relative, unit, 1] = unit == 0
            units[0, seat, relative, unit, 2:4] = np.asarray(
                unit_position, dtype=np.float32
            ) / max(1, board_size - 1)

    private_raw = _get(obs, "private", {}) or {}
    shed = _get(private_raw, "shed", {}) or {}
    seeds = _get(private_raw, "seeds", {}) or {}
    inventories = list(_get(private_raw, "inventories", []) or [])
    for item, name in enumerate(ITEM_NAMES):
        private[0, seat, item] = int(_get(shed, name, 0)) / config.shed_capacity
    for crop, name in enumerate(CROP_NAMES):
        private[0, seat, 12 + crop] = int(_get(seeds, name, 0)) / 10.0
    for unit, inventory in enumerate(inventories[:max_units]):
        units[0, seat, 0, unit, 4] = 1.0
        for item, name in enumerate(ITEM_NAMES):
            units[0, seat, 0, unit, 5 + item] = int(
                _get(inventory, name, 0)
            ) / config.shed_capacity
    active_units[0, seat, :max_units] = True

    unit_ops = np.zeros((1, 2, max_units, len(UnitOp)), dtype=np.bool_)
    unit_args = np.zeros((1, 2, max_units, len(UnitOp), len(Item)), dtype=np.bool_)
    unit_ops[0, seat] = True
    unit_args[0, seat] = True
    market_ops = np.ones((1, 2, len(MarketOp)), dtype=np.bool_)
    market_args = np.ones((1, 2, len(MarketOp), len(Item)), dtype=np.bool_)
    observations = np.empty((1, 2, 0), dtype=np.float32)
    action_masks = np.empty((1, 2, 0), dtype=np.bool_)
    return Batch(
        observations=observations,
        action_masks=action_masks,
        active_units=active_units,
        rewards=np.zeros((1, 2), dtype=np.float64),
        economic_values=np.zeros((1, 2), dtype=np.float64),
        terminal_economic_values=np.zeros((1, 2), dtype=np.float64),
        dones=np.zeros((1, 2), dtype=np.bool_),
        episode_ids=np.zeros(1, dtype=np.uint64),
        overflow=np.zeros((1, 2), dtype=np.bool_),
        observation_views=ObservationViews(global_features, farms, tiles, units, private),
        mask_views=MaskViews(unit_ops, unit_args, market_ops, market_args),
    )


def _unit_action(row: np.ndarray[Any, np.dtype[np.int64]]) -> list[object]:
    operation = UnitOp(int(row[0]))
    if operation in {UnitOp.PICKUP, UnitOp.PLACE}:
        action: list[object] = [operation.name, ITEM_NAMES[int(row[1])]]
        if int(row[2]) > 0:
            action.append(int(row[2]))
        return action
    if operation == UnitOp.PLANT:
        return [operation.name, CROP_NAMES[int(row[1])]]
    return [operation.name]


def _market_action(row: np.ndarray[Any, np.dtype[np.int64]]) -> list[object] | None:
    operation = MarketOp(int(row[0]))
    if operation == MarketOp.NONE:
        return None
    if operation in {MarketOp.HIRE, MarketOp.BUY_LAND}:
        return [operation.name]
    item = int(row[1])
    return [operation.name, ITEM_NAMES[item], int(row[2])]


class KaggleAgent:
    """Stateful Kaggle adapter parameterized by a version's policy factory."""

    def __init__(self, policy_factory: PolicyFactory) -> None:
        self.policy_factory = policy_factory
        self.policies: dict[int, VectorRulePolicy] = {}

    def __call__(
        self, obs: object, configuration: object = None
    ) -> dict[str, object]:
        seat = 1 if int(_get(obs, "player", 0)) == 1 else 0
        step = int(_get(obs, "step", 0))
        if step == 0 or seat not in self.policies:
            self.policies[seat] = self.policy_factory(_rule_config(configuration))
        policy = self.policies[seat]
        batch = observation_batch(obs, policy.config)
        actions = policy.act(
            batch,
            max_orders=int(
                _config_value(configuration, "maxMarketOrdersPerTurn", 10)
            ),
        )
        unit_rows = actions.unit_actions[0, seat]
        market_count = int(actions.market_lengths[0, seat])
        market = [
            encoded
            for row in actions.market_actions[0, seat, :market_count]
            if (encoded := _market_action(row)) is not None
        ]
        return {
            "farmer": _unit_action(unit_rows[0]),
            "hands": [_unit_action(row) for row in unit_rows[1:]],
            "market": market,
        }


def make_agent(policy_factory: PolicyFactory) -> KaggleAgent:
    """Create an isolated Kaggle entry point for one strategy version."""
    return KaggleAgent(policy_factory)


# An opening-only adapter retained as a minimal reusable default. Submission
# packages replace this with their selected version's build_policy factory.
agent = make_agent(VectorRulePolicy)


__all__ = ["KaggleAgent", "agent", "make_agent", "observation_batch"]
