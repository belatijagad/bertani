"""Check Rust rule-market output against the pre-V10 Python implementation."""

from __future__ import annotations

import argparse
import numpy as np

from bertani.market import MarketPlanBatch
from bertani.vec_env import VecEnv, Item, MarketOp
from bertani_rules.agent import EconomyMarketRule, build_policy

# SHOP_NAMES order from bertani.kaggle_agent. Keeping the demand table beside
# the strategy avoids coupling the native batch policy to the Kaggle adapter.
SHOP_DEMAND = np.asarray(
    (
        # wheat, carrot, tomato, strawberry, melon, egg, milk, wool, fertilizer
        (1, 0, 0, 0, 0, 1, 0, 0, 0),  # bakery
        (1, 0, 0, 1, 0, 1, 0, 0, 0),  # brunch spot
        (1, 1, 1, 1, 0, 0, 0, 0, 0),  # farmers market
        (1, 0, 0, 1, 0, 0, 1, 0, 0),  # ice cream shop
        (0, 2, 0, 0, 0, 0, 0, 0, 0),  # pet cafe
        (1, 0, 1, 0, 0, 0, 1, 0, 0),  # pizza shop
        (0, 0, 0, 1, 0, 0, 1, 0, 0),  # smoothie shop
        (0, 0, 0, 0, 0, 0, 0, 2, 0),  # yarn store
    ),
    dtype=np.int64,
)

# Median requested order sizes in the downloaded 55463512 replays.
SALE_BATCHES = {
    Item.WHEAT: 7,
    Item.CARROT: 4,
    Item.STRAWBERRY: 8,
    Item.MELON: 12,
    Item.MILK: 6,
    Item.WOOL: 4,
    Item.FERTILIZER: 18,
}

# Product base prices indexed by Item. Current market-price observations are
# encoded as price/base, so these reconstruct the current quotes closely enough
# for small, targeted land-financing sales.
MARKET_BASE_PRICES = np.asarray(
    (25, 35, 60, 120, 250, 50, 160, 200, 100),
    dtype=np.float64,
)

SEED_BUY_BATCHES = {
    Item.WHEAT: 8,
    Item.CARROT: 4,
    Item.TOMATO: 4,
    Item.STRAWBERRY: 4,
    Item.MELON: 12,
}


class PythonEconomyMarketRule:
    """Fund daily work, replenish inputs, and sell inventory opportunistically."""

    def __init__(
        self,
        starting_money: int = 3_000,
        shed_capacity: int = 100,
        episode_steps: int = 720,
        turns_per_day: int = 24,
    ) -> None:
        self.starting_money = starting_money
        self.shed_capacity = shed_capacity
        self.turns_per_day = turns_per_day
        self.last_step = max(1, episode_steps - 1)
        self.episode_days = max(1, (episode_steps + turns_per_day - 1) // turns_per_day)

    def propose(
        self,
        batch: Batch,
        intent: StrategicIntent,
        plan: MarketPlanBatch,
    ) -> None:
        views = batch.observation_views
        active = ~intent.liquidate
        shed = np.rint(views.private[..., :12] * self.shed_capacity).astype(
            np.int64
        )
        seeds = np.rint(views.private[..., 12:17] * 10).astype(np.int64)
        tiles = views.tiles[:, :, 0]
        crops = np.rint(tiles[..., 9:14].sum(axis=(2, 3))).astype(np.int64)
        units = views.units[:, :, 0]
        carried = np.rint(units[..., 5:17] * self.shed_capacity).astype(
            np.int64
        )
        carried *= batch.active_units[..., None]
        carried_wheat = carried[..., Item.WHEAT].sum(axis=-1)
        money = views.farms[:, :, 0, 0] * self.starting_money
        hands = np.maximum(0, batch.active_units.sum(axis=-1) - 1)
        shed_total = shed.sum(axis=-1)
        pressure = shed_total >= int(self.shed_capacity * 0.7)
        ratios = views.global_features[..., 5:22:2]
        step = np.rint(
            views.global_features[..., 0] * self.last_step
        ).astype(np.int64)
        post_town_demand = (step % 4) == 1
        town_tick = (step % 4) == 0
        town_center_tick = (step % 24) == 0
        shops = np.rint(views.global_features[..., 22:30] * 8).astype(
            np.int64
        )
        farms = views.farms[:, :, 0]
        unlocked = np.rint(farms[..., 4:8].sum(axis=-1)).astype(np.int64)
        day = step // self.turns_per_day

        # Land expansion is scheduled rather than opportunistic:
        #   quadrant 2 -> day 7,  $1,000
        #   quadrant 3 -> day 11, $2,000
        #
        # On day 6/day 10 we only protect the future land bankroll from
        # discretionary livestock purchases. On the target day, if cash is
        # still slightly short, sell just enough premium inventory BEFORE
        # BUY_LAND. This preserves the existing economy instead of entering a
        # broad "expansion mode".
        next_land_cost = np.where(
            unlocked == 1,
            1_000,
            np.where(unlocked == 2, 2_000, 0),
        ).astype(np.int64)
        next_land_day = np.where(
            unlocked == 1,
            7,
            np.where(unlocked == 2, 11, -1),
        ).astype(np.int64)

        reserve_for_land = (
            active
            & (next_land_day >= 0)
            & (day == (next_land_day - 1))
        )
        target_land_day = (
            active
            & (next_land_day >= 0)
            & (day >= next_land_day)
        )

        land_shortfall = np.maximum(
            0,
            next_land_cost - money,
        ).astype(np.int64)

        current_prices = np.maximum(
            1,
            np.rint(ratios * MARKET_BASE_PRICES).astype(np.int64),
        )

        # Use Melon first because the existing strategy explicitly treats its
        # second cohort as bridge capital. If that is insufficient, use Wool.
        melon_price = current_prices[..., Item.MELON]
        melon_needed = np.where(
            melon_price > 0,
            (land_shortfall + melon_price - 1) // melon_price,
            0,
        )
        land_finance_melon = np.minimum(
            shed[..., Item.MELON],
            melon_needed,
        )
        estimated_after_melon = (
            money + land_finance_melon * melon_price
        )

        remaining_shortfall = np.maximum(
            0,
            next_land_cost - estimated_after_melon,
        ).astype(np.int64)
        wool_price = current_prices[..., Item.WOOL]
        wool_needed = np.where(
            wool_price > 0,
            (remaining_shortfall + wool_price - 1) // wool_price,
            0,
        )
        land_finance_wool = np.minimum(
            shed[..., Item.WOOL],
            wool_needed,
        )

        financing = target_land_day & (land_shortfall > 0)
        plan.append(
            financing & (land_finance_melon > 0),
            MarketOp.SELL,
            item=Item.MELON,
            count=land_finance_melon,
        )
        plan.append(
            financing & (land_finance_wool > 0),
            MarketOp.SELL,
            item=Item.WOOL,
            count=land_finance_wool,
        )

        estimated_land_cash = (
            money
            + np.where(financing, land_finance_melon * melon_price, 0)
            + np.where(financing, land_finance_wool * wool_price, 0)
        )
        scheduled_first_expansion = (
            target_land_day
            & (unlocked == 1)
            & (estimated_land_cash >= 1_000)
        )
        scheduled_third_expansion = (
            target_land_day
            & (unlocked == 2)
            & (estimated_land_cash >= 2_000)
        )

        # Keep the existing optional fourth-quadrant Yarn expansion unchanged.
        yarn_expansion_early = (
            active
            & (day >= 12)
            & (unlocked == 3)
            & (intent.target_animal_counts[..., 2] >= 12)
            & (money >= 4_000)
        )
        early_land_buy = (
            scheduled_first_expansion
            | scheduled_third_expansion
            | yarn_expansion_early
        )
        plan.append(early_land_buy, MarketOp.BUY_LAND)

        # Wool is the bridge between the livestock opening and the second
        # crop field. A normal four-unit sale is price-efficient later, but it
        # strands cash in the shed during expansion. The reference baseline
        # liquidates a large wool stack here and immediately converts it into
        # land, labor, and Strawberry seeds.
        expansion_financing = (
            active
            & (day >= 6)
            & (day <= 8)
            & (unlocked <= 2)
            & (money < 2_000)
        )

        # Sell in the leader's median batch sizes. Premium stock is normally
        # sold immediately after town demand. At a four-turn tick with no
        # demand for that product, selling now front-runs the opponent instead.
        ongoing_count = (tiles[..., 11] + tiles[..., 12]).sum(axis=(2, 3))
        fertilizer_reserve = np.minimum(
            9, np.ceil(ongoing_count / 3.0)
        ).astype(np.int64)
        fertilizer_surplus = np.maximum(
            0, shed[..., Item.FERTILIZER] - fertilizer_reserve
        )
        plan.append(
            active & (fertilizer_surplus > 0),
            MarketOp.SELL,
            item=Item.FERTILIZER,
            count=np.minimum(
                fertilizer_surplus, SALE_BATCHES[Item.FERTILIZER]
            ),
        )
        for item in (
            Item.MILK,
            Item.WOOL,
            Item.MELON,
            Item.STRAWBERRY,
            Item.CARROT,
        ):
            shop_demand = np.tensordot(
                shops, SHOP_DEMAND[:, int(item)], axes=([-1], [0])
            )
            demand_now = (
                town_center_tick.astype(np.int64)
                + np.where(town_tick, shop_demand, 0)
            )
            sale_window = post_town_demand | (town_tick & (demand_now == 0))
            normal_count = np.minimum(shed[..., item], SALE_BATCHES[item])
            count = np.where(
                (item == Item.WOOL) & expansion_financing,
                shed[..., item],
                normal_count,
            )
            sell = active & (count > 0) & (
                sale_window
                | pressure
                | ((item == Item.WOOL) & expansion_financing)
            )
            plan.append(sell, MarketOp.SELL, item=item, count=count)

        # These products are outside the intended strategy, but monetize any
        # accidental inventory instead of occupying shed space forever.
        for item in (Item.TOMATO, Item.EGG):
            count = shed[..., item]
            plan.append(
                active & (count > 0) & (post_town_demand | pressure),
                MarketOp.SELL,
                item=item,
                count=count,
            )

        wheat_surplus = np.maximum(0, shed[..., Item.WHEAT] - intent.wheat_reserve)
        wheat_count = np.minimum(wheat_surplus, SALE_BATCHES[Item.WHEAT])
        plan.append(
            active
            & (wheat_count > 0)
            & (((ratios[..., Item.WHEAT] >= 1.0) & post_town_demand) | pressure),
            MarketOp.SELL,
            item=Item.WHEAT,
            count=wheat_count,
        )

        # BUY_LAND was intentionally emitted before normal market sales so it
        # cannot be pushed past the ten-order cap. Keep this mask for downstream
        # cash budgeting.
        land_buy = early_land_buy

        # Expansion is processed before replenishing feed so a routine Wheat
        # purchase cannot consume the cash earmarked for the day's land unlock.
        wheat_owned = shed[..., Item.WHEAT] + carried_wheat
        wheat_shortfall = np.maximum(0, intent.wheat_reserve - wheat_owned)
        plan.append(
            active & (wheat_shortfall > 0),
            MarketOp.BUY_PRODUCT,
            item=Item.WHEAT,
            # Refill over multiple turns so feed cannot consume the entire
            # expansion bankroll before the premium seed order is processed.
            count=np.minimum(wheat_shortfall, 4),
        )

        animal_counts = np.rint(tiles[..., 6:9].sum(axis=(2, 3))).astype(
            np.int64
        )
        carried_animals = carried[..., Item.COW : Item.SHEEP + 1].sum(axis=-2)
        owned_cows = (
            animal_counts[..., 1]
            + shed[..., Item.COW]
            + carried_animals[..., 0]
        )
        owned_sheep = (
            animal_counts[..., 2]
            + shed[..., Item.SHEEP]
            + carried_animals[..., 1]
        )
        missing_cows = np.maximum(
            0, intent.target_animal_counts[..., 1] - owned_cows
        )
        missing_sheep = np.maximum(
            0, intent.target_animal_counts[..., 2] - owned_sheep
        )
        # Once the first expansion date arrives, do not let livestock consume
        # its cash while BUY_LAND is still unaffordable.
        expansion_ready = active & (
            (
                (day < 6)
                | (unlocked >= 2)
                | land_buy
            )
            & ~reserve_for_land
        )
        land_cost = np.choose(np.minimum(unlocked, 3), (0, 1_000, 2_000, 4_000))
        # Sheep expansion is more expensive to maintain than its purchase
        # price suggests. Preserve a feed-and-seed buffer in Yarn branches;
        # otherwise seat-order price differences can spend the bank down to a
        # few coins and cascade into every animal escaping.
        animal_cash_reserve = np.where(
            intent.target_animal_counts[..., 2] >= 8,
            800,
            np.where(day < 6, 0, 200),
        )
        scheduled_land_reserve = np.where(
            reserve_for_land,
            next_land_cost,
            0,
        )

        budget = np.maximum(
            0,
            money
            - animal_cash_reserve
            - scheduled_land_reserve
            - land_buy * land_cost,
        ).astype(np.int64)
        buy_sheep = np.minimum(missing_sheep, budget // 500)
        budget -= buy_sheep * 500
        buy_cows = np.minimum(missing_cows, budget // 400)
        establishing_second_field = (day >= 7) & (day <= 9)
        plan.append(
            expansion_ready & ~establishing_second_field & (buy_sheep > 0),
            MarketOp.BUY_ANIMAL,
            item=Item.SHEEP,
            count=buy_sheep,
        )
        plan.append(
            expansion_ready & ~establishing_second_field & (buy_cows > 0),
            MarketOp.BUY_ANIMAL,
            item=Item.COW,
            count=buy_cows,
        )
        # Secure a minimum useful workforce before seed orders.  Remaining
        # hires come after seeds so neither survival work nor planting can
        # monopolize the ten available market-order slots.
        missing_hands = np.maximum(0, intent.target_hands - hands)
        can_hire = (intent.target_hands > 0) & (money >= 12)
        essential_hands = 5
        for hire_index in range(
            min(essential_hands, int(intent.target_hands.max(initial=0)))
        ):
            plan.append(can_hire & (missing_hands > hire_index), MarketOp.HIRE)

        # Buy exactly the missing crop stock. The unit executor will plant only
        # seeds already visible at the start of a turn.  Crops one day from
        # harvest also get replacement stock, eliminating the post-harvest day
        # where their tiles previously sat empty.  Seed orders precede HIRE so
        # the ten-order market cap cannot starve planting indefinitely.
        crop_age = np.rint(
            tiles[..., 14] * self.episode_days
        ).astype(np.int64)
        crop_channels = tiles[..., 9:14] > 0.5
        # Only recurring short crops need replacement stock. The replay plants
        # one Strawberry and one Melon cohort rather than rolling them forever.
        replacement_ages = np.asarray((3, 2, 99, 99, 99), dtype=np.int64)
        replacement_seeds = (
            crop_channels & (crop_age[..., None] >= replacement_ages)
        ).sum(axis=(2, 3), dtype=np.int64)
        market_crop_targets = intent.target_crop_counts.copy()
        total_missing = np.maximum(
            0,
            market_crop_targets.sum(axis=-1)
            + replacement_seeds.sum(axis=-1)
            - crops.sum(axis=-1)
            - seeds.sum(axis=-1),
        )
        wheat_missing = np.maximum(
            0,
            market_crop_targets[..., Item.WHEAT]
            + replacement_seeds[..., Item.WHEAT]
            - crops[..., Item.WHEAT]
            - seeds[..., Item.WHEAT],
        )
        cash_seed_missing = np.maximum(0, total_missing - wheat_missing)
        for crop in (
            Item.WHEAT,
            Item.CARROT,
            Item.TOMATO,
            Item.STRAWBERRY,
            Item.MELON,
        ):
            if crop == Item.WHEAT:
                missing = wheat_missing
            else:
                preferred_deficit = np.maximum(
                    0,
                    market_crop_targets[..., crop] - crops[..., crop],
                )
                missing = np.minimum(cash_seed_missing, preferred_deficit)
                cash_seed_missing -= missing
            batch_limit = np.full(missing.shape, SEED_BUY_BATCHES[crop])
            if crop == Item.WHEAT:
                batch_limit = np.where(day >= 11, 12, batch_limit)
            elif crop == Item.STRAWBERRY:
                batch_limit = np.where(
                    (day >= 7) & (day <= 9), 9, batch_limit
                )
                batch_limit = np.where(day >= 11, 16, batch_limit)
            plan.append(
                active & (missing > 0),
                MarketOp.BUY_SEED,
                item=crop,
                count=np.minimum(missing, batch_limit),
            )

        # Field establishment takes precedence over adding livestock. A large
        # animal purchase here used to consume the cash for Strawberry seeds,
        # leaving the newly unlocked quadrant idle for most of day eight.
        plan.append(
            expansion_ready & establishing_second_field & (buy_sheep > 0),
            MarketOp.BUY_ANIMAL,
            item=Item.SHEEP,
            count=buy_sheep,
        )
        plan.append(
            expansion_ready & establishing_second_field & (buy_cows > 0),
            MarketOp.BUY_ANIMAL,
            item=Item.COW,
            count=buy_cows,
        )

        # Daily labor must be present early enough to finish survival and field
        # work. HIRE has no count argument and consumes one market slot per hand.
        for hire_index in range(
            essential_hands, int(intent.target_hands.max(initial=0))
        ):
            plan.append(can_hire & (missing_hands > hire_index), MarketOp.HIRE)


def compare_plan(old: MarketPlanBatch, new: MarketPlanBatch, turn: int) -> None:
    for name in ("actions", "lengths", "overflow"):
        left = getattr(old, name)
        right = getattr(new, name)
        if np.array_equal(left, right):
            continue
        mismatch = np.argwhere(left != right)
        first = tuple(int(x) for x in mismatch[0]) if mismatch.size else ()
        raise AssertionError(
            f"turn {turn} {name} mismatch at {first}: "
            f"python={left[first] if first else left} rust={right[first] if first else right}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--seed-source", type=int, default=2026)
    args = parser.parse_args()

    env = VecEnv(args.num_seeds, seed=args.seed_source, auto_reset=False, weed_spawn_chance=0.0)
    seeds = np.random.default_rng(args.seed_source).integers(
        0, np.iinfo(np.uint64).max, size=args.num_seeds, dtype=np.uint64
    )
    batch = env.reset(seeds)
    policy = build_policy()
    old_rule = PythonEconomyMarketRule()
    new_rule = EconomyMarketRule()
    old_plan = MarketPlanBatch.allocate(args.num_seeds, 2, env.max_orders)
    new_plan = MarketPlanBatch.allocate(args.num_seeds, 2, env.max_orders)

    checked = 0
    for turn in range(719):
        features = policy.extract_features(batch)
        planner = policy.intent_planner
        from_features = getattr(planner, "from_features")
        intent = from_features(batch, features)

        old_plan.clear()
        new_plan.clear()
        old_rule.propose(batch, intent, old_plan)
        new_rule.propose(batch, intent, new_plan)
        compare_plan(old_plan, new_plan, turn)
        checked += args.num_seeds * 2

        actions = policy.act(batch, max_orders=env.max_orders)
        batch = env.step(
            actions.unit_actions, actions.market_actions, actions.market_lengths
        )

    print(
        f"market parity passed: {args.num_seeds} environments x 719 turns "
        f"({checked} seat-turns)"
    )


if __name__ == "__main__":
    main()
