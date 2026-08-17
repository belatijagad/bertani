#!/usr/bin/env python3
"""Diagnose the economic gap between the current rule agent and V16.

This script is deliberately read-only: it does not change either policy. It runs
paired seeds (rule as each seat), records day-boundary state, every submitted
unit/market action, exact per-turn bank deltas, and terminal inventory, then
prints aggregate rule-vs-V16 comparisons and writes a JSON report.

The first pass focuses on *where* the gap opens rather than reconstructing every
mixed market transaction into per-item cash revenue. Exact cash attribution is
reported whenever a turn contains only sells or only purchases; mixed turns are
reported separately.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np

from bertani import Item, MarketOp, RuleConfig, UnitOp, VecEnv
from bertani.rule_based import RuleFeatures, extract_rule_features
from bertani.v16_native import NativeV16Policy, load_v16_actions
from bertani_rules.agent import build_policy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "baselines" / "v16_rc5" / "main.py"
DEFAULT_OUTPUT = ROOT / "outputs" / "rule-v16-economy.json"

CROPS = (Item.WHEAT, Item.CARROT, Item.TOMATO, Item.STRAWBERRY, Item.MELON)
ANIMALS = (Item.GOOSE, Item.COW, Item.SHEEP)
PRODUCTS = (
    Item.WHEAT,
    Item.CARROT,
    Item.TOMATO,
    Item.STRAWBERRY,
    Item.MELON,
    Item.EGG,
    Item.MILK,
    Item.WOOL,
    Item.FERTILIZER,
)


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def generated_seeds(count: int, source: int) -> list[int]:
    generator = np.random.default_rng(source)
    return generator.integers(
        0, np.iinfo(np.uint32).max, size=count, dtype=np.uint64
    ).astype(object).tolist()


def _role_for_seat(game: int, seat: int) -> str:
    return "rule" if seat == game % 2 else "v16"


def _item_name(raw: int) -> str:
    try:
        return Item(int(raw)).name
    except ValueError:
        return f"ITEM_{int(raw)}"


def _unit_op_name(raw: int) -> str:
    try:
        return UnitOp(int(raw)).name
    except ValueError:
        return f"UNIT_{int(raw)}"


def _market_op_name(raw: int) -> str:
    try:
        return MarketOp(int(raw)).name
    except ValueError:
        return f"MARKET_{int(raw)}"


def _animal_item(raw: Any) -> Item | None:
    if raw is None:
        return None
    value = int(raw)
    # Native snapshots store animal species as 0..2; public observations may
    # instead use shared Item IDs. Accept both so diagnostics remain robust.
    if 0 <= value < 3:
        value += int(Item.GOOSE)
    try:
        item = Item(value)
    except ValueError:
        return None
    return item if item in ANIMALS else None


def _snapshot_tile_metrics(farm: dict[str, Any]) -> dict[str, float]:
    result: Counter[str] = Counter()
    for row in farm.get("tiles", []):
        for tile in row:
            if tile is None:
                result["empty"] += 1
                continue
            if tile == "LOCKED":
                result["locked"] += 1
                continue
            if not isinstance(tile, dict):
                continue
            kind = str(tile.get("kind", "EMPTY"))
            if kind == "EMPTY":
                result["empty"] += 1
            elif kind == "WEED":
                result["weeds"] += 1
            else:
                result["occupied"] += 1
            if kind == "COOP":
                result["coop"] += 1
            elif kind == "PASTURE":
                result["pasture"] += 1
            if kind == "PLANT":
                crop = tile.get("crop")
                if crop is not None:
                    try:
                        crop_item = Item(int(crop))
                    except ValueError:
                        crop_item = None
                    if crop_item in CROPS:
                        result[f"field_yield_{crop_item.name}"] += int(
                            tile.get("yield_units", 0) or 0
                        )
                    if int(tile.get("consecutive_unwatered", 0) or 0) > 0:
                        result["plants_unwatered_streak"] += 1
            animal = _animal_item(tile.get("animal"))
            if animal is not None:
                product = {
                    Item.GOOSE: Item.EGG,
                    Item.COW: Item.MILK,
                    Item.SHEEP: Item.WOOL,
                }[animal]
                result[f"field_yield_{product.name}"] += int(
                    tile.get("yield_units", 0) or 0
                )
                result["pending_care_bonus"] += int(
                    tile.get("pending_care_bonus", 0) or 0
                )
                result["fertilizer_available"] += int(
                    bool(tile.get("fertilizer_available", False))
                )
                if int(tile.get("consecutive_unfed", 0) or 0) > 0:
                    result["animals_unfed_streak"] += 1
    return {key: float(value) for key, value in result.items()}


def _day_state_rows(
    env: VecEnv,
    features: RuleFeatures,
    day: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for game in range(env.num_envs):
        snapshot = env.state_snapshot(game)
        for seat in range(2):
            role = _role_for_seat(game, seat)
            farm = snapshot["farms"][seat]
            row: dict[str, Any] = {
                "game": game,
                "seat": seat,
                "role": role,
                "day": day,
                "money": float(features.money[game, seat]),
                "hands": float(len(farm.get("hands", []))),
                "land": float(len(farm.get("unlocked_quadrants", []))),
            }
            row.update(_snapshot_tile_metrics(farm))
            row["shed_total"] = float(features.shed[game, seat, :9].sum())
            row["seed_total"] = float(features.seeds[game, seat].sum())
            for crop_index, crop in enumerate(CROPS):
                row[f"crop_{crop.name}"] = float(
                    features.crop_counts[game, seat, crop_index]
                )
                row[f"seed_{crop.name}"] = float(
                    features.seeds[game, seat, crop_index]
                )
            for animal_index, animal in enumerate(ANIMALS):
                row[f"animal_{animal.name}"] = float(
                    features.animal_counts[game, seat, animal_index]
                )
            for item in PRODUCTS:
                row[f"shed_{item.name}"] = float(features.shed[game, seat, int(item)])
            rows.append(row)
    return rows


def _record_actions(
    counters: dict[tuple[int, str], Counter[str]],
    batch_active_units: np.ndarray,
    unit_actions: np.ndarray,
    market_actions: np.ndarray,
    market_lengths: np.ndarray,
    day: int,
) -> None:
    for game in range(unit_actions.shape[0]):
        for seat in range(2):
            role = _role_for_seat(game, seat)
            counter = counters[(day, role)]
            active_count = int(batch_active_units[game, seat].sum())
            counter["active_unit_turns"] += active_count
            for raw in unit_actions[game, seat, :active_count]:
                op_name = _unit_op_name(int(raw[0]))
                if op_name in {"NORTH", "SOUTH", "EAST", "WEST"}:
                    counter["unit_MOVE"] += 1
                else:
                    counter[f"unit_{op_name}"] += 1
            order_count = int(market_lengths[game, seat])
            counter["market_slots_used"] += order_count
            counter["market_turns"] += int(order_count > 0)
            for raw in market_actions[game, seat, :order_count]:
                op = MarketOp(int(raw[0]))
                counter[f"market_{op.name}_orders"] += 1
                if op in {
                    MarketOp.BUY_SEED,
                    MarketOp.BUY_PRODUCT,
                    MarketOp.BUY_ANIMAL,
                    MarketOp.SELL,
                }:
                    item_name = _item_name(int(raw[1]))
                    counter[f"market_{op.name}_{item_name}_qty"] += int(raw[2])


def _record_cash_deltas(
    counters: dict[tuple[int, str], Counter[str]],
    money_before: np.ndarray,
    money_after: np.ndarray,
    market_actions: np.ndarray,
    market_lengths: np.ndarray,
    overflow: np.ndarray,
    day: int,
) -> None:
    for game in range(market_actions.shape[0]):
        for seat in range(2):
            role = _role_for_seat(game, seat)
            counter = counters[(day, role)]
            delta = float(money_after[game, seat] - money_before[game, seat])
            counter["cash_delta_cents"] += int(round(delta * 100.0))
            counter["market_overflow_turns"] += int(bool(overflow[game, seat]))
            order_count = int(market_lengths[game, seat])
            ops = [
                MarketOp(int(raw[0]))
                for raw in market_actions[game, seat, :order_count]
                if int(raw[0]) != int(MarketOp.NONE)
            ]
            if not ops:
                continue
            has_sell = any(op == MarketOp.SELL for op in ops)
            has_non_sell = any(op != MarketOp.SELL for op in ops)
            if has_sell and not has_non_sell:
                counter["pure_sell_turns"] += 1
                counter["pure_sell_revenue_cents"] += int(round(delta * 100.0))
                sell_items = {
                    int(raw[1])
                    for raw in market_actions[game, seat, :order_count]
                    if int(raw[0]) == int(MarketOp.SELL)
                }
                if len(sell_items) == 1:
                    item = next(iter(sell_items))
                    item_name = _item_name(item)
                    quantity = sum(
                        int(raw[2])
                        for raw in market_actions[game, seat, :order_count]
                        if int(raw[0]) == int(MarketOp.SELL) and int(raw[1]) == item
                    )
                    counter[f"single_sell_revenue_{item_name}_cents"] += int(
                        round(delta * 100.0)
                    )
                    counter[f"single_sell_qty_{item_name}"] += quantity
            elif not has_sell and has_non_sell:
                counter["pure_buy_turns"] += 1
                counter["pure_buy_spend_cents"] += int(round(-delta * 100.0))
            else:
                counter["mixed_market_turns"] += 1
                counter["mixed_cash_delta_cents"] += int(round(delta * 100.0))


def _aggregate_state_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["day"]), str(row["role"]))].append(row)
    output: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    ignored = {"game", "seat", "role", "day"}
    for (day, role), group in grouped.items():
        metrics = sorted(set().union(*(row.keys() for row in group)) - ignored)
        summary: dict[str, float] = {"samples": float(len(group))}
        for metric in metrics:
            values = np.asarray([float(row.get(metric, 0.0)) for row in group])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_median"] = float(np.median(values))
        output[str(day)][role] = summary
    return dict(output)


def _merge_counters(
    destinations: dict[tuple[int, str], Counter[str]],
    source: dict[str, dict[str, int]],
) -> None:
    for raw_key, values in source.items():
        day_text, role = raw_key.split("|", 1)
        destinations[(int(day_text), role)].update(values)


def _serialize_counters(
    counters: dict[tuple[int, str], Counter[str]],
) -> dict[str, dict[str, int]]:
    return {
        f"{day}|{role}": dict(counter)
        for (day, role), counter in counters.items()
    }


def run_chunk(
    seeds: list[int], baseline: str, weed_spawn_chance: float
) -> dict[str, Any]:
    paired_seeds = np.repeat(np.asarray(seeds, dtype=np.uint64), 2)
    env = VecEnv(
        len(paired_seeds),
        auto_reset=False,
        weed_spawn_chance=weed_spawn_chance,
    )
    batch = env.reset(paired_seeds)
    rule = build_policy()
    v16 = NativeV16Policy(load_v16_actions(Path(baseline)), max_orders=env.max_orders)
    v16.reset()
    unit_actions, market_actions, market_lengths = env.clear_actions()
    games = np.arange(len(paired_seeds), dtype=np.int64)
    rule_seats = games % 2
    v16_seats = 1 - rule_seats
    rule_mask = np.zeros((len(paired_seeds), 2), dtype=np.bool_)
    rule_mask[games, rule_seats] = True

    feature_config = RuleConfig()
    feature_buffer: RuleFeatures | None = None
    state_rows: list[dict[str, Any]] = []
    counters: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)

    for turn in range(719):
        day = turn // 24
        feature_buffer = extract_rule_features(batch, feature_config, feature_buffer)
        if turn % 24 == 0:
            state_rows.extend(_day_state_rows(env, feature_buffer, day))

        money_before = feature_buffer.money.copy()
        active_before = batch.active_units.copy()
        rule_actions = rule.act(batch, max_orders=env.max_orders, seat_mask=rule_mask)
        v16_actions = v16.act(batch)
        unit_actions[games, rule_seats] = rule_actions.unit_actions[games, rule_seats]
        unit_actions[games, v16_seats] = v16_actions.unit_actions[games, v16_seats]
        market_actions[games, rule_seats] = rule_actions.market_actions[games, rule_seats]
        market_actions[games, v16_seats] = v16_actions.market_actions[games, v16_seats]
        market_lengths[games, rule_seats] = rule_actions.market_lengths[games, rule_seats]
        market_lengths[games, v16_seats] = v16_actions.market_lengths[games, v16_seats]

        _record_actions(
            counters,
            active_before,
            unit_actions,
            market_actions,
            market_lengths,
            day,
        )
        batch = env.step(unit_actions, market_actions, market_lengths)
        feature_buffer = extract_rule_features(batch, feature_config, feature_buffer)
        _record_cash_deltas(
            counters,
            money_before,
            feature_buffer.money,
            market_actions,
            market_lengths,
            batch.overflow,
            day,
        )

    if not batch.dones.all():
        raise RuntimeError("diagnostic batch did not reach terminal states")
    # Terminal state is labelled day 30 so it is distinct from the day-29 start.
    assert feature_buffer is not None
    state_rows.extend(_day_state_rows(env, feature_buffer, 30))
    return {
        "rewards": batch.rewards.copy().tolist(),
        "state_rows": state_rows,
        "actions": _serialize_counters(counters),
    }


def _action_summary(
    counters: dict[tuple[int, str], Counter[str]],
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (day, role), counter in sorted(counters.items()):
        result[str(day)][role] = {
            key: (value / 100.0 if key.endswith("_cents") else float(value))
            for key, value in counter.items()
        }
    return dict(result)


def _sum_role_actions(
    counters: dict[tuple[int, str], Counter[str]], role: str
) -> Counter[str]:
    total: Counter[str] = Counter()
    for (day, current_role), counter in counters.items():
        if current_role == role:
            total.update(counter)
    return total


def _mean_metric(
    states: dict[str, dict[str, dict[str, float]]], day: int, role: str, metric: str
) -> float:
    return float(states.get(str(day), {}).get(role, {}).get(f"{metric}_mean", 0.0))


def _print_cash_table(states: dict[str, dict[str, dict[str, float]]]) -> None:
    print("\nCASH / ASSET TRAJECTORY (means per game)")
    print("day   rule$      v16$      gap$    land R/V  cows R/V  sheep R/V  shed R/V")
    for day in [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29, 30]:
        if str(day) not in states:
            continue
        rule_money = _mean_metric(states, day, "rule", "money")
        v16_money = _mean_metric(states, day, "v16", "money")
        rule_land = _mean_metric(states, day, "rule", "land")
        v16_land = _mean_metric(states, day, "v16", "land")
        rule_cows = _mean_metric(states, day, "rule", "animal_COW")
        v16_cows = _mean_metric(states, day, "v16", "animal_COW")
        rule_sheep = _mean_metric(states, day, "rule", "animal_SHEEP")
        v16_sheep = _mean_metric(states, day, "v16", "animal_SHEEP")
        rule_shed = _mean_metric(states, day, "rule", "shed_total")
        v16_shed = _mean_metric(states, day, "v16", "shed_total")
        print(
            f"{day:>3} {rule_money:>9.0f} {v16_money:>9.0f} "
            f"{rule_money-v16_money:>+9.0f}   "
            f"{rule_land:.2f}/{v16_land:.2f}   "
            f"{rule_cows:.1f}/{v16_cows:.1f}   "
            f"{rule_sheep:.1f}/{v16_sheep:.1f}   "
            f"{rule_shed:.1f}/{v16_shed:.1f}"
        )


def _print_gap_days(
    states: dict[str, dict[str, dict[str, float]]],
    counters: dict[tuple[int, str], Counter[str]],
) -> None:
    rows = []
    for day in range(30):
        rule_cash = counters[(day, "rule")]["cash_delta_cents"] / 100.0
        v16_cash = counters[(day, "v16")]["cash_delta_cents"] / 100.0
        # Each role has the same number of game samples in the paired batch.
        samples = _mean_metric(states, day, "rule", "samples")
        if samples <= 0:
            samples = float(states.get(str(day), {}).get("rule", {}).get("samples", 1.0))
        # counters are totals; state samples are number of games for that role.
        n = float(states.get(str(day), {}).get("rule", {}).get("samples", 1.0))
        rows.append((day, rule_cash / n, v16_cash / n, (rule_cash - v16_cash) / n))
    print("\nDAYS WHERE THE CASH GAP WIDENS MOST")
    print("day  rule net$  v16 net$   delta")
    for day, rule_cash, v16_cash, delta in sorted(rows, key=lambda row: row[3])[:10]:
        print(f"{day:>3} {rule_cash:>10.0f} {v16_cash:>9.0f} {delta:>+8.0f}")


def _print_market_summary(
    counters: dict[tuple[int, str], Counter[str]],
    games_per_role: int,
) -> None:
    rule = _sum_role_actions(counters, "rule")
    v16 = _sum_role_actions(counters, "v16")
    print("\nMARKET / LABOR SUMMARY (mean per game)")
    print("metric                              rule        v16      delta")
    metrics = [
        "market_HIRE_orders",
        "market_BUY_LAND_orders",
        "market_slots_used",
        "market_overflow_turns",
        "pure_sell_revenue_cents",
        "pure_buy_spend_cents",
        "mixed_cash_delta_cents",
        "active_unit_turns",
        "unit_PASS",
        "unit_MOVE",
        "unit_HARVEST",
        "unit_WATER",
        "unit_FEED",
        "unit_CARE",
    ]
    for metric in metrics:
        scale = 100.0 if metric.endswith("_cents") else 1.0
        r = rule[metric] / scale / games_per_role
        v = v16[metric] / scale / games_per_role
        print(f"{metric.removesuffix('_cents'):<32} {r:>10.1f} {v:>10.1f} {r-v:>+10.1f}")

    print("\nORDERED QUANTITIES BY ITEM (mean per game)")
    print("operation/item                      rule        v16      delta")
    keys = sorted(
        key
        for key in set(rule) | set(v16)
        if key.startswith("market_") and key.endswith("_qty")
    )
    for key in keys:
        r = rule[key] / games_per_role
        v = v16[key] / games_per_role
        if r == 0 and v == 0:
            continue
        print(f"{key.removeprefix('market_').removesuffix('_qty'):<32} {r:>10.1f} {v:>10.1f} {r-v:>+10.1f}")

    print("\nREALIZED PRICE ON PURE SINGLE-PRODUCT SELL TURNS")
    print("product             rule      v16    delta")
    for item in PRODUCTS:
        name = item.name
        rule_qty = rule[f"single_sell_qty_{name}"]
        v16_qty = v16[f"single_sell_qty_{name}"]
        rule_price = (
            rule[f"single_sell_revenue_{name}_cents"] / 100.0 / rule_qty
            if rule_qty
            else 0.0
        )
        v16_price = (
            v16[f"single_sell_revenue_{name}_cents"] / 100.0 / v16_qty
            if v16_qty
            else 0.0
        )
        if not rule_qty and not v16_qty:
            continue
        print(f"{name:<16} {rule_price:>8.1f} {v16_price:>8.1f} {rule_price-v16_price:>+8.1f}")


def _print_terminal_inventory(states: dict[str, dict[str, dict[str, float]]]) -> None:
    print("\nTERMINAL UNSOLD SHED INVENTORY (mean units/game)")
    print("product             rule      v16    delta")
    for item in PRODUCTS:
        metric = f"shed_{item.name}"
        r = _mean_metric(states, 30, "rule", metric)
        v = _mean_metric(states, 30, "v16", metric)
        print(f"{item.name:<16} {r:>8.2f} {v:>8.2f} {r-v:>+8.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--num-seeds", type=positive_int, default=100)
    parser.add_argument("--seed-source", type=int, default=2026)
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=min(8, os.cpu_count() or 1),
        help="independent native batches to run concurrently",
    )
    parser.add_argument("--rust-threads", type=positive_int, default=1)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weed-spawn-chance", type=float, default=0.005)
    args = parser.parse_args()

    os.environ["RAYON_NUM_THREADS"] = str(args.rust_threads)
    seeds = args.seeds or generated_seeds(args.num_seeds, args.seed_source)
    workers = min(args.workers, len(seeds))
    chunks = [
        chunk.astype(object).tolist()
        for chunk in np.array_split(np.asarray(seeds, dtype=np.uint64), workers)
        if chunk.size
    ]
    started = time.perf_counter()
    if workers == 1:
        results = [run_chunk(chunks[0], str(args.baseline), args.weed_spawn_chance)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    run_chunk,
                    chunk,
                    str(args.baseline),
                    args.weed_spawn_chance,
                )
                for chunk in chunks
            ]
            results = [future.result() for future in futures]

    elapsed = time.perf_counter() - started
    state_rows: list[dict[str, Any]] = []
    counters: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    rewards = []
    for result in results:
        state_rows.extend(result["state_rows"])
        _merge_counters(counters, result["actions"])
        rewards.extend(result["rewards"])

    states = _aggregate_state_rows(state_rows)
    actions = _action_summary(counters)
    reward_array = np.asarray(rewards, dtype=np.float64)
    # Rule is seat 0 in even paired games and seat 1 in odd paired games.
    game_indices = np.arange(reward_array.shape[0])
    rule_seats = game_indices % 2
    v16_seats = 1 - rule_seats
    margins = reward_array[game_indices, rule_seats] - reward_array[game_indices, v16_seats]

    report = {
        "metadata": {
            "seed_source": args.seed_source,
            "seed_count": len(seeds),
            "games": 2 * len(seeds),
            "workers": workers,
            "elapsed_seconds": elapsed,
            "mean_margin": float(margins.mean()),
            "worst_margin": float(margins.min()),
        },
        "daily_state": states,
        "daily_actions": actions,
    }

    print(
        f"diagnosed {2 * len(seeds)} games ({len(seeds)} paired seeds) "
        f"in {elapsed:.2f}s; mean margin={margins.mean():+.1f}; worst={margins.min():+.0f}"
    )
    _print_cash_table(states)
    _print_gap_days(states, counters)
    _print_market_summary(counters, games_per_role=2 * len(seeds))
    _print_terminal_inventory(states)

    output = args.json_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
