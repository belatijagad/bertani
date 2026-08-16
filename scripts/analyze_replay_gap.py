#!/usr/bin/env python3
"""Compare the current rule trajectory with downloaded leaderboard replays."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

import numpy as np

from bertani import Item, MarketOp, UnitOp, VecEnv
from bertani.v16_native import NativeV16Policy, load_v16_actions
from bertani_rules.agent import build_policy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAYS = ROOT / "inputs" / "replays" / "カワシギ"
DEFAULT_BASELINE = ROOT / "baselines" / "v16_rc5" / "main.py"
DEFAULT_OUTPUT = ROOT / "outputs" / "replay-gap.json"
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)
DISPLAY_METRICS = (
    "money",
    "land",
    "occupied",
    "empty",
    "crop_WHEAT",
    "crop_CARROT",
    "crop_STRAWBERRY",
    "crop_MELON",
    "animal_COW",
    "animal_SHEEP",
    "seed_WHEAT",
    "seed_CARROT",
    "seed_STRAWBERRY",
    "action_PASS",
    "action_MOVE",
    "action_PLANT",
    "action_WATER",
)


def player_index(replay: dict[str, Any], name: str) -> int | None:
    names = replay.get("info", {}).get("TeamNames") or [
        agent.get("Name") for agent in replay.get("info", {}).get("Agents", [])
    ]
    try:
        return names.index(name)
    except ValueError:
        return None


def shop_signature(observation: dict[str, Any]) -> str:
    shops = observation.get("town", {}).get("unlocked_shops", [])
    counts = Counter(shops)
    if not counts:
        return "none"
    return "+".join(
        f"{name}x{count}" if count > 1 else name
        for name, count in sorted(counts.items())
    )


def summarize_farm(
    farm: dict[str, Any],
    private: dict[str, Any],
    *,
    numeric_items: bool,
) -> dict[str, float]:
    result: dict[str, float] = {
        "money": float(farm["money"]),
        "hands": float(len(farm.get("hands", []))),
        "land": float(len(farm.get("unlocked_quadrants", []))),
    }
    crop_counts = Counter[str]()
    animal_counts = Counter[str]()
    kinds = Counter[str]()
    for row in farm["tiles"]:
        for raw_tile in row:
            if raw_tile is None:
                kinds["EMPTY"] += 1
                continue
            if raw_tile == "LOCKED":
                kinds["LOCKED"] += 1
                continue
            tile = raw_tile
            kind = str(tile.get("kind", "EMPTY"))
            kinds[kind] += 1
            if kind == "PLANT":
                crop = tile.get("crop")
                crop_name = Item(int(crop)).name if numeric_items else str(crop)
                crop_counts[crop_name] += 1
            animal = tile.get("animal")
            if animal is not None:
                animal_name = (
                    Item(int(Item.GOOSE) + int(animal)).name
                    if numeric_items
                    else str(animal)
                )
                animal_counts[animal_name] += 1
    result["empty"] = float(kinds["EMPTY"])
    result["weeds"] = float(kinds["WEED"])
    result["occupied"] = float(
        100 - kinds["EMPTY"] - kinds["LOCKED"] - kinds["WEED"]
    )
    for crop in CROPS:
        result[f"crop_{crop}"] = float(crop_counts[crop])
    for animal in ANIMALS:
        result[f"animal_{animal}"] = float(animal_counts[animal])

    seeds = private.get("seeds", farm.get("seeds", {}))
    shed = private.get("shed", farm.get("shed", {}))
    if numeric_items:
        for crop in CROPS:
            result[f"seed_{crop}"] = float(seeds[int(Item[crop])])
        for product in PRODUCTS:
            result[f"shed_{product}"] = float(shed[int(Item[product])])
    else:
        for crop in CROPS:
            result[f"seed_{crop}"] = float(seeds.get(crop, 0))
        for product in PRODUCTS:
            result[f"shed_{product}"] = float(shed.get(product, 0))
    return result


def add_string_actions(
    totals: Counter[str], action: dict[str, Any] | None
) -> None:
    if not action:
        return
    unit_actions = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    for unit_action in unit_actions:
        op = unit_action[0] if unit_action else "PASS"
        totals["action_MOVE" if op in {"NORTH", "SOUTH", "EAST", "WEST"} else f"action_{op}"] += 1
    for order in action.get("market", []):
        if not order:
            continue
        op = order[0]
        totals[f"market_{op}"] += 1
        if op in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"} and len(order) >= 3:
            totals[f"market_{op}_{order[1]}_quantity"] += int(order[2])


def replay_rows(path: Path, player: str) -> list[dict[str, Any]]:
    replay = json.loads(path.read_text())
    seat = player_index(replay, player)
    if seat is None:
        return []
    rows: list[dict[str, Any]] = []
    turns_per_day = int(replay.get("configuration", {}).get("turnsPerDay", 24))
    for start in range(0, len(replay["steps"]), turns_per_day):
        state = replay["steps"][start][seat]
        observation = state["observation"]
        day = int(observation.get("day", start // turns_per_day))
        farm = observation["farms"][seat]
        row: dict[str, Any] = {
            "source": "leader",
            "replay": path.name,
            "day": day,
            "shops": shop_signature(observation),
            **summarize_farm(farm, observation.get("private", {}), numeric_items=False),
        }
        actions: Counter[str] = Counter()
        for step in replay["steps"][start : start + turns_per_day]:
            add_string_actions(actions, step[seat].get("action"))
        row.update({key: float(value) for key, value in actions.items()})
        rows.append(row)
    return rows


def add_native_actions(
    totals: Counter[str], unit_actions: np.ndarray, market_actions: np.ndarray
) -> None:
    for raw in unit_actions:
        op = UnitOp(int(raw[0])).name
        totals["action_MOVE" if op in {"NORTH", "SOUTH", "EAST", "WEST"} else f"action_{op}"] += 1
    for raw in market_actions:
        op = MarketOp(int(raw[0])).name
        totals[f"market_{op}"] += 1
        if op in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}:
            item = Item(int(raw[1])).name
            totals[f"market_{op}_{item}_quantity"] += int(raw[2])


def local_rows(seeds: Iterable[int], baseline: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for game_index, seed in enumerate(seeds):
        env = VecEnv(1, seed=seed, auto_reset=False)
        batch = env.reset(np.asarray([seed], dtype=np.uint64))
        rule = build_policy()
        opponent = NativeV16Policy(load_v16_actions(baseline), max_orders=env.max_orders)
        opponent.reset()
        rule_seat = game_index % 2
        opponent_seat = 1 - rule_seat
        unit_actions, market_actions, market_lengths = env.clear_actions()
        action_totals: Counter[str] = Counter()
        current_row: dict[str, Any] | None = None
        for turn in range(720):
            if turn % 24 == 0:
                if current_row is not None:
                    current_row.update(
                        {key: float(value) for key, value in action_totals.items()}
                    )
                    action_totals.clear()
                snapshot = env.state_snapshot(0)
                farm = snapshot["farms"][rule_seat]
                current_row = {
                    "source": "rule",
                    "replay": f"local-seed-{seed}-seat-{rule_seat}",
                    "day": turn // 24,
                    "shops": "+".join(
                        map(str, sorted(snapshot["town"]["unlocked_shops"]))
                    )
                    or "none",
                    **summarize_farm(farm, farm, numeric_items=True),
                }
                rows.append(current_row)
            if turn == 719:
                break
            rule_plan = rule.act(batch, max_orders=env.max_orders)
            opponent_plan = opponent.act(batch)
            unit_actions[0, rule_seat] = rule_plan.unit_actions[0, rule_seat]
            unit_actions[0, opponent_seat] = opponent_plan.unit_actions[0, opponent_seat]
            market_actions[0, rule_seat] = rule_plan.market_actions[0, rule_seat]
            market_actions[0, opponent_seat] = opponent_plan.market_actions[0, opponent_seat]
            market_lengths[0, rule_seat] = rule_plan.market_lengths[0, rule_seat]
            market_lengths[0, opponent_seat] = opponent_plan.market_lengths[0, opponent_seat]
            active_count = int(batch.active_units[0, rule_seat].sum())
            order_count = int(rule_plan.market_lengths[0, rule_seat])
            add_native_actions(
                action_totals,
                rule_plan.unit_actions[0, rule_seat, :active_count],
                rule_plan.market_actions[0, rule_seat, :order_count],
            )
            batch = env.step(unit_actions, market_actions, market_lengths)
        if current_row is not None:
            current_row.update(
                {key: float(value) for key, value in action_totals.items()}
            )
    return rows


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source"], int(row["day"]))].append(row)
    output: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    ignored = {"source", "replay", "day", "shops"}
    for (source, day), group in grouped.items():
        metrics = sorted(set().union(*(row.keys() for row in group)) - ignored)
        summary: dict[str, float] = {"samples": float(len(group))}
        for metric in metrics:
            values = [float(row.get(metric, 0.0)) for row in group]
            summary[f"{metric}_p25"] = percentile(values, 0.25)
            summary[f"{metric}_median"] = float(statistics.median(values))
            summary[f"{metric}_p75"] = percentile(values, 0.75)
        output[source][str(day)] = summary
    return dict(output)


def shop_clusters(rows: list[dict[str, Any]], day: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["source"] == "leader" and row["day"] == day]
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        clusters[row["shops"]].append(row)
    output = []
    for shops, group in sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0])):
        output.append(
            {
                "shops": shops,
                "samples": len(group),
                "median": {
                    metric: float(statistics.median(float(row.get(metric, 0)) for row in group))
                    for metric in DISPLAY_METRICS
                },
            }
        )
    return output


def print_gap(summary: dict[str, Any], days: list[int]) -> None:
    print("day metric                 leader    rule    delta")
    for day in days:
        leader = summary.get("leader", {}).get(str(day), {})
        rule = summary.get("rule", {}).get(str(day), {})
        if not leader or not rule:
            continue
        for metric in DISPLAY_METRICS:
            top = leader.get(f"{metric}_median", 0.0)
            ours = rule.get(f"{metric}_median", 0.0)
            print(f"{day:>3} {metric:<21} {top:>8.1f} {ours:>8.1f} {ours-top:>8.1f}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, default=DEFAULT_REPLAYS)
    parser.add_argument("--player", default=None)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--local-seeds", nargs="*", type=int, default=[100, 2026, 451781128])
    parser.add_argument("--days", nargs="*", type=int, default=list(range(3, 15)))
    parser.add_argument("--json-output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    player = args.player or args.replays.name
    paths = sorted(args.replays.glob("*.json"))
    if not paths:
        raise SystemExit(f"no replay JSON files found under {args.replays}")
    leader_rows = [row for path in paths for row in replay_rows(path, player)]
    used_replays = len({row["replay"] for row in leader_rows})
    current_rows = local_rows(args.local_seeds, args.baseline)
    rows = leader_rows + current_rows
    summary = aggregate(rows)
    report = {
        "player": player,
        "leader_replays": used_replays,
        "skipped_files": len(paths) - used_replays,
        "local_seeds": args.local_seeds,
        "daily": summary,
        "shop_clusters": {
            str(day): shop_clusters(leader_rows, day) for day in args.days
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print_gap(summary, args.days)
    print(
        f"leader_replays={used_replays} skipped_files={len(paths)-used_replays} "
        f"local_games={len(args.local_seeds)}"
    )
    print(f"wrote {args.json_output}")


if __name__ == "__main__":
    main()
