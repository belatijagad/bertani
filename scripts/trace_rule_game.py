#!/usr/bin/env python3
"""Trace one current-rule-vs-V16 Kaggriculture game to detailed JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bertani import VecEnv
from bertani.tasks import TaskKind, WorkRole
from bertani.vec_env import Item, MarketOp, UnitOp
from bertani.v16_native import NativeV16Policy, load_v16_actions
from bertani_rules.agent import build_policy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "baselines" / "v16_rc5" / "main.py"


def enum_name(enum_type, value: int) -> str:
    try:
        return enum_type(int(value)).name
    except ValueError:
        return str(int(value))


def finite_float(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def item_name(value: int) -> str | None:
    if int(value) < 0:
        return None
    return enum_name(Item, int(value))


def decode_task(tasks, env: int, player: int, slot: int) -> dict[str, Any]:
    return {
        "slot": int(slot),
        "kind": enum_name(TaskKind, int(tasks.kind[env, player, slot])),
        "kind_id": int(tasks.kind[env, player, slot]),
        "target": [
            int(tasks.target_x[env, player, slot]),
            int(tasks.target_y[env, player, slot]),
        ],
        "item": item_name(int(tasks.item[env, player, slot])),
        "item_id": int(tasks.item[env, player, slot]),
        "quantity": int(tasks.quantity[env, player, slot]),
        "priority": finite_float(tasks.priority[env, player, slot]),
        "deadline": int(tasks.deadline[env, player, slot]),
        "estimated_value": finite_float(tasks.estimated_value[env, player, slot]),
        "required_item": item_name(int(tasks.required_item[env, player, slot])),
        "required_item_id": int(tasks.required_item[env, player, slot]),
        "required_count": int(tasks.required_count[env, player, slot]),
        "exclusive": bool(tasks.exclusive[env, player, slot]),
        "work_role": enum_name(WorkRole, int(tasks.work_role[env, player, slot])),
        "is_global": bool(slot >= tasks.tile_slots),
    }


def decode_unit_action(row: np.ndarray) -> dict[str, Any]:
    op = int(row[0])
    arg = int(row[1])
    count = int(row[2])
    return {
        "op": enum_name(UnitOp, op),
        "op_id": op,
        "arg": item_name(arg) if arg > 0 else None,
        "arg_id": arg,
        "count": count,
    }


def decode_market_action(row: np.ndarray) -> dict[str, Any]:
    op = int(row[0])
    item = int(row[1])
    count = int(row[2])
    return {
        "op": enum_name(MarketOp, op),
        "op_id": op,
        "item": item_name(item) if item >= 0 else None,
        "item_id": item,
        "count": count,
    }


def live_unit_count(batch, env: int, player: int) -> int:
    active = np.flatnonzero(batch.active_units[env, player])
    return int(active[-1]) + 1 if active.size else 0


def current_intent(rule, batch):
    features = rule._features
    planner = rule.intent_planner
    if features is None or planner is None:
        return None
    from_features = getattr(planner, "from_features", None)
    if from_features is not None:
        return from_features(batch, features)
    return rule.plan(batch)


def trace_frame(rule, batch, actions, *, env: int, player: int) -> dict[str, Any]:
    features = rule._features
    tasks = rule.last_tasks
    assignments = rule.last_assignments
    scheduler = rule._task_scheduler
    intent = current_intent(rule, batch)

    if features is None:
        step = int(round(float(batch.observation_views.global_features[env, player, 0]) * 719))
        day, hour = divmod(step, 24)
        money = None
    else:
        step = int(features.step[env, player])
        day = int(features.day[env, player])
        hour = int(features.hour[env, player])
        money = float(features.money[env, player])

    unit_limit = live_unit_count(batch, env, player)
    units_view = batch.observation_views.units
    private = batch.observation_views.private

    workers = []
    for unit in range(unit_limit):
        if not bool(batch.active_units[env, player, unit]):
            continue
        # units is [env, observer, farm, unit, channels].
        # farm=0 is the controlled player's own farm. Position channels 2/3
        # are normalized by (board_size - 1), matching the native scheduler.
        raw_unit_array = units_view[env, player, 0, unit]
        raw_unit = raw_unit_array.tolist()
        board_size = int(batch.observation_views.tiles.shape[-1])
        scale = max(1, board_size - 1)
        assigned = -1 if assignments is None else int(assignments.task_index[env, player, unit])
        workers.append({
            "unit": unit,
            "position": [
                int(round(float(raw_unit_array[2]) * scale)),
                int(round(float(raw_unit_array[3]) * scale)),
            ],
            "raw_observation": raw_unit,
            "assigned_task": assigned,
            "assignment_score": (
                None if assignments is None
                else finite_float(assignments.score[env, player, unit])
            ),
            "action": decode_unit_action(actions.unit_actions[env, player, unit]),
        })

    active_tasks = []
    if tasks is not None:
        for slot in np.flatnonzero(tasks.active[env, player]):
            active_tasks.append(decode_task(tasks, env, player, int(slot)))

    routes = []
    if scheduler is not None and tasks is not None:
        for unit, route in scheduler.debug_routes(env, player):
            routes.append({
                "unit": unit,
                "task_slots": route,
                "tasks": [
                    decode_task(tasks, env, player, slot)
                    for slot in route
                    if 0 <= slot < tasks.capacity
                ],
            })

    market_len = int(actions.market_lengths[env, player])
    market = [
        decode_market_action(actions.market_actions[env, player, order])
        for order in range(market_len)
    ]

    frame = {
        "step": step,
        "day": day,
        "hour": hour,
        "money": money,
        "workers": workers,
        "tasks": active_tasks,
        "routes": routes,
        "market_actions": market,
        "private_raw": private[env, player].tolist(),
    }

    if intent is not None:
        frame["intent"] = {
            "phase": int(intent.phase[env, player]),
            "target_hands": int(intent.target_hands[env, player]),
            "cash_reserve": float(intent.cash_reserve[env, player]),
            "wheat_reserve": int(intent.wheat_reserve[env, player]),
            "target_crop_counts": intent.target_crop_counts[env, player].astype(int).tolist(),
            "target_animal_counts": intent.target_animal_counts[env, player].astype(int).tolist(),
            "liquidate": bool(intent.liquidate[env, player]),
        }

    if scheduler is not None:
        frame["scheduler_cumulative"] = {
            "full_solves": scheduler.full_solves,
            "cache_hits": scheduler.cache_hits,
            "idle_worker_steals": scheduler.idle_worker_steals,
            "cache_miss_reasons": scheduler.cache_miss_reasons,
            "force_replan_reasons": scheduler.force_replan_reasons,
        }

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weed-spawn-chance", type=float, default=0.005)
    args = parser.parse_args()

    env = VecEnv(1, auto_reset=False, weed_spawn_chance=args.weed_spawn_chance)
    batch = env.reset(np.asarray([args.seed], dtype=np.uint64))

    rule = build_policy(profile=False)
    baseline = NativeV16Policy(
        load_v16_actions(args.baseline),
        max_orders=env.max_orders,
    )
    baseline.reset()

    unit_actions, market_actions, market_lengths = env.clear_actions()
    rule_mask = np.zeros((1, 2), dtype=np.bool_)
    rule_mask[0, args.seat] = True
    other = 1 - args.seat
    frames = []

    for _ in range(719):
        rule_actions = rule.act(batch, max_orders=env.max_orders, seat_mask=rule_mask)
        baseline_actions = baseline.act(batch)

        frames.append(trace_frame(rule, batch, rule_actions, env=0, player=args.seat))

        unit_actions[0, args.seat] = rule_actions.unit_actions[0, args.seat]
        market_actions[0, args.seat] = rule_actions.market_actions[0, args.seat]
        market_lengths[0, args.seat] = rule_actions.market_lengths[0, args.seat]

        unit_actions[0, other] = baseline_actions.unit_actions[0, other]
        market_actions[0, other] = baseline_actions.market_actions[0, other]
        market_lengths[0, other] = baseline_actions.market_lengths[0, other]

        batch = env.step(unit_actions, market_actions, market_lengths)

    # Match scripts/pit_v16_native.py: dones is batched/per-player, so the
    # environment is terminal only when all player done flags are true.
    if not bool(batch.dones[0].all()):
        raise RuntimeError(
            f"trace game did not reach terminal state; dones={batch.dones[0].tolist()}"
        )

    rewards = batch.rewards[0].astype(float).tolist()
    rule_reward = rewards[args.seat]
    baseline_reward = rewards[other]
    output = {
        "seed": args.seed,
        "rule_seat": args.seat,
        "rule_reward": rule_reward,
        "baseline_reward": baseline_reward,
        "rule_margin": rule_reward - baseline_reward,
        "frames": frames,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        f"seed={args.seed} seat={args.seat} "
        f"rule={rule_reward:.0f} baseline={baseline_reward:.0f} "
        f"margin={rule_reward - baseline_reward:+.0f}"
    )
    print(f"frames={len(frames)} wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
