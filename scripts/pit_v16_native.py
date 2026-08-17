#!/usr/bin/env python3
"""Benchmark the current rule agent against V16 in one native Rust batch."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np

from bertani import VecEnv
from bertani.v16_native import NativeV16Policy, load_v16_actions
from bertani_rules.agent import build_policy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "baselines" / "v16_rc5" / "main.py"
DEFAULT_OUTPUT = ROOT / "outputs" / "rule-v16-native.json"


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


def summarize(
    seeds: list[int], rewards: np.ndarray, elapsed: float
) -> dict[str, Any]:
    pairs = []
    margins = []
    wins = ties = losses = 0
    for index, seed in enumerate(seeds):
        first = rewards[2 * index]
        second = rewards[2 * index + 1]
        first_margin = float(first[0] - first[1])
        second_margin = float(second[1] - second[0])
        pair_margin = first_margin + second_margin
        margins.extend((first_margin, second_margin))
        wins += int(first_margin > 0) + int(second_margin > 0)
        ties += int(first_margin == 0) + int(second_margin == 0)
        losses += int(first_margin < 0) + int(second_margin < 0)
        pairs.append(
            {
                "seed": seed,
                "a_as_seat_0": {
                    "a_reward": float(first[0]),
                    "b_reward": float(first[1]),
                    "a_margin": first_margin,
                },
                "a_as_seat_1": {
                    "a_reward": float(second[1]),
                    "b_reward": float(second[0]),
                    "a_margin": second_margin,
                },
                "paired_margin": pair_margin,
            }
        )
    return {
        "seeds": seeds,
        "pairs": pairs,
        "summary": {
            "seed_count": len(seeds),
            "games": 2 * len(seeds),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "mean_game_margin": statistics.fmean(margins),
            "worst_game_margin": min(margins),
            "elapsed_seconds": elapsed,
            "games_per_second": 2 * len(seeds) / elapsed,
        },
    }


def run_native_batch(
    seeds: list[int],
    baseline: str,
    weed_spawn_chance: float,
    scheduler: str = "route",
    profile: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one independent seed chunk and return rewards + diagnostics."""
    paired_seeds = np.repeat(np.asarray(seeds, dtype=np.uint64), 2)
    environment = VecEnv(
        len(paired_seeds),
        auto_reset=False,
        weed_spawn_chance=weed_spawn_chance,
    )
    batch = environment.reset(paired_seeds)
    rule = build_policy(scheduler_mode=scheduler, profile=profile)
    v16 = NativeV16Policy(
        load_v16_actions(Path(baseline)),
        max_orders=environment.max_orders,
    )
    v16.reset()
    unit_actions, market_actions, market_lengths = environment.clear_actions()
    games = np.arange(len(paired_seeds), dtype=np.int64)
    rule_seats = games % 2
    v16_seats = 1 - rule_seats

    rule_seat_mask = np.zeros((len(paired_seeds), 2), dtype=np.bool_)
    rule_seat_mask[games, rule_seats] = True

    rule_ns = 0
    v16_ns = 0
    env_ns = 0
    for _ in range(719):
        if profile:
            started = time.perf_counter_ns()
            rule_actions = rule.act(
                batch,
                max_orders=environment.max_orders,
                seat_mask=rule_seat_mask,
            )
            rule_ns += time.perf_counter_ns() - started

            started = time.perf_counter_ns()
            v16_actions = v16.act(batch)
            v16_ns += time.perf_counter_ns() - started
        else:
            rule_actions = rule.act(
                batch,
                max_orders=environment.max_orders,
                seat_mask=rule_seat_mask,
            )
            v16_actions = v16.act(batch)

        unit_actions[games, rule_seats] = rule_actions.unit_actions[
            games, rule_seats
        ]
        unit_actions[games, v16_seats] = v16_actions.unit_actions[
            games, v16_seats
        ]
        market_actions[games, rule_seats] = rule_actions.market_actions[
            games, rule_seats
        ]
        market_actions[games, v16_seats] = v16_actions.market_actions[
            games, v16_seats
        ]
        market_lengths[games, rule_seats] = rule_actions.market_lengths[
            games, rule_seats
        ]
        market_lengths[games, v16_seats] = v16_actions.market_lengths[
            games, v16_seats
        ]

        if profile:
            started = time.perf_counter_ns()
            batch = environment.step(unit_actions, market_actions, market_lengths)
            env_ns += time.perf_counter_ns() - started
        else:
            batch = environment.step(unit_actions, market_actions, market_lengths)

    if not batch.dones.all():
        raise RuntimeError("native benchmark did not reach terminal states")

    scheduler_obj = rule._task_scheduler
    diagnostics: dict[str, Any] = {
        "rule_ns": rule_ns,
        "v16_ns": v16_ns,
        "env_ns": env_ns,
        "rule_profile_ns": dict(rule.profile_ns),
        "scheduler_full_solves": (
            0 if scheduler_obj is None else int(scheduler_obj.full_solves)
        ),
        "scheduler_cache_hits": (
            0 if scheduler_obj is None else int(scheduler_obj.cache_hits)
        ),
        "scheduler_cache_miss_reasons": (
            {} if scheduler_obj is None else dict(scheduler_obj.cache_miss_reasons)
        ),
        "scheduler_force_replan_reasons": (
            {} if scheduler_obj is None else dict(scheduler_obj.force_replan_reasons)
        ),
    }
    return batch.rewards.copy(), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--num-seeds", type=positive_int, default=100)
    parser.add_argument("--seed-source", type=int, default=2026)
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=min(8, os.cpu_count() or 1),
        help="independent native batches to run concurrently (default: up to 8)",
    )
    parser.add_argument(
        "--rust-threads",
        type=positive_int,
        default=1,
        help=(
            "Rayon threads inside each worker (default: 1, avoiding nested "
            "parallelism when multiple workers are used)"
        ),
    )
    parser.add_argument(
        "--scheduler",
        choices=("route", "native"),
        default="route",
        help=(
            "task scheduler: current route-aware Python planner or the "
            "existing Rust priority-greedy fast path"
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="print coarse policy/environment timing diagnostics",
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weed-spawn-chance", type=float, default=0.005)
    args = parser.parse_args()
    # Each process owns a native vector environment. Let process-level batches
    # provide the parallelism by default instead of creating a Rayon pool in
    # every worker and oversubscribing the same CPU cores.
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
        results = [
            run_native_batch(
                chunks[0],
                str(args.baseline),
                args.weed_spawn_chance,
                args.scheduler,
                args.profile,
            )
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    run_native_batch,
                    chunk,
                    str(args.baseline),
                    args.weed_spawn_chance,
                    args.scheduler,
                    args.profile,
                )
                for chunk in chunks
            ]
            results = [future.result() for future in futures]
    elapsed = time.perf_counter() - started
    reward_chunks = [rewards for rewards, _ in results]
    diagnostics = [diagnostic for _, diagnostic in results]
    rewards = np.concatenate(reward_chunks, axis=0)
    result = summarize(seeds, rewards, elapsed)
    summary = result["summary"]
    print(
        f"{summary['games']} games ({len(seeds)} paired seeds, {workers} workers) "
        f"in {elapsed:.3f}s; {summary['games_per_second']:.1f} games/s"
    )
    print(
        f"{summary['wins']}W/{summary['ties']}T/{summary['losses']}L; "
        f"mean margin={summary['mean_game_margin']:+.1f}; "
        f"worst={summary['worst_game_margin']:+.0f}; "
        f"scheduler={args.scheduler}"
    )
    if args.profile:
        total_rule_ns = sum(d["rule_ns"] for d in diagnostics)
        total_v16_ns = sum(d["v16_ns"] for d in diagnostics)
        total_env_ns = sum(d["env_ns"] for d in diagnostics)
        component_ns: dict[str, int] = {}
        for diagnostic in diagnostics:
            for name, value in diagnostic["rule_profile_ns"].items():
                component_ns[name] = component_ns.get(name, 0) + int(value)
        denominator = max(1, total_rule_ns)
        component_text = ", ".join(
            f"{name}={value / 1e9:.2f}s ({100.0 * value / denominator:.0f}%)"
            for name, value in component_ns.items()
            if value
        )
        print(
            "profile summed worker CPU: "
            f"rule={total_rule_ns / 1e9:.2f}s, "
            f"v16={total_v16_ns / 1e9:.2f}s, "
            f"env={total_env_ns / 1e9:.2f}s"
        )
        if component_text:
            print(f"rule components: {component_text}")
        print(
            "scheduler calls: "
            f"full_solves={sum(d['scheduler_full_solves'] for d in diagnostics)}, "
            f"cache_hits={sum(d['scheduler_cache_hits'] for d in diagnostics)}"
        )
        miss_totals: dict[str, int] = {}
        force_totals: dict[str, int] = {}
        for diagnostic in diagnostics:
            for name, value in diagnostic["scheduler_cache_miss_reasons"].items():
                miss_totals[name] = miss_totals.get(name, 0) + int(value)
            for name, value in diagnostic["scheduler_force_replan_reasons"].items():
                force_totals[name] = force_totals.get(name, 0) + int(value)
        print(
            "scheduler cache misses: "
            + ", ".join(f"{name}={value}" for name, value in miss_totals.items())
        )
        print(
            "scheduler forced replans: "
            + ", ".join(f"{name}={value}" for name, value in force_totals.items())
        )
    output = args.json_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
