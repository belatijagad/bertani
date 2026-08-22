#!/usr/bin/env python3
"""Benchmark the native simulator and Python-authored rule policies."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

from bertani import RuleConfig, VecEnv
from bertani.rule_based import extract_rule_features
from bertani_rules.agent import build_policy as build_current_policy
from bertani_rules.strategies.simple import build_policy as build_simple_policy
from bertani_rules.strategy import PythonRulePlanner

ACTING_TRANSITIONS = 719


def positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def benchmark_pass(num_envs: int, seed: int) -> float:
    """Measure native simulation through one batched Python step call."""

    environment = VecEnv(num_envs, seed=seed, auto_reset=False)
    environment.reset()
    started = time.perf_counter()
    for _ in range(ACTING_TRANSITIONS):
        environment.step()
    elapsed = time.perf_counter() - started
    return num_envs / elapsed


def benchmark_policy(
    num_envs: int,
    seed: int,
    factory: Callable[..., object],
) -> float:
    """Measure a complete Python-authored policy plus native simulation."""

    environment = VecEnv(num_envs, seed=seed, auto_reset=False)
    policy = factory()
    batch = environment.reset()
    started = time.perf_counter()
    for _ in range(ACTING_TRANSITIONS):
        actions = policy.act(batch, max_orders=environment.max_orders)  # type: ignore[attr-defined]
        batch = environment.step(
            actions.unit_actions,
            actions.market_actions,
            actions.market_lengths,
        )
    elapsed = time.perf_counter() - started
    return num_envs / elapsed


def benchmark_python_boundary(num_envs: int, seed: int) -> float:
    """Isolate native feature extraction plus one Python batch callback."""

    def no_op_strategy(features, targets) -> None:
        del features, targets

    environment = VecEnv(num_envs, seed=seed, auto_reset=False)
    config = RuleConfig()
    planner = PythonRulePlanner(no_op_strategy, config)
    features = None
    batch = environment.reset()
    started = time.perf_counter()
    for _ in range(ACTING_TRANSITIONS):
        features = extract_rule_features(batch, config, features)
        planner.from_features(batch, features)
        batch = environment.step()
    elapsed = time.perf_counter() - started
    return num_envs / elapsed


def measure(
    name: str,
    runner: Callable[[int, int], float],
    *,
    num_envs: int,
    repeats: int,
    seed: int,
) -> float:
    values = [runner(num_envs, seed + repeat) for repeat in range(repeats)]
    median = statistics.median(values)
    spread = min(values), max(values)
    print(
        f"{name:<34} {median:9.1f} games/sec "
        f"(range {spread[0]:.1f}..{spread[1]:.1f}, "
        f"batch={num_envs}, repeats={repeats})"
    )
    return median


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=positive_int, default=256)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--include-current",
        action="store_true",
        help="also benchmark the larger current competitive strategy",
    )
    args = parser.parse_args()

    native = measure(
        "Native VecEnv, pass/pass",
        benchmark_pass,
        num_envs=args.num_envs,
        repeats=args.repeats,
        seed=args.seed,
    )
    boundary = measure(
        "Python callback + native pass",
        benchmark_python_boundary,
        num_envs=args.num_envs,
        repeats=args.repeats,
        seed=args.seed,
    )
    print(
        f"Python callback throughput vs native pass baseline: {boundary / native:.1%}"
    )
    custom = measure(
        "Python simple rules + native runtime",
        lambda count, seed: benchmark_policy(count, seed, build_simple_policy),
        num_envs=args.num_envs,
        repeats=args.repeats,
        seed=args.seed,
    )
    print(f"Full simple-rule throughput vs native pass baseline: {custom / native:.1%}")

    if args.include_current:
        measure(
            "Python current rules + native runtime",
            lambda count, seed: benchmark_policy(count, seed, build_current_policy),
            num_envs=args.num_envs,
            repeats=args.repeats,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
