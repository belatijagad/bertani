#!/usr/bin/env python3
"""Benchmark full kaggle-environments pass/pass episodes.

This intentionally times framework construction and agent orchestration as well
as Kaggriculture's Python interpreter. It is not a core-to-core comparison with
the Rust typed-engine benchmark.
"""

from __future__ import annotations

import argparse
import gc
import time
from collections.abc import Mapping
from typing import Any

from kaggle_environments import make


DEFAULT_EPISODES = 3
EPISODE_STEPS = 720


def pass_agent(
    _observation: Mapping[str, Any], _configuration: Mapping[str, Any]
) -> dict[str, object]:
    """Return one typed-equivalent pass action for the framework to validate."""
    return {"farmer": ["PASS"], "hands": [], "market": []}


def run_episode(seed: int):
    """Construct and run one complete framework episode."""
    environment = make(
        "kaggriculture",
        configuration={"episodeSteps": EPISODE_STEPS, "seed": seed},
        debug=False,
    )
    environment.run([pass_agent, pass_agent])
    return environment


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "episodes",
        nargs="?",
        type=positive_int,
        default=DEFAULT_EPISODES,
        help=f"number of timed episodes (default: {DEFAULT_EPISODES})",
    )
    args = parser.parse_args()

    # Imports, one framework warm-up, validation, and an explicit collection are
    # all outside the timed region.
    warmup = run_episode(args.episodes)
    if len(warmup.steps) != EPISODE_STEPS:
        raise RuntimeError(
            f"expected {EPISODE_STEPS} recorded states, got {len(warmup.steps)}"
        )
    del warmup
    gc.collect()

    started = time.perf_counter()
    checksum = 0.0
    for seed in range(args.episodes):
        environment = run_episode(seed)
        checksum += sum(float(agent.reward or 0.0) for agent in environment.steps[-1])
    elapsed = time.perf_counter() - started
    # Keep the final observations live until after the timer and make it obvious
    # that completed runs, rather than environment construction alone, were used.
    if checksum < 0.0:
        raise RuntimeError("unreachable negative pass/pass reward checksum")

    episodes_per_second = args.episodes / elapsed
    milliseconds_per_episode = elapsed * 1_000.0 / args.episodes
    print(
        "Python full kaggle-environments framework "
        f"(make + agents + 719 pass/pass turns): {args.episodes} episodes in "
        f"{elapsed:.3f}s = {episodes_per_second:.2f} episodes/sec, "
        f"{milliseconds_per_episode:.1f} ms/episode"
    )
    print(
        "Scope: includes framework/schema/agent orchestration; this is not "
        "apples-to-apples with the Rust typed-core benchmark."
    )


if __name__ == "__main__":
    main()
