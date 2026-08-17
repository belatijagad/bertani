"""Exact parity check for the persistent native scheduler controller.

The production scheduler is the Rust-backed ``TaskScheduler``. This diagnostic
keeps the previous Python controller only under ``scripts/`` as an oracle and
compares assignments, scores, and cache/replan diagnostics turn by turn.

Run after rebuilding the extension:
    uv run python scripts/check_scheduler_controller_parity.py --num-seeds 8
"""
from __future__ import annotations

import argparse

import numpy as np

from bertani.tasks import TaskBatch, TaskScheduler
from bertani.vec_env import VecEnv
from bertani_rules.agent import build_policy
from scheduler_python_oracle import PythonTaskSchedulerOracle


def assert_equal(turn: int, name: str, left: object, right: object) -> None:
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if np.array_equal(left, right):
            return
        mismatch = np.argwhere(left != right)
        index = tuple(int(value) for value in mismatch[0])
        raise AssertionError(
            f"turn {turn} {name} mismatch at {index}: "
            f"python={left[index]!r} rust={right[index]!r}"
        )
    if left != right:
        raise AssertionError(
            f"turn {turn} {name} mismatch: python={left!r} rust={right!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--seed-source", type=int, default=2026)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed_source)
    seeds = rng.integers(
        0,
        np.iinfo(np.uint64).max,
        size=args.num_seeds,
        dtype=np.uint64,
    )
    env = VecEnv(args.num_seeds, seed=args.seed_source, auto_reset=False)
    batch = env.reset(seeds)
    policy = build_policy()
    assert policy.intent_planner is not None

    python_scheduler = PythonTaskSchedulerOracle(
        env.board_size,
        shed_capacity=100,
        episode_steps=720,
        turns_per_day=24,
    )
    native_scheduler = TaskScheduler(
        env.board_size,
        shed_capacity=100,
        episode_steps=720,
        turns_per_day=24,
    )
    tasks = TaskBatch.allocate(args.num_seeds, 2, env.board_size)

    # Match the pit's controlled-seat usage: each environment schedules exactly
    # one seat. Alternating seats exercises both player indices and the behavior
    # where uncontrolled previous-task affinity is reset each call.
    seat_mask = np.zeros((args.num_seeds, 2), dtype=np.bool_)
    seat_mask[np.arange(args.num_seeds), np.arange(args.num_seeds) & 1] = True

    for turn in range(719):
        features = policy.extract_features(batch)
        planner_from_features = getattr(policy.intent_planner, "from_features")
        intent = planner_from_features(batch, features)

        tasks.clear()
        for rule in policy.task_rules:
            rule.propose(batch, intent, tasks)
        workforce = (
            policy.workforce_planner(batch, intent, tasks)
            if policy.workforce_planner is not None
            else None
        )

        python_assignments = python_scheduler.assign(
            batch,
            tasks,
            workforce,
            seat_mask=seat_mask,
        )
        native_assignments = native_scheduler.assign(
            batch,
            tasks,
            workforce,
            seat_mask=seat_mask,
        )

        assert_equal(
            turn,
            "task_index",
            python_assignments.task_index,
            native_assignments.task_index,
        )
        assert_equal(
            turn,
            "score",
            python_assignments.score,
            native_assignments.score,
        )
        assert_equal(turn, "full_solves", python_scheduler.full_solves, native_scheduler.full_solves)
        assert_equal(turn, "cache_hits", python_scheduler.cache_hits, native_scheduler.cache_hits)
        assert_equal(
            turn,
            "cache_miss_reasons",
            python_scheduler.cache_miss_reasons,
            native_scheduler.cache_miss_reasons,
        )
        assert_equal(
            turn,
            "force_replan_reasons",
            python_scheduler.force_replan_reasons,
            native_scheduler.force_replan_reasons,
        )

        actions = policy.act(batch, max_orders=env.max_orders)
        batch = env.step(
            actions.unit_actions,
            actions.market_actions,
            actions.market_lengths,
        )

    print(
        "scheduler-controller parity passed: "
        f"{args.num_seeds} environments x 719 turns"
    )


if __name__ == "__main__":
    main()
