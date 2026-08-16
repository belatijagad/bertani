#!/usr/bin/env python3
"""Benchmark batched pass/pass transitions through ``bertani.VecEnv``.

Observation encoding, action-mask encoding, and terminal auto-reset are all
inside the timed ``step`` calls. Environment construction, reset, and action
buffer setup are deliberately outside the timed region.
"""

from __future__ import annotations

import argparse
import filecmp
import os
import statistics
import sys
import time
from pathlib import Path

import bertani._rust as native
from bertani import VecEnv


DEFAULT_ENVS = 1_024
DEFAULT_TRANSITIONS = 719
DEFAULT_REPEATS = 5
DEFAULT_WARMUP = 1
DEBUG_OVERRIDE = "BERTANI_BENCHMARK_ALLOW_DEBUG"


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def build_profile() -> str | None:
    """Best-effort profile detection without requiring a native build flag."""

    declared = getattr(native, "BUILD_PROFILE", None)
    if isinstance(declared, str) and declared.lower() in {"debug", "release"}:
        return declared.lower()

    extension = Path(native.__file__).resolve()
    lowered_parts = {part.lower() for part in extension.parts}
    if "debug" in lowered_parts:
        return "debug"
    if "release" in lowered_parts:
        return "release"

    # ``maturin develop`` commonly installs the extension into ``src``. In a
    # source checkout, match it to the corresponding Cargo artifact.
    repository = Path(__file__).resolve().parents[1]
    for profile in ("debug", "release"):
        candidate = repository / "target" / profile / "lib_rust.so"
        try:
            if candidate.exists() and (
                extension.samefile(candidate)
                or filecmp.cmp(extension, candidate, shallow=False)
            ):
                return profile
        except OSError:
            pass
    return None


def require_release_build() -> None:
    profile = build_profile()
    if profile == "debug" and os.environ.get(DEBUG_OVERRIDE) != "1":
        raise SystemExit(
            "Refusing to benchmark a debug native extension. Build it with "
            "`uv run maturin develop --release`, or set "
            f"{DEBUG_OVERRIDE}=1 to override."
        )
    if profile is None:
        print(
            "WARNING: native build profile could not be verified. Build with "
            "`uv run maturin develop --release` before trusting these numbers.",
            file=sys.stderr,
        )
    elif profile == "debug":
        print(
            "WARNING: benchmarking a debug native extension by request.",
            file=sys.stderr,
        )


def run_steps(environment: VecEnv, transitions: int) -> None:
    units = environment.unit_actions
    market = environment.market_actions
    lengths = environment.market_lengths
    for _ in range(transitions):
        environment.step(units, market, lengths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--envs",
        type=positive_int,
        default=DEFAULT_ENVS,
        help=f"parallel environments (default: {DEFAULT_ENVS})",
    )
    parser.add_argument(
        "--transitions",
        type=positive_int,
        default=DEFAULT_TRANSITIONS,
        help=f"batched step calls per repeat (default: {DEFAULT_TRANSITIONS})",
    )
    parser.add_argument(
        "--repeats",
        type=positive_int,
        default=DEFAULT_REPEATS,
        help=f"timed repeats (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--warmup",
        type=nonnegative_int,
        default=DEFAULT_WARMUP,
        help=f"untimed repeats (default: {DEFAULT_WARMUP})",
    )
    args = parser.parse_args()

    require_release_build()
    environment = VecEnv(args.envs, auto_reset=True)
    environment.clear_actions()

    for _ in range(args.warmup):
        environment.reset()
        run_steps(environment, args.transitions)

    samples: list[float] = []
    for _ in range(args.repeats):
        environment.reset()
        started = time.perf_counter()
        run_steps(environment, args.transitions)
        samples.append(time.perf_counter() - started)

    elapsed = sum(samples)
    batched_transitions = args.transitions * args.repeats
    env_steps = batched_transitions * args.envs
    transitions_per_second = batched_transitions / elapsed
    env_steps_per_second = env_steps / elapsed
    microseconds_per_env_step = elapsed * 1_000_000.0 / env_steps

    print(
        f"VecEnv pass/pass: {args.envs:,} envs x {batched_transitions:,} "
        f"timed transitions in {elapsed:.3f}s "
        f"({statistics.median(samples):.3f}s median/repeat)"
    )
    print(f"batched transitions/sec: {transitions_per_second:,.2f}")
    print(f"env-steps/sec:            {env_steps_per_second:,.0f}")
    print(f"microseconds/env-step:    {microseconds_per_env_step:.3f}")
    print("Scope: step includes observation + action-mask encoding and auto-reset.")


if __name__ == "__main__":
    main()
