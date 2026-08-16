#!/usr/bin/env python3
"""Run two Kaggriculture agents on common seeds with both seat orders."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import statistics
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from kaggle_environments import make


DEFAULT_SEED = 451_781_128
DEFAULT_EPISODE_STEPS = 720
Agent = Callable[..., dict[str, object]]


@dataclass(frozen=True)
class GameResult:
    """One game, with rewards and margin normalized to agent A."""

    seed: int
    a_seat: int
    a_reward: float
    b_reward: float
    a_margin: float
    statuses: tuple[str, str]


@dataclass(frozen=True)
class PairResult:
    """The two seat orders played for one common seed."""

    seed: int
    a_as_seat_0: GameResult
    a_as_seat_1: GameResult
    paired_margin: float


def load_agent(path: Path, tag: str) -> Agent:
    """Load one isolated copy so module globals cannot leak between games."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"agent does not exist: {resolved}")
    module_name = f"bertani_match_{tag}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load agent module: {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise TypeError(f"{resolved} does not define a callable agent")
    return agent


def run_game(
    agent_a_path: Path,
    agent_b_path: Path,
    *,
    seed: int,
    a_seat: int,
    episode_steps: int,
    debug: bool,
    show_agent_output: bool,
) -> GameResult:
    """Run one game and express its reward margin from agent A's view."""
    if a_seat not in (0, 1):
        raise ValueError("a_seat must be 0 or 1")

    agent_a = load_agent(agent_a_path, f"a_{seed}_{a_seat}")
    agent_b = load_agent(agent_b_path, f"b_{seed}_{a_seat}")
    agents = [agent_a, agent_b] if a_seat == 0 else [agent_b, agent_a]
    environment = make(
        "kaggriculture",
        configuration={"episodeSteps": episode_steps, "seed": seed},
        debug=debug,
    )

    captured = io.StringIO()
    output_context = (
        contextlib.nullcontext()
        if show_agent_output
        else contextlib.redirect_stdout(captured)
    )
    error_context = (
        contextlib.nullcontext()
        if show_agent_output
        else contextlib.redirect_stderr(captured)
    )
    with output_context, error_context:
        environment.run(agents)

    final = environment.steps[-1]
    statuses = tuple(str(state.status) for state in final)
    if statuses != ("DONE", "DONE"):
        tail = captured.getvalue()[-2_000:]
        detail = f"\nCaptured output:\n{tail}" if tail else ""
        raise RuntimeError(
            f"seed {seed}, A seat {a_seat} ended {statuses}{detail}"
        )
    rewards = tuple(float(state.reward or 0.0) for state in final)
    a_reward = rewards[a_seat]
    b_reward = rewards[1 - a_seat]
    return GameResult(
        seed=seed,
        a_seat=a_seat,
        a_reward=a_reward,
        b_reward=b_reward,
        a_margin=a_reward - b_reward,
        statuses=(statuses[0], statuses[1]),
    )


def run_pairs(
    agent_a_path: Path,
    agent_b_path: Path,
    seeds: list[int],
    *,
    episode_steps: int,
    debug: bool,
    show_agent_output: bool,
) -> list[PairResult]:
    """Play both seat orders for every seed."""
    pairs = []
    for seed in seeds:
        first = run_game(
            agent_a_path,
            agent_b_path,
            seed=seed,
            a_seat=0,
            episode_steps=episode_steps,
            debug=debug,
            show_agent_output=show_agent_output,
        )
        second = run_game(
            agent_a_path,
            agent_b_path,
            seed=seed,
            a_seat=1,
            episode_steps=episode_steps,
            debug=debug,
            show_agent_output=show_agent_output,
        )
        pair = PairResult(
            seed=seed,
            a_as_seat_0=first,
            a_as_seat_1=second,
            paired_margin=first.a_margin + second.a_margin,
        )
        pairs.append(pair)
        print(
            f"seed={seed}  A@0={first.a_margin:+.0f}  "
            f"A@1={second.a_margin:+.0f}  paired={pair.paired_margin:+.0f}"
        )
    return pairs


def summarize(pairs: list[PairResult]) -> dict[str, Any]:
    """Return game-level and paired-seat comparison statistics."""
    games = [game for pair in pairs for game in (pair.a_as_seat_0, pair.a_as_seat_1)]
    game_margins = [game.a_margin for game in games]
    paired_margins = [pair.paired_margin for pair in pairs]
    return {
        "seeds": len(pairs),
        "games": len(games),
        "wins": sum(margin > 0 for margin in game_margins),
        "ties": sum(margin == 0 for margin in game_margins),
        "losses": sum(margin < 0 for margin in game_margins),
        "mean_game_margin": statistics.fmean(game_margins),
        "worst_game_margin": min(game_margins),
        "paired_positive": sum(margin > 0 for margin in paired_margins),
        "paired_zero": sum(margin == 0 for margin in paired_margins),
        "paired_negative": sum(margin < 0 for margin in paired_margins),
        "mean_paired_margin": statistics.fmean(paired_margins),
    }


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent_a", type=Path, help="path to agent A's main.py")
    parser.add_argument("agent_b", type=Path, help="path to agent B's main.py")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[DEFAULT_SEED],
        help=f"common seeds to play (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--episode-steps",
        type=positive_int,
        default=DEFAULT_EPISODE_STEPS,
        help=f"episode length (default: {DEFAULT_EPISODE_STEPS})",
    )
    parser.add_argument("--debug", action="store_true", help="enable environment debug")
    parser.add_argument(
        "--show-agent-output",
        action="store_true",
        help="do not suppress agent stdout and stderr",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="optionally write all game results and the summary as JSON",
    )
    args = parser.parse_args()

    pairs = run_pairs(
        args.agent_a,
        args.agent_b,
        args.seeds,
        episode_steps=args.episode_steps,
        debug=args.debug,
        show_agent_output=args.show_agent_output,
    )
    summary = summarize(pairs)
    print(
        "summary: "
        f"{summary['wins']}W/{summary['ties']}T/{summary['losses']}L, "
        f"mean game margin={summary['mean_game_margin']:+.1f}, "
        "paired="
        f"{summary['paired_positive']}+/{summary['paired_zero']}=/{summary['paired_negative']}-, "
        f"mean paired margin={summary['mean_paired_margin']:+.1f}"
    )

    if args.json_output is not None:
        payload = {
            "agent_a": str(args.agent_a.resolve()),
            "agent_b": str(args.agent_b.resolve()),
            "episode_steps": args.episode_steps,
            "pairs": [asdict(pair) for pair in pairs],
            "summary": summary,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

