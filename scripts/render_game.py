#!/usr/bin/env python3
"""Play the current rule agent locally and open its HTML replay."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess

from kaggle_environments import make

from bertani.kaggle_agent import make_agent
from bertani_rules.agent import build_policy

try:
    from scripts.pit_agents import load_agent
except ModuleNotFoundError:  # Direct execution adds scripts/, not its parent.
    from pit_agents import load_agent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPPONENT = ROOT / "baselines" / "v16_rc5" / "main.py"
BUILTIN_AGENTS = frozenset(("pass", "random", "starter"))
WSL_BRAVE_PATHS = (
    Path("/mnt/c/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"),
    Path("/mnt/c/Program Files (x86)/BraveSoftware/Brave-Browser/Application/brave.exe"),
)


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def opponent_agent(value: str):
    """Resolve a Kaggle built-in name or a submission-compatible Python file."""

    if value in BUILTIN_AGENTS:
        return value
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return load_agent(path, "replay_opponent")


def open_replay(path: Path) -> bool:
    """Launch a replay without waiting for the desktop opener to exit."""

    resolved = path.resolve()
    if os.environ.get("WSL_DISTRO_NAME"):
        converted = subprocess.run(
            ("wslpath", "-w", str(resolved)),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        brave = next(
            (candidate for candidate in WSL_BRAVE_PATHS if candidate.is_file()),
            None,
        )
        if brave is not None:
            command = (str(brave), converted)
        elif shutil.which("explorer.exe"):
            command = ("explorer.exe", converted)
        else:
            return False
    elif shutil.which("xdg-open"):
        command = ("xdg-open", str(resolved))
    elif shutil.which("open"):
        command = ("open", str(resolved))
    else:
        return False
    subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--rule-seat",
        type=int,
        choices=(0, 1),
        default=0,
        help="seat occupied by the rule agent (default: 0)",
    )
    parser.add_argument(
        "--opponent",
        default=str(DEFAULT_OPPONENT),
        help="opponent main.py or one of: pass, random, starter",
    )
    parser.add_argument(
        "--episode-steps",
        type=positive_int,
        default=720,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="HTML destination (default: outputs/replays/local-...html)",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="generate the replay without opening a browser",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        opponent_path = Path(args.opponent)
        opponent_name = (
            args.opponent
            if args.opponent in BUILTIN_AGENTS
            else (
                opponent_path.parent.name
                if opponent_path.name == "main.py"
                else opponent_path.stem
            )
        )
        output = ROOT / "outputs" / "replays" / (
            f"rule-vs-{opponent_name}-seed-{args.seed}"
            f"-seat-{args.rule_seat}.html"
        )
    elif not output.is_absolute():
        output = ROOT / output
    if output.suffix.lower() != ".html":
        parser.error("--output must end in .html")

    rule = make_agent(build_policy)
    opponent = opponent_agent(args.opponent)
    agents = [rule, opponent] if args.rule_seat == 0 else [opponent, rule]
    environment = make(
        "kaggriculture",
        configuration={
            "episodeSteps": args.episode_steps,
            "seed": args.seed,
        },
        debug=args.debug,
    )
    environment.run(agents)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(environment.render(mode="html"), encoding="utf-8")
    final = environment.steps[-1]
    rewards = [float(state.reward or 0.0) for state in final]
    statuses = [str(state.status) for state in final]
    print(f"rewards={rewards} statuses={statuses}")
    print(f"replay={output}")

    if not args.no_open:
        if not open_replay(output):
            print("Browser launch was unavailable; open the replay path manually.")


if __name__ == "__main__":
    main()
