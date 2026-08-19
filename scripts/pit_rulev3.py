#!/usr/bin/env python3
"""Pit the current rule policy against the preserved RuleV3 agent."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pit_agents import (
    DEFAULT_EPISODE_STEPS,
    DEFAULT_SEED,
    positive_int,
    run_pairs,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURRENT = ROOT / "scripts" / "current_rule_agent.py"
DEFAULT_RULEV3 = ROOT / "references" / "rule" / "rulev3.py"
DEFAULT_OUTPUT = ROOT / "outputs" / "current-rule-rulev3.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--rulev3", type=Path, default=DEFAULT_RULEV3)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[DEFAULT_SEED]
    )
    parser.add_argument(
        "--episode-steps", type=positive_int, default=DEFAULT_EPISODE_STEPS
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show-agent-output", action="store_true")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    pairs = run_pairs(
        args.current,
        args.rulev3,
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
        f"{summary['paired_positive']}+/{summary['paired_zero']}=/"
        f"{summary['paired_negative']}-, "
        f"mean paired margin={summary['mean_paired_margin']:+.1f}"
    )
    output = args.json_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "agent_a": str(args.current.resolve()),
                "agent_b": str(args.rulev3.resolve()),
                "episode_steps": args.episode_steps,
                "pairs": [asdict(pair) for pair in pairs],
                "summary": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
