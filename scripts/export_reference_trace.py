#!/usr/bin/env python3
"""Export deterministic Kaggriculture reference states for Rust parity tests.

The action stored on Kaggle state ``t + 1`` is the action that produced that
state from state ``t``.  The JSONL records retain that convention so a replay
consumer cannot accidentally compare against an agent's next action.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture


EXPECTED_PACKAGE_VERSION = "1.32.7"
EXPECTED_SOURCE_SHA256 = (
    "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e"
)
SOURCE_RECORD = "kaggle_environments/envs/kaggriculture/kaggriculture.py"

PRODUCTS = tuple(kaggriculture.PRODUCTS)
CROPS = tuple(kaggriculture.CROPS)
ANIMALS = tuple(kaggriculture.ANIMALS)
ITEMS = PRODUCTS + ANIMALS


def _plain(value: Any) -> Any:
    """Copy Struct/dict/list values without changing list or mapping order."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported replay value {value!r} ({type(value).__name__})")


def _normalize_tile(tile: Any) -> Any:
    if tile is None or tile == "LOCKED":
        return tile
    if not isinstance(tile, Mapping):
        raise TypeError(f"unexpected tile value: {tile!r}")
    # Keeping the mapping instead of reducing it to a tag is intentional: a
    # fixture records every reference field and makes schema drift conspicuous.
    return _plain(tile)


def _normalize_farm(farm: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "money": float(farm["money"]),
        "tiles": [
            [_normalize_tile(tile) for tile in row]
            for row in farm["tiles"]
        ],
        "farmer": [int(value) for value in farm["farmer"]],
        "hands": [[int(value) for value in hand] for hand in farm["hands"]],
        "unlocked_quadrants": list(farm["unlocked_quadrants"]),
        "hires_today": int(farm["hires_today"]),
    }


def _normalize_private(private: Mapping[str, Any]) -> dict[str, Any]:
    # Inventories are arrays of pairs, rather than objects, because Python dict
    # insertion order affects DROP and the end-of-day shed overflow behavior.
    inventories = [
        [[str(item), int(count)] for item, count in inventory.items()]
        for inventory in private["inventories"]
    ]
    return {
        "shed": [int(private["shed"].get(item, 0)) for item in ITEMS],
        "seeds": [int(private["seeds"].get(crop, 0)) for crop in CROPS],
        "inventories": inventories,
    }


def _normalize_state(step: int, agents: list[Any]) -> dict[str, Any]:
    public = agents[0].observation
    observed_step = int(public.step)
    if observed_step != step:
        raise AssertionError(
            f"framework state index {step} has observation.step={observed_step}"
        )

    return {
        "step": observed_step,
        "day": int(public.day),
        "hour": int(public.hour),
        "agents": [
            {
                "player": int(agent.observation.player),
                "status": str(agent.status),
                "reward": float(agent.reward),
            }
            for agent in agents
        ],
        "farms": [_normalize_farm(farm) for farm in public.farms],
        "privates": [
            _normalize_private(agent.observation.private) for agent in agents
        ],
        "market": {
            "inventory": [
                int(public.market["inventory"][product]) for product in PRODUCTS
            ],
            "prices": [
                float(public.market["prices"][product]) for product in PRODUCTS
            ],
        },
        "town": {"unlocked_shops": list(public.town["unlocked_shops"])},
    }


def _normalize_record(step: int, agents: list[Any]) -> dict[str, Any]:
    return {
        "state": _normalize_state(step, agents),
        "actions": [_plain(agent.action) for agent in agents],
    }


def _verify_reference_install() -> tuple[str, Path, str]:
    version = importlib.metadata.version("kaggle-environments")
    source = Path(inspect.getsourcefile(kaggriculture) or "").resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if version != EXPECTED_PACKAGE_VERSION:
        raise SystemExit(
            "refusing to export from kaggle-environments "
            f"{version}; expected {EXPECTED_PACKAGE_VERSION}"
        )
    if digest != EXPECTED_SOURCE_SHA256:
        raise SystemExit(
            f"refusing to export modified Kaggriculture source {source}: "
            f"sha256={digest}, expected {EXPECTED_SOURCE_SHA256}"
        )
    return version, source, digest


def export_trace(output: Path, seed: int, players: tuple[str, str]) -> None:
    version, source, digest = _verify_reference_install()
    requested_configuration = {"seed": seed, "episodeSteps": 720}
    environment = make(
        "kaggriculture", configuration=requested_configuration, debug=True
    )
    environment.run(list(players))

    episode_steps = int(environment.configuration.episodeSteps)
    if len(environment.steps) != episode_steps:
        raise AssertionError(
            f"expected {episode_steps} recorded states, got {len(environment.steps)}"
        )

    header = {
        "format": "kaggriculture-reference-trace-v1",
        "kaggle_environments_version": version,
        "source": SOURCE_RECORD,
        "source_sha256": digest,
        "seed": seed,
        "players": list(players),
        "state_count": len(environment.steps),
        "transition_count": len(environment.steps) - 1,
        "item_order": list(ITEMS),
        "crop_order": list(CROPS),
        "product_order": list(PRODUCTS),
        "configuration": {
            "episodeSteps": episode_steps,
            "boardSize": int(environment.configuration.boardSize),
            "startingMoney": int(environment.configuration.startingMoney),
            "maxMarketOrdersPerTurn": int(
                environment.configuration.maxMarketOrdersPerTurn
            ),
            "turnsPerDay": int(environment.configuration.turnsPerDay),
            "shedCapacity": int(environment.configuration.shedCapacity),
            "weedSpawnChance": float(environment.configuration.weedSpawnChance),
            "townShopUnlockInterval": int(
                environment.configuration.townShopUnlockInterval
            ),
            "townShopSellInterval": int(
                environment.configuration.townShopSellInterval
            ),
            "townCenterSellInterval": int(
                environment.configuration.townCenterSellInterval
            ),
            "farmHandCostMult": int(environment.configuration.farmHandCostMult),
            "seed": seed,
        },
        "action_alignment": (
            "record[t].actions is env.steps[t][*].action; for t>0 it produced "
            "record[t] from record[t-1]"
        ),
    }

    rows = [
        json.dumps(header, separators=(",", ":"), sort_keys=True),
        *(
            json.dumps(
                _normalize_record(step, agents), separators=(",", ":"), sort_keys=True
            )
            for step, agents in enumerate(environment.steps)
        ),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = ("\n".join(rows) + "\n").encode()
    if output.suffix == ".gz":
        # mtime=0 makes regeneration byte-for-byte deterministic.
        with output.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                zipped.write(payload)
    else:
        output.write_bytes(payload)
    print(
        f"wrote {len(environment.steps)} states / {len(environment.steps) - 1} "
        f"transitions to {output} from {source}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "crates/kaggriculture-core/tests/fixtures/starter_vs_pass_seed11.jsonl.gz"
        ),
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--agents",
        help="comma-separated player agents (equivalent to --player-0/--player-1)",
    )
    parser.add_argument("--player-0", default="starter")
    parser.add_argument("--player-1", default="pass")
    args = parser.parse_args()
    players = (args.player_0, args.player_1)
    if args.agents is not None:
        parsed_players = tuple(part.strip() for part in args.agents.split(","))
        if len(parsed_players) != 2 or not all(parsed_players):
            parser.error("--agents must contain exactly two non-empty comma-separated names")
        players = parsed_players
    export_trace(args.output, args.seed, players)


if __name__ == "__main__":
    main()
