from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bertani._rust", reason="the maturin extension has not been built")

from bertani_rules.agent import OPENING_BOOK, build_policy

from bertani import (
    Item,
    UnitOp,
    VecEnv,
)


EXPECTED_FIELD = (
    "WWWWM",
    "MWMMW",
    "MMWM.",
    "MMMSS",
    "MMPCC",
)


def field_layout(snapshot: dict[str, object], player: int) -> tuple[str, ...]:
    farms = snapshot["farms"]
    assert isinstance(farms, list)
    farm = farms[player]
    rows: list[str] = []
    for row in farm["tiles"]:
        labels: list[str] = []
        for tile in row[:5]:
            kind = tile["kind"]
            if kind == "EMPTY":
                labels.append(".")
            elif kind == "PLANT":
                labels.append("W" if tile["crop"] == int(Item.WHEAT) else "M")
            elif kind == "PASTURE" and "animal" not in tile:
                labels.append("P")
            elif kind == "PASTURE":
                labels.append("C" if tile["animal"] == 1 else "S")
            else:
                labels.append("?")
        rows.append("".join(labels))
    return tuple(rows[:5])


def test_nominal_opening_reaches_the_observed_day_three_position() -> None:
    assert len(OPENING_BOOK) == 72
    env = VecEnv(4, seed=100, weed_spawn_chance=0.0)
    policy = build_policy()
    batch = env.reset()

    for turn in range(len(OPENING_BOOK)):
        actions = policy.act(batch, max_orders=env.max_orders)
        diagnostics = policy.last_opening_diagnostics
        assert diagnostics is not None
        assert diagnostics.active.all()
        assert not diagnostics.recovering.any()
        assert not diagnostics.invalid_nominal_action.any(), turn
        batch = env.step(
            actions.unit_actions,
            actions.market_actions,
            actions.market_lengths,
        )

    for environment in range(env.num_envs):
        snapshot = env.state_snapshot(environment)
        assert snapshot["step"] == 72
        assert snapshot["day"] == 3
        for player in range(2):
            farm = snapshot["farms"][player]
            assert field_layout(snapshot, player) == EXPECTED_FIELD
            assert farm["shed"][Item.WHEAT] == 4
            assert farm["shed"][Item.FERTILIZER] == 4
            assert farm["hands"] == []

    policy.act(batch, max_orders=env.max_orders)
    diagnostics = policy.last_opening_diagnostics
    assert diagnostics is not None
    assert not diagnostics.active.any()
    assert diagnostics.finished.all()


def test_opening_repairs_replay_seed_weed_and_reconverges() -> None:
    # Seed 874717982 is episode 93177718, where カワシギ found a weed on the
    # planned pasture at (2, 4). The second slot is a nominal counterexample.
    env = VecEnv(2)
    seeds = np.array([874_717_982, 192_124_1818], dtype=np.uint64)
    policy = build_policy()
    batch = env.reset(seeds)
    recovery_actions: list[tuple[int, int]] = []

    for turn in range(len(OPENING_BOOK)):
        actions = policy.act(batch, max_orders=env.max_orders)
        diagnostics = policy.last_opening_diagnostics
        assert diagnostics is not None
        if diagnostics.recovering[0, 0]:
            recovery_actions.append((turn, int(actions.unit_actions[0, 0, 0, 0])))
        batch = env.step(
            actions.unit_actions,
            actions.market_actions,
            actions.market_lengths,
        )

    assert recovery_actions == [
        (66, int(UnitOp.DIG)),
        (67, int(UnitOp.BUILD_PASTURE)),
    ]
    assert env.state_snapshot(0)["farms"][0]["tiles"][4][2] == {
        "kind": "PASTURE"
    }
    assert field_layout(env.state_snapshot(0), 0) == EXPECTED_FIELD
