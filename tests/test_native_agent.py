from __future__ import annotations

import json

import numpy as np
from kaggle_environments import make

from bertani import Item, MarketOp, UnitOp, VecEnv
from bertani.native_agent import NativeFileAgentPolicy, snapshot_observation


def test_native_snapshot_converts_to_the_public_initial_observation() -> None:
    seed = 451_781_128
    python_environment = make(
        "kaggriculture", configuration={"seed": seed}, debug=False
    )
    python_environment.reset(2)
    expected = json.loads(
        json.dumps(python_environment.state[1].observation)
    )
    # The local framework omits ``step`` from seat 1 at reset even though it is
    # part of the documented observation contract. Supplying the equivalent
    # value keeps file agents seat-independent.
    expected.setdefault("step", 0)

    native_environment = VecEnv(1, seed=seed, auto_reset=False)
    native_environment.reset(np.asarray([seed], dtype=np.uint64))
    actual = snapshot_observation(native_environment.state_snapshot(0), 1)

    assert actual == expected


def test_file_agent_actions_encode_for_only_the_selected_seat() -> None:
    seen: list[dict[str, object]] = []

    def agent(
        observation: dict[str, object], configuration: object = None
    ) -> dict[str, object]:
        seen.append(observation)
        return {
            "farmer": ["BUILD_PASTURE"],
            "hands": [],
            "market": [
                ["HIRE"],
                ["BUY_ANIMAL", "COW", 2],
                ["BUY_SEED", "WHEAT", 7],
            ],
        }

    environment = VecEnv(2, auto_reset=False)
    batch = environment.reset(np.asarray([11, 12], dtype=np.uint64))
    policy = NativeFileAgentPolicy(
        agent,
        configuration={"episodeSteps": 720},
        max_orders=environment.max_orders,
    )
    seat_mask = np.asarray([[True, False], [False, True]], dtype=np.bool_)

    actions = policy.act(environment, batch, seat_mask=seat_mask)

    assert [observation["player"] for observation in seen] == [0, 1]
    np.testing.assert_array_equal(
        actions.unit_actions[:, :, 0, 0],
        [[UnitOp.BUILD_PASTURE, UnitOp.PASS], [UnitOp.PASS, UnitOp.BUILD_PASTURE]],
    )
    assert tuple(actions.market_actions[0, 0, 0]) == (MarketOp.HIRE, 0, 0)
    assert tuple(actions.market_actions[0, 0, 1]) == (
        MarketOp.BUY_ANIMAL,
        Item.COW,
        2,
    )
    assert tuple(actions.market_actions[0, 0, 2]) == (
        MarketOp.BUY_SEED,
        Item.WHEAT,
        7,
    )
    np.testing.assert_array_equal(actions.market_lengths, [[3, 0], [0, 3]])
