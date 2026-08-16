from __future__ import annotations

from kaggle_environments import make

from bertani_rules.v1 import build_policy
from bertani.kaggle_agent import make_agent


agent = make_agent(build_policy)


def test_kaggle_adapter_emits_the_opening_controller_action() -> None:
    environment = make(
        "kaggriculture",
        configuration={"episodeSteps": 720, "seed": 11},
        debug=False,
    )
    environment.reset(2)

    action = agent(environment.state[0].observation, environment.configuration)

    assert action == {
        "farmer": ["BUILD_PASTURE"],
        "hands": [],
        "market": [
            ["HIRE"],
            ["HIRE"],
            ["HIRE"],
            ["HIRE"],
            ["HIRE"],
            ["BUY_ANIMAL", "COW", 2],
            ["BUY_ANIMAL", "SHEEP", 2],
            ["BUY_SEED", "WHEAT", 7],
            ["BUY_SEED", "MELON", 12],
            ["BUY_PRODUCT", "WHEAT", 6],
        ],
    }
