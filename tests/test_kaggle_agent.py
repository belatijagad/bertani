from __future__ import annotations

import numpy as np
from kaggle_environments import make

from bertani_rules.agent import build_policy
from bertani.kaggle_agent import SHOP_NAMES, make_agent, observation_batch
from bertani.rule_based import RuleConfig


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


def test_submission_encoder_matches_derived_town_feature_contract() -> None:
    farm = {
        "money": 3_000,
        "farmer": [0, 0],
        "hands": [],
        "tiles": [[None]],
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
    }
    observation = {
        "player": 0,
        "step": 71,
        "day": 2,
        "farms": [farm, farm],
        "private": {},
        "market": {},
        "town": {"unlocked_shops": list(SHOP_NAMES)},
    }

    batch = observation_batch(observation, RuleConfig())
    features = batch.observation_views.global_features[0, 0]

    np.testing.assert_allclose(
        features[30:39], np.asarray([5, 3, 2, 4, 0, 2, 3, 2, 0]) / 16
    )
    np.testing.assert_allclose(features[39:42], [0.25, 1 / 24, 0.0])
