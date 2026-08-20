from __future__ import annotations

import gc
from pathlib import Path
from types import ModuleType

import numpy as np

from bertani import MarketOp, VecEnv
from bertani.native_agent import load_agent_module
from bertani.v9_opponent import V9OpponentPolicy, V9SelfPlayEnv


def _fake_v9(steps: int = 8) -> ModuleType:
    module = ModuleType("fake_v9")
    module._R9_MODE = 5
    module._R9_LAST_TOWN = None
    module._R9_LAST_STEP = -1
    module._V8_COMMIT = None
    module._V8_LAST_STEP = -1
    module._R9_BANK = [{1: [(None, None, None)]} for _ in range(steps)]
    module.calls = 0

    def feature(observation: dict[str, object]) -> list[object]:
        del observation
        # V9's eight-part feature shape. Symmetric reset farms share this key.
        return [3_000, [0], [[0]], [0], [0], ["_"] * 100, [10], 1]

    def agent(
        observation: dict[str, object], configuration: object = None
    ) -> dict[str, object]:
        del configuration
        module.calls += 1
        step = int(observation["step"])
        town = tuple(observation["town"]["unlocked_shops"])
        if town != module._R9_LAST_TOWN:
            module._R9_MODE = 5
            module._R9_LAST_TOWN = town
        module._R9_MODE += 1
        module._R9_LAST_STEP = step
        return {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["HIRE"]],
        }

    module._feature = feature
    module.agent = agent
    return module


def _policy(module: ModuleType, environment: VecEnv) -> V9OpponentPolicy:
    return V9OpponentPolicy(
        module,
        configuration={"episodeSteps": 720},
        max_orders=environment.max_orders,
    )


def test_v9_policy_shares_bank_and_caches_symmetric_openings() -> None:
    environment = VecEnv(4, auto_reset=False, weed_spawn_chance=0.0)
    seeds = np.asarray([31, 31, 31, 31], dtype=np.uint64)
    batch = environment.reset(seeds)
    module = _fake_v9()
    policy = _policy(module, environment)
    seats = np.asarray([1, 0, 1, 0], dtype=np.int64)

    actions = policy.act(environment, batch, seats=seats)

    assert module.calls == 1
    assert policy.cache_stats.hits == 3
    assert policy.cache_stats.misses == 1
    np.testing.assert_array_equal(
        actions.market_lengths[np.arange(4), seats], np.ones(4)
    )
    np.testing.assert_array_equal(
        actions.market_actions[np.arange(4), seats, 0, 0],
        np.full(4, MarketOp.HIRE),
    )
    assert not actions.market_lengths[np.arange(4), 1 - seats].any()


def test_self_play_wrapper_composes_alternating_seats() -> None:
    environment = VecEnv(
        4,
        auto_reset=False,
        episode_steps=4,
        turns_per_day=2,
        weed_spawn_chance=0.0,
    )
    wrapper = V9SelfPlayEnv(environment, _policy(_fake_v9(4), environment))
    wrapper.reset(np.asarray([4, 5, 6, 7], dtype=np.uint64))
    learner_actions = np.zeros(
        (environment.num_envs, environment.max_units, 3), dtype=np.int64
    )

    batch = wrapper.step(learner_actions)

    np.testing.assert_array_equal(wrapper.learner_seats, [0, 1, 0, 1])
    np.testing.assert_array_equal(wrapper.opponent_seats, [1, 0, 1, 0])
    assert batch.active_units[
        wrapper.games, wrapper.opponent_seats, 1
    ].all()
    assert not batch.active_units[
        wrapper.games, wrapper.learner_seats, 1
    ].any()
    assert (wrapper.learner_rewards() == 0.0).all()
    assert not wrapper.learner_dones().any()


def test_agent_loader_restores_process_gc_setting(tmp_path: Path) -> None:
    agent_path = tmp_path / "gc_agent.py"
    agent_path.write_text(
        "import gc\ngc.disable()\ndef agent(observation):\n    return {}\n"
    )
    gc.enable()

    load_agent_module(agent_path)

    assert gc.isenabled()


def test_self_play_rejects_market_lengths_without_orders() -> None:
    environment = VecEnv(1, auto_reset=False, episode_steps=3)
    wrapper = V9SelfPlayEnv(environment, _policy(_fake_v9(3), environment))
    wrapper.reset(np.asarray([8], dtype=np.uint64))
    learner_actions = np.zeros(
        (environment.num_envs, environment.max_units, 3), dtype=np.int64
    )

    try:
        wrapper.step(
            learner_actions,
            learner_market_lengths=np.zeros(1, dtype=np.int64),
        )
    except ValueError as error:
        assert "requires learner_market_actions" in str(error)
    else:
        raise AssertionError("missing learner market actions should be rejected")
