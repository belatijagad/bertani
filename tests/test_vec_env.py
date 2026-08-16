from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bertani._rust", reason="the maturin extension has not been built")

from bertani import Item, MarketOp, VecEnv


def _quiet_env(**kwargs: object) -> VecEnv:
    options: dict[str, object] = {
        "weed_spawn_chance": 0.0,
        "turns_per_day": 2,
        "max_market_orders": 2,
    }
    options.update(kwargs)
    return VecEnv(2, **options)


def test_batches_and_named_views_reuse_the_same_memory() -> None:
    env = _quiet_env(seed=123)
    initial = env.reset()
    pointers = {
        "observations": initial.observations.ctypes.data,
        "masks": initial.action_masks.ctypes.data,
        "rewards": initial.rewards.ctypes.data,
        "dones": initial.dones.ctypes.data,
    }

    stepped = env.step()

    assert stepped is initial
    assert stepped.observations.ctypes.data == pointers["observations"]
    assert stepped.action_masks.ctypes.data == pointers["masks"]
    assert stepped.rewards.ctypes.data == pointers["rewards"]
    assert stepped.dones.ctypes.data == pointers["dones"]
    assert np.shares_memory(
        stepped.observations, stepped.observation_views.global_features
    )
    assert np.shares_memory(stepped.observations, stepped.observation_views.tiles)
    assert np.shares_memory(stepped.action_masks, stepped.mask_views.unit_args)
    assert np.shares_memory(stepped.action_masks, stepped.mask_views.market_args)
    assert stepped.observation_views.tiles.shape == (2, 2, 2, 10, 10, 24)
    assert stepped.observation_views.units.shape == (
        2,
        2,
        2,
        env.max_units,
        29,
    )
    assert stepped.mask_views.unit_ops.shape == (2, 2, env.max_units, 18)
    assert stepped.mask_views.market_args.shape == (2, 2, 7, 12)


def test_identical_base_and_explicit_seeds_are_deterministic() -> None:
    first = _quiet_env(seed=431)
    second = _quiet_env(seed=431)

    first_initial = first.reset().observations.copy()
    second_initial = second.reset().observations.copy()
    np.testing.assert_array_equal(first_initial, second_initial)
    assert first.state_snapshot(0) == second.state_snapshot(0)
    assert first.state_snapshot(1) == second.state_snapshot(1)

    seeds = np.array([91, 92], dtype=np.uint64)
    first_explicit = first.reset(seeds).observations.copy()
    second_explicit = second.reset(seeds).observations.copy()
    np.testing.assert_array_equal(first_explicit, second_explicit)
    assert first.state_snapshot(0) == second.state_snapshot(0)
    assert first.state_snapshot(1) == second.state_snapshot(1)


def test_actions_are_validated_without_dtype_or_layout_coercion() -> None:
    env = _quiet_env()
    env.reset()

    shape = (2, 2, env.max_units, 3)
    wrong_dtype = np.zeros(shape, dtype=np.float32)
    with pytest.raises(TypeError, match="unit_actions must have dtype int64"):
        env.step(unit_actions=wrong_dtype)

    wrong_shape = np.zeros((2, 2, env.max_units - 1, 3), dtype=np.int64)
    with pytest.raises(ValueError, match="unit_actions must have shape"):
        env.step(unit_actions=wrong_shape)

    noncontiguous = np.asfortranarray(np.zeros(shape, dtype=np.int64))
    assert not noncontiguous.flags.c_contiguous
    with pytest.raises(ValueError, match="unit_actions must be C-contiguous"):
        env.step(unit_actions=noncontiguous)

    wrong_lengths = np.zeros((2, 2), dtype=np.int32)
    with pytest.raises(TypeError, match="market_lengths must have dtype int64"):
        env.step(market_lengths=wrong_lengths)

    with pytest.raises(TypeError, match="seeds must have dtype uint64"):
        env.reset(np.array([1, 2], dtype=np.int64))

    with pytest.raises(TypeError, match="unit_actions must be a NumPy array"):
        env.step(unit_actions=[0])


def test_terminal_reward_snapshot_and_auto_reset_are_unambiguous() -> None:
    env = VecEnv(
        1,
        seed=7,
        auto_reset=True,
        episode_steps=2,
        max_market_orders=1,
        turns_per_day=2,
        weed_spawn_chance=0.0,
    )
    env.reset()
    market = np.zeros((1, 2, env.max_orders, 3), dtype=np.int64)
    market[0, 0, 0] = (MarketOp.BUY_SEED, Item.CARROT, 1)
    lengths = np.array([[1, 0]], dtype=np.int64)

    terminal_step = env.step(market_actions=market, market_lengths=lengths)

    np.testing.assert_array_equal(terminal_step.dones, [[True, True]])
    np.testing.assert_array_equal(terminal_step.rewards, [[2_980.0, 3_000.0]])

    current = env.state_snapshot(0)
    terminal = env.terminal_snapshot(0)
    assert terminal is not None
    assert current["step"] == 0
    assert current["done"] is False
    assert current["farms"][0]["money"] == 3_000
    assert terminal["step"] == 1
    assert terminal["done"] is True
    assert terminal["farms"][0]["money"] == 2_980
    assert terminal["farms"][0]["seeds"][Item.CARROT] == 1
    assert current["episode_id"] == terminal["episode_id"] + 1
