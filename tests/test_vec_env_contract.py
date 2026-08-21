from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("bertani._rust", reason="the maturin extension has not been built")

from bertani import Item, MarketOp, VecEnv
from bertani._rust import NativeVecEnv


def _market_env(num_envs: int = 1, **kwargs: object) -> VecEnv:
    options: dict[str, object] = {
        "seed": 5,
        "episode_steps": 20,
        "max_market_orders": 3,
        "turns_per_day": 2,
        "weed_spawn_chance": 0.0,
    }
    options.update(kwargs)
    return VecEnv(num_envs, **options)


def test_default_max_units_is_the_exact_reachable_bound() -> None:
    env = VecEnv(1)

    # Hires made on the end-of-day turn are cleared before they can appear in
    # another observation, leaving 23 observable hire rounds plus the farmer.
    assert env.max_units == (24 - 1) * 10 + 1 == 231

    with pytest.raises(ValueError, match=r"max_units=230 is too small.*231 slots"):
        VecEnv(1, max_units=230)

    assert VecEnv(1, turns_per_day=1).max_units == 1


def test_native_decode_errors_are_transactional_for_the_whole_batch() -> None:
    env = _market_env(2)
    env.reset(np.array([101, 202], dtype=np.uint64))
    before = [copy.deepcopy(env.state_snapshot(index)) for index in range(2)]

    units = np.zeros((2, 2, env.max_units, 3), dtype=np.int64)
    market = np.zeros((2, 2, env.max_orders, 3), dtype=np.int64)
    lengths = np.zeros((2, 2), dtype=np.int64)

    # Put a valid, observable order in the first environment. If decoding and
    # stepping were interleaved, it could mutate before the later bad row was
    # discovered.
    market[0, 0, 0] = (MarketOp.BUY_SEED, Item.CARROT, 1)
    lengths[0, 0] = 1
    units[1, 0, 0, 0] = 18  # one past the largest valid unit operation

    with pytest.raises(ValueError, match=r"environment 1: invalid unit op 18"):
        env.step(units, market, lengths)
    assert [env.state_snapshot(index) for index in range(2)] == before

    units[1, 0, 0, 0] = 0
    lengths[1, 1] = env.max_orders + 1
    with pytest.raises(ValueError, match=r"market length 4.*exceeds max_orders \(3\)"):
        env.step(units, market, lengths)
    assert [env.state_snapshot(index) for index in range(2)] == before


def test_asymmetric_market_lengths_keep_an_internal_none_slot() -> None:
    env = _market_env(2)
    # Duplicate seeds make the two slots exact counterfactuals. Slot zero keeps
    # the NONE gap; slot one manually compacts the same two wheat purchases.
    env.reset(np.array([5, 5], dtype=np.uint64))
    market = np.zeros((2, 2, env.max_orders, 3), dtype=np.int64)
    lengths = np.array([[3, 2], [2, 2]], dtype=np.int64)
    buy_wheat = (MarketOp.BUY_PRODUCT, Item.WHEAT, 1)

    market[0, 0, 0] = buy_wheat
    market[0, 0, 2] = buy_wheat
    market[0, 1, 1] = buy_wheat

    market[1, 0, 0] = buy_wheat
    market[1, 0, 1] = buy_wheat
    market[1, 1, 1] = buy_wheat

    env.step(market_actions=market, market_lengths=lengths)
    with_gap = env.state_snapshot(0)
    compacted = env.state_snapshot(1)

    assert [farm["shed"][Item.WHEAT] for farm in with_gap["farms"]] == [2, 1]
    assert with_gap["market"]["inventory"] == compacted["market"]["inventory"]
    # Market slots resolve in lockstep. Keeping the internal NONE causes the
    # second player to buy alone before player zero's final order, changing its
    # refreshed quote by one compared with the deliberately compacted case.
    assert [farm["money"] for farm in with_gap["farms"]] == [2_947.0, 2_974.0]
    assert [farm["money"] for farm in compacted["farms"]] == [2_948.0, 2_974.0]


def test_explicit_reset_clears_the_retained_terminal_snapshot() -> None:
    env = _market_env(
        auto_reset=True,
        episode_steps=2,
        max_market_orders=1,
    )
    env.reset()
    assert env.terminal_snapshot(0) is None

    env.step()
    assert env.terminal_snapshot(0) is not None
    assert env.state_snapshot(0)["episode_id"] == 1

    env.reset()
    assert env.terminal_snapshot(0) is None
    assert env.state_snapshot(0)["episode_id"] == 0


def test_player_relative_private_features_do_not_expose_opponent_seeds() -> None:
    env = _market_env(max_market_orders=1)
    initial = env.reset().observation_views.private.copy()
    market = np.zeros((1, 2, env.max_orders, 3), dtype=np.int64)
    market[0, 0, 0] = (MarketOp.BUY_SEED, Item.CARROT, 1)
    lengths = np.array([[1, 0]], dtype=np.int64)

    stepped = env.step(market_actions=market, market_lengths=lengths)
    private = stepped.observation_views.private
    carrot_seed_channel = 12 + int(Item.CARROT)

    assert private[0, 0, carrot_seed_channel] == pytest.approx(0.1)
    np.testing.assert_array_equal(private[0, 1], initial[0, 1])

    # The observation did update with public opponent information, so the
    # unchanged private slice is specifically about visibility, not staleness.
    assert stepped.observation_views.farms[0, 1, 1, 0] == pytest.approx(
        2_980 / 3_000
    )
    assert env.state_snapshot(0)["farms"][0]["seeds"][Item.CARROT] == 1


def test_no_autoreset_delivers_terminal_reward_once_and_requires_reset() -> None:
    env = VecEnv(
        1,
        auto_reset=False,
        episode_steps=2,
        max_market_orders=1,
        turns_per_day=2,
        weed_spawn_chance=0.0,
    )
    env.reset()
    terminal = env.step()
    np.testing.assert_array_equal(terminal.dones, [[True, True]])
    np.testing.assert_array_equal(terminal.rewards, [[3_000.0, 3_000.0]])
    snapshot = copy.deepcopy(env.state_snapshot(0))

    with pytest.raises(ValueError, match="is terminal; call reset"):
        env.step()
    assert env.state_snapshot(0) == snapshot

    env.auto_reset = True
    with pytest.raises(ValueError, match="is terminal; call reset"):
        env.step()
    env.reset()
    assert env.state_snapshot(0)["done"] is False


def _native_outputs(native: NativeVecEnv) -> tuple[np.ndarray, ...]:
    specs = native.buffer_specs()
    return (
        np.zeros(specs["observation_shape"], dtype=np.float32),
        np.zeros(specs["action_mask_shape"], dtype=np.uint8),
        np.zeros(specs["unit_active_shape"], dtype=np.uint8),
        np.zeros(specs["reward_shape"], dtype=np.float64),
        np.zeros(specs["reward_shape"], dtype=np.float64),
        np.zeros(specs["reward_shape"], dtype=np.float64),
        np.zeros(specs["done_shape"], dtype=np.uint8),
        np.zeros(specs["episode_id_shape"], dtype=np.uint64),
        np.zeros(specs["overflow_shape"], dtype=np.uint8),
    )


def test_native_boundary_rejects_fortran_and_readonly_buffers_without_mutation() -> None:
    native = NativeVecEnv(
        1,
        episode_steps=20,
        max_market_orders=1,
        turns_per_day=2,
        weed_spawn_chance=0.0,
    )
    outputs = _native_outputs(native)
    native.reset_into(None, *outputs)
    before = copy.deepcopy(native.state_snapshot(0))
    specs = native.buffer_specs()
    units = np.asfortranarray(
        np.zeros(specs["unit_action_shape"], dtype=np.int64)
    )
    market = np.zeros(specs["market_action_shape"], dtype=np.int64)
    lengths = np.zeros(specs["market_length_shape"], dtype=np.int64)

    assert not units.flags.c_contiguous
    with pytest.raises(ValueError, match="unit_actions must be a C-contiguous"):
        native.step_into(units, market, lengths, *outputs)
    assert native.state_snapshot(0) == before

    outputs[0].setflags(write=False)
    with pytest.raises(ValueError, match="could not borrow observations"):
        native.reset_into(None, *outputs)
    assert native.state_snapshot(0) == before

    outputs[0].setflags(write=True)
    contiguous_units = np.zeros(specs["unit_action_shape"], dtype=np.int64)
    native.step_into(contiguous_units, market, lengths, *outputs)
    before_overlap = copy.deepcopy(native.state_snapshot(0))
    mask_values = int(np.prod(specs["action_mask_shape"]))
    active_values = int(np.prod(specs["unit_active_shape"]))
    shared = np.zeros(max(mask_values, active_values), dtype=np.uint8)
    overlapping_outputs = list(outputs)
    overlapping_outputs[1] = shared[:mask_values].reshape(
        specs["action_mask_shape"]
    )
    overlapping_outputs[2] = shared[:active_values].reshape(
        specs["unit_active_shape"]
    )
    with pytest.raises(ValueError, match="could not borrow unit_active"):
        native.reset_into(None, *overlapping_outputs)
    assert native.state_snapshot(0) == before_overlap


def test_binding_config_validation_matches_supported_schema_domain() -> None:
    with pytest.raises(ValueError, match="starting_money cannot be negative"):
        VecEnv(1, starting_money=-1)

    # The Kaggle schema gives weed probability a minimum but no maximum;
    # values above one simply make every eligible draw succeed.
    VecEnv(1, weed_spawn_chance=1.5)

    with pytest.raises(ValueError, match="dimensions overflow"):
        VecEnv(
            1,
            max_units=2**63,
            max_market_orders=1,
            turns_per_day=2,
        )
