from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bertani._rust", reason="the maturin extension has not been built")

from bertani import MarketOp, RuleActions, V9SelfPlayEnv, VecEnv
from bertani.models import (
    ActorCriticConfig,
    TorchActionInfo,
    TorchObservation,
    build_actor_critic,
)
from bertani.ppo import (
    CompetitiveReward,
    PPOConfig,
    PPOTrainer,
    WorkforceMarketPolicy,
    clipped_policy_loss,
    collect_rollout,
    generalized_advantage_estimate,
)


class _PassOpponent:
    def __init__(self, environment: VecEnv) -> None:
        self.environment = environment
        self.cache_stats = type("Stats", (), {"hits": 0, "misses": 0})()

    def reset(self) -> None:
        pass

    def act(
        self, environment: VecEnv, batch: object, *, seats: np.ndarray
    ) -> RuleActions:
        del batch, seats
        units = np.zeros(
            (environment.num_envs, 2, environment.max_units, 3), dtype=np.int64
        )
        market = np.zeros(
            (environment.num_envs, 2, environment.max_orders, 3), dtype=np.int64
        )
        lengths = np.zeros((environment.num_envs, 2), dtype=np.int64)
        return RuleActions(units, market, lengths)


class _BaseMarket:
    def act(
        self,
        batch: object,
        max_orders: int,
        seat_mask: np.ndarray,
    ) -> RuleActions:
        del batch
        environments = seat_mask.shape[0]
        units = np.zeros((environments, 2, 1, 3), dtype=np.int64)
        market = np.zeros((environments, 2, max_orders, 3), dtype=np.int64)
        lengths = np.zeros((environments, 2), dtype=np.int64)
        games, seats = np.nonzero(seat_mask)
        market[games, seats, 0] = (MarketOp.HIRE, 0, 0)
        market[games, seats, 1] = (MarketOp.BUY_SEED, 0, 3)
        lengths[games, seats] = 2
        return RuleActions(units, market, lengths)


def test_selected_seat_adapter_matches_flat_player_rows() -> None:
    environment = VecEnv(3, episode_steps=4, turns_per_day=2)
    batch = environment.reset(np.asarray([1, 2, 3], dtype=np.uint64))
    seats = np.asarray([0, 1, 0], dtype=np.int64)
    flat_indices = torch.tensor([0, 3, 4])

    full_observation = TorchObservation.from_batch(batch)
    selected_observation = TorchObservation.from_batch_seats(batch, seats)
    full_action_info = TorchActionInfo.from_batch(batch)
    selected_action_info = TorchActionInfo.from_batch_seats(batch, seats)

    for full, selected in zip(full_observation, selected_observation, strict=True):
        torch.testing.assert_close(full[flat_indices], selected)
    for full, selected in zip(full_action_info, selected_action_info, strict=True):
        torch.testing.assert_close(full[flat_indices], selected)


def test_generalized_advantage_estimate_bootstraps_backwards() -> None:
    values = torch.zeros((3, 1))
    rewards = torch.tensor([[1.0], [1.0]])
    dones = torch.zeros((2, 1), dtype=torch.bool)

    advantages, returns = generalized_advantage_estimate(
        values, rewards, dones, gamma=1.0, gae_lambda=1.0
    )

    torch.testing.assert_close(advantages[:, 0], torch.tensor([2.0, 1.0]))
    torch.testing.assert_close(returns, advantages)


def test_clipped_policy_loss_uses_pessimistic_surrogate() -> None:
    advantages = torch.tensor([1.0, -1.0])
    ratios = torch.tensor([1.5, 0.5])

    loss = clipped_policy_loss(advantages, ratios, 0.2)

    assert loss.item() == pytest.approx(-0.2)


def test_workforce_market_replaces_base_hires_and_retains_economy() -> None:
    environment = VecEnv(2, episode_steps=4, turns_per_day=2)
    batch = environment.reset(np.asarray([9, 10], dtype=np.uint64))
    seats = np.asarray([0, 1], dtype=np.int64)
    market_policy = WorkforceMarketPolicy(
        _BaseMarket(),
        max_hires_per_turn=2,  # type: ignore[arg-type]
    )

    actions, lengths = market_policy.actions(
        batch,
        seats,
        np.asarray([3, 3], dtype=np.int64),
        max_orders=environment.max_orders,
    )

    np.testing.assert_array_equal(lengths, [3, 3])
    np.testing.assert_array_equal(
        actions[:, :3, 0],
        [
            [MarketOp.HIRE, MarketOp.HIRE, MarketOp.BUY_SEED],
            [MarketOp.HIRE, MarketOp.HIRE, MarketOp.BUY_SEED],
        ],
    )


def test_collect_and_update_ppo_smoke() -> None:
    torch.manual_seed(12)
    environment = VecEnv(
        2,
        auto_reset=True,
        episode_steps=3,
        turns_per_day=3,
        weed_spawn_chance=0.0,
    )
    self_play = V9SelfPlayEnv(
        environment,
        _PassOpponent(environment),  # type: ignore[arg-type]
    )
    batch = self_play.reset(np.asarray([21, 22], dtype=np.uint64))
    model = build_actor_critic(ActorCriticConfig(d_model=16, n_blocks=1, max_hands=3))
    config = PPOConfig(
        steps_per_update=2,
        epochs_per_update=2,
        minibatch_size=2,
        mixed_precision=False,
        profile=True,
    )
    reward = CompetitiveReward()
    reward.reset(self_play, batch)
    collection = collect_rollout(
        self_play,
        model,
        WorkforceMarketPolicy(max_hires_per_turn=1),
        reward,
        config,
        device=torch.device("cpu"),
    )
    rollout = collection.rollout
    trainer = PPOTrainer(model, config, device="cpu")
    before = next(model.parameters()).detach().clone()

    stats = trainer.update(rollout)

    assert rollout.rewards.shape == (2, 2)
    assert rollout.values.shape == (3, 2)
    assert collection.profile.transitions == 4
    assert collection.profile.total_seconds > 0.0
    assert collection.profile.policy_forward_seconds > 0.0
    assert collection.profile.synchronized
    assert collection.episodes.completed == 2
    assert (
        collection.episodes.wins + collection.episodes.ties + collection.episodes.losses
        == 2
    )
    assert trainer.updates == 1
    assert not torch.equal(before, next(model.parameters()).detach())
    assert all(np.isfinite(value) for value in stats.as_dict().values())
    assert stats.samples_per_second > 0.0
    assert stats.forward_seconds > 0.0
    assert stats.backward_seconds > 0.0
    assert stats.optimizer_seconds > 0.0
    assert stats.profile_synchronized == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_update_uses_gpu_optimized_layout_and_rollout_preload() -> None:
    torch.manual_seed(13)
    environment = VecEnv(
        2,
        auto_reset=True,
        episode_steps=3,
        turns_per_day=3,
        weed_spawn_chance=0.0,
    )
    self_play = V9SelfPlayEnv(
        environment,
        _PassOpponent(environment),  # type: ignore[arg-type]
    )
    batch = self_play.reset(np.asarray([31, 32], dtype=np.uint64))
    model = build_actor_critic(ActorCriticConfig(d_model=16, n_blocks=1, max_hands=3))
    config = PPOConfig(
        steps_per_update=2,
        epochs_per_update=2,
        minibatch_size=2,
        compile_model=False,
        mixed_precision=True,
        profile=True,
    )
    reward = CompetitiveReward()
    reward.reset(self_play, batch)
    trainer = PPOTrainer(model, config, device="cuda")
    collection = collect_rollout(
        self_play,
        model,
        WorkforceMarketPolicy(max_hires_per_turn=1),
        reward,
        config,
        device=torch.device("cuda"),
    )

    stats = trainer.update(collection.rollout)

    convolution = model.encoder.spatial_input[0]
    assert convolution.weight.is_contiguous(memory_format=torch.channels_last)
    assert trainer.scaler.is_enabled()
    assert stats.device_transfer_seconds > 0.0
    assert stats.peak_gpu_memory_mb > 0.0


def test_ppo_config_rejects_unknown_compile_mode() -> None:
    with pytest.raises(ValueError, match="compile_mode must be one of"):
        PPOConfig(compile_mode="fastest-please")
