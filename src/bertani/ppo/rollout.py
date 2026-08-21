"""Trajectory collection against a frozen batched self-play opponent."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from ..models import ActorCritic, TorchActionInfo, TorchObservation
from ..self_play import SelfPlayEnv
from ..vec_env import MarketOp
from .config import PPOConfig
from .market import LearnerMarketPolicy
from .rewards import CompetitiveReward
from .storage import PPOActions, RolloutBatch


@dataclass(frozen=True, slots=True)
class EpisodeStats:
    completed: int = 0
    wins: int = 0
    ties: int = 0
    losses: int = 0
    final_margin_sum: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.completed if self.completed else 0.0

    @property
    def mean_final_margin(self) -> float:
        return self.final_margin_sum / self.completed if self.completed else 0.0


@dataclass(frozen=True, slots=True)
class WorkforceStats:
    """Aggregate neural workforce decisions and resulting hire activity."""

    decisions: int = 0
    target_hands_sum: int = 0
    current_hands_sum: int = 0
    targets_met: int = 0
    hire_orders: int = 0
    observed_hires: int = 0

    @property
    def mean_target_hands(self) -> float:
        return self.target_hands_sum / self.decisions if self.decisions else 0.0

    @property
    def mean_current_hands(self) -> float:
        return self.current_hands_sum / self.decisions if self.decisions else 0.0

    @property
    def target_met_rate(self) -> float:
        return self.targets_met / self.decisions if self.decisions else 0.0


@dataclass(frozen=True, slots=True)
class RolloutProfile:
    total_seconds: float
    observation_seconds: float
    device_transfer_seconds: float
    policy_forward_seconds: float
    action_transfer_seconds: float
    market_seconds: float
    opponent_seconds: float
    action_composition_seconds: float
    environment_seconds: float
    reward_seconds: float
    transitions: int
    opponent_cache_hits: int
    opponent_cache_misses: int
    synchronized: bool

    @property
    def transitions_per_second(self) -> float:
        return self.transitions / max(self.total_seconds, 1e-9)


@dataclass(frozen=True, slots=True)
class RolloutCollection:
    rollout: RolloutBatch
    episodes: EpisodeStats
    workforce: WorkforceStats
    profile: RolloutProfile


def _synchronize(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def collect_rollout(
    self_play: SelfPlayEnv,
    model: ActorCritic,
    market_policy: LearnerMarketPolicy,
    reward: CompetitiveReward,
    config: PPOConfig,
    *,
    device: torch.device,
    step_callback: Callable[[], None] | None = None,
) -> RolloutCollection:
    """Collect one contiguous learner trajectory block on CPU."""

    total_started = time.perf_counter()
    observations: list[TorchObservation] = []
    action_information: list[TorchActionInfo] = []
    actions: list[PPOActions] = []
    old_log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []
    use_amp = config.mixed_precision and device.type == "cuda"
    timings = {
        "observation": 0.0,
        "device_transfer": 0.0,
        "policy_forward": 0.0,
        "action_transfer": 0.0,
        "market": 0.0,
        "opponent": 0.0,
        "composition": 0.0,
        "environment": 0.0,
        "reward": 0.0,
    }
    completed = wins = ties = losses = 0
    final_margin_sum = 0.0
    workforce_decisions = 0
    target_hands_sum = 0
    current_hands_sum = 0
    workforce_targets_met = 0
    hire_orders = 0
    observed_hires = 0
    cache_before = self_play.opponent.cache_stats

    for _ in range(config.steps_per_update):
        started = time.perf_counter()
        observation = TorchObservation.from_batch_seats(
            self_play.batch, self_play.learner_seats
        )
        action_info = TorchActionInfo.from_batch_seats(
            self_play.batch, self_play.learner_seats
        )
        timings["observation"] += time.perf_counter() - started

        _synchronize(device, config.profile)
        started = time.perf_counter()
        device_observation = observation.to_device(
            device, channels_last=config.channels_last and device.type == "cuda"
        )
        device_action_info = action_info.to_device(device)
        _synchronize(device, config.profile)
        timings["device_transfer"] += time.perf_counter() - started

        _synchronize(device, config.profile)
        started = time.perf_counter()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            output = model(device_observation, device_action_info)
            joint_log_probs = output.joint_log_probs(
                device_action_info.active_workers,
                include_workforce=config.include_workforce,
            )
        _synchronize(device, config.profile)
        timings["policy_forward"] += time.perf_counter() - started

        started = time.perf_counter()
        # Pack worker actions before the device transfer. This avoids copying
        # operations and arguments once for PPO storage and again for the env.
        device_unit_actions = output.to_unit_actions()
        packed_actions = torch.cat(
            (
                device_unit_actions.flatten(start_dim=1),
                output.target_hands.unsqueeze(-1),
            ),
            dim=-1,
        ).to(torch.int16).cpu()
        unit_action_tensor = packed_actions[:, :-1].reshape_as(device_unit_actions)
        sampled_actions = PPOActions(
            unit_action_tensor[..., 0],
            unit_action_tensor[..., 1],
            packed_actions[:, -1],
        )
        unit_actions = unit_action_tensor.numpy()
        target_hands = sampled_actions.target_hands.numpy()
        games = self_play.games
        seats = self_play.learner_seats
        current_hands = (
            self_play.batch.active_units[games, seats].sum(axis=-1) - 1
        )
        workforce_decisions += len(target_hands)
        target_hands_sum += int(target_hands.sum())
        current_hands_sum += int(current_hands.sum())
        workforce_targets_met += int((current_hands >= target_hands).sum())
        timings["action_transfer"] += time.perf_counter() - started

        started = time.perf_counter()
        learner_market, learner_market_lengths = market_policy.actions(
            self_play.batch,
            self_play.learner_seats,
            target_hands,
            max_orders=self_play.environment.max_orders,
        )
        active_orders = (
            np.arange(self_play.environment.max_orders)[None, :]
            < learner_market_lengths[:, None]
        )
        hire_orders += int(
            (active_orders & (learner_market[..., 0] == int(MarketOp.HIRE))).sum()
        )
        timings["market"] += time.perf_counter() - started
        self_play.step(
            unit_actions,
            learner_market,
            learner_market_lengths,
        )
        next_hands = (
            self_play.batch.active_units[games, seats].sum(axis=-1) - 1
        )
        observed_hires += int(np.maximum(next_hands - current_hands, 0).sum())
        step_profile = self_play.last_step_profile
        timings["opponent"] += step_profile.opponent_seconds
        timings["composition"] += step_profile.composition_seconds
        timings["environment"] += step_profile.environment_seconds

        observations.append(observation)
        action_information.append(action_info)
        actions.append(sampled_actions)
        policy_values = (
            torch.stack((joint_log_probs, output.value), dim=-1).float().cpu()
        )
        old_log_probs.append(policy_values[..., 0])
        values.append(policy_values[..., 1])
        started = time.perf_counter()
        rewards.append(reward.transition(self_play, self_play.batch))
        learner_dones = self_play.learner_dones().astype(bool, copy=True)
        dones.append(torch.from_numpy(learner_dones))
        terminal = np.flatnonzero(learner_dones)
        if terminal.size:
            learner = self_play.learner_seats[terminal]
            opponent = self_play.opponent_seats[terminal]
            margins = (
                self_play.batch.rewards[terminal, learner]
                - self_play.batch.rewards[terminal, opponent]
            )
            completed += len(margins)
            wins += int((margins > 0).sum())
            ties += int((margins == 0).sum())
            losses += int((margins < 0).sum())
            final_margin_sum += float(margins.sum())
        timings["reward"] += time.perf_counter() - started
        if step_callback is not None:
            step_callback()

    started = time.perf_counter()
    bootstrap_observation = TorchObservation.from_batch_seats(
        self_play.batch, self_play.learner_seats
    )
    bootstrap_action_info = TorchActionInfo.from_batch_seats(
        self_play.batch, self_play.learner_seats
    )
    timings["observation"] += time.perf_counter() - started
    _synchronize(device, config.profile)
    started = time.perf_counter()
    bootstrap_observation = bootstrap_observation.to_device(
        device, channels_last=config.channels_last and device.type == "cuda"
    )
    bootstrap_action_info = bootstrap_action_info.to_device(device)
    _synchronize(device, config.profile)
    timings["device_transfer"] += time.perf_counter() - started
    _synchronize(device, config.profile)
    started = time.perf_counter()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=use_amp,
    ):
        bootstrap_value = model(
            bootstrap_observation,
            bootstrap_action_info,
            worker_temperature=0.0,
            workforce_temperature=0.0,
        ).value
    _synchronize(device, config.profile)
    timings["policy_forward"] += time.perf_counter() - started
    values.append(bootstrap_value.float().cpu())

    rollout = RolloutBatch.from_lists(
        observations=observations,
        action_info=action_information,
        actions=actions,
        old_log_probs=old_log_probs,
        values=values,
        rewards=rewards,
        dones=dones,
    )
    cache_after = self_play.opponent.cache_stats
    return RolloutCollection(
        rollout=rollout,
        episodes=EpisodeStats(
            completed=completed,
            wins=wins,
            ties=ties,
            losses=losses,
            final_margin_sum=final_margin_sum,
        ),
        workforce=WorkforceStats(
            decisions=workforce_decisions,
            target_hands_sum=target_hands_sum,
            current_hands_sum=current_hands_sum,
            targets_met=workforce_targets_met,
            hire_orders=hire_orders,
            observed_hires=observed_hires,
        ),
        profile=RolloutProfile(
            total_seconds=time.perf_counter() - total_started,
            observation_seconds=timings["observation"],
            device_transfer_seconds=timings["device_transfer"],
            policy_forward_seconds=timings["policy_forward"],
            action_transfer_seconds=timings["action_transfer"],
            market_seconds=timings["market"],
            opponent_seconds=timings["opponent"],
            action_composition_seconds=timings["composition"],
            environment_seconds=timings["environment"],
            reward_seconds=timings["reward"],
            transitions=config.steps_per_update * self_play.environment.num_envs,
            opponent_cache_hits=cache_after.hits - cache_before.hits,
            opponent_cache_misses=cache_after.misses - cache_before.misses,
            synchronized=config.profile,
        ),
    )


__all__ = [
    "EpisodeStats",
    "RolloutCollection",
    "RolloutProfile",
    "WorkforceStats",
    "collect_rollout",
]
