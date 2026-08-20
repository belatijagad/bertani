"""Trajectory collection against the frozen V9 self-play wrapper."""

from __future__ import annotations

import torch

from ..models import ActorCritic, TorchActionInfo, TorchObservation
from ..v9_opponent import V9SelfPlayEnv
from .config import PPOConfig
from .market import LearnerMarketPolicy
from .rewards import CompetitiveReward
from .storage import PPOActions, RolloutBatch


@torch.no_grad()
def collect_rollout(
    self_play: V9SelfPlayEnv,
    model: ActorCritic,
    market_policy: LearnerMarketPolicy,
    reward: CompetitiveReward,
    config: PPOConfig,
    *,
    device: torch.device,
) -> RolloutBatch:
    """Collect one contiguous learner trajectory block on CPU."""

    observations: list[TorchObservation] = []
    action_information: list[TorchActionInfo] = []
    actions: list[PPOActions] = []
    old_log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []
    use_amp = config.mixed_precision and device.type == "cuda"

    for _ in range(config.steps_per_update):
        observation = TorchObservation.from_batch_seats(
            self_play.batch, self_play.learner_seats
        )
        action_info = TorchActionInfo.from_batch_seats(
            self_play.batch, self_play.learner_seats
        )
        device_observation = observation.to_device(device)
        device_action_info = action_info.to_device(device)
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

        sampled_actions = PPOActions(
            output.operations.cpu(),
            output.arguments.cpu(),
            output.target_hands.cpu(),
        )
        learner_market, learner_market_lengths = market_policy.actions(
            self_play.batch,
            self_play.learner_seats,
            sampled_actions.target_hands.numpy(),
            max_orders=self_play.environment.max_orders,
        )
        self_play.step(
            output.to_unit_actions().cpu().numpy(),
            learner_market,
            learner_market_lengths,
        )

        observations.append(observation)
        action_information.append(action_info)
        actions.append(sampled_actions)
        old_log_probs.append(joint_log_probs.float().cpu())
        values.append(output.value.float().cpu())
        rewards.append(reward.transition(self_play, self_play.batch))
        dones.append(
            torch.from_numpy(self_play.learner_dones().astype(bool, copy=True))
        )

    bootstrap_observation = TorchObservation.from_batch_seats(
        self_play.batch, self_play.learner_seats, device
    )
    bootstrap_action_info = TorchActionInfo.from_batch_seats(
        self_play.batch, self_play.learner_seats, device
    )
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
    values.append(bootstrap_value.float().cpu())

    return RolloutBatch.from_lists(
        observations=observations,
        action_info=action_information,
        actions=actions,
        old_log_probs=old_log_probs,
        values=values,
        rewards=rewards,
        dones=dones,
    )


__all__ = ["collect_rollout"]
