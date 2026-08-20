"""Small, testable PPO equations following Isaiah Pressman's layout."""

from __future__ import annotations

import torch
from torch.nn import functional

from ..models import ActorCriticOutput, TorchActionInfo


def generalized_advantage_estimate(
    values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bootstrap GAE from ``T + 1`` values and ``T`` transitions."""

    if values.shape[0] != rewards.shape[0] + 1:
        raise ValueError("values must contain one bootstrap step")
    if values.shape[1:] != rewards.shape[1:] or dones.shape != rewards.shape:
        raise ValueError("values, rewards, and dones have incompatible shapes")
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(rewards[-1])
    for step in range(rewards.shape[0] - 1, -1, -1):
        not_done = (~dones[step]).to(rewards.dtype)
        delta = (
            rewards[step]
            + gamma * values[step + 1] * not_done
            - values[step]
        )
        last_advantage = (
            delta + gamma * gae_lambda * not_done * last_advantage
        )
        advantages[step] = last_advantage
    return advantages, advantages + values[:-1]


def clipped_policy_loss(
    advantages: torch.Tensor,
    probability_ratio: torch.Tensor,
    clip_coefficient: float,
) -> torch.Tensor:
    """Return the clipped PPO surrogate loss."""

    if advantages.shape != probability_ratio.shape:
        raise ValueError("advantages and probability_ratio must have equal shapes")
    unclipped = -advantages * probability_ratio
    clipped = -advantages * probability_ratio.clamp(
        1.0 - clip_coefficient, 1.0 + clip_coefficient
    )
    return torch.maximum(unclipped, clipped).mean()


def value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    if values.shape != returns.shape:
        raise ValueError("values and returns must have equal shapes")
    return functional.huber_loss(values, returns)


def _categorical_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(log_probs)
    safe_log_probs = torch.where(finite, log_probs, 0.0)
    probabilities = torch.where(finite, log_probs.exp(), 0.0)
    return -(probabilities * safe_log_probs).sum(dim=-1)


def policy_entropy(
    output: ActorCriticOutput,
    action_info: TorchActionInfo,
    *,
    include_workforce: bool,
) -> torch.Tensor:
    """Mean entropy of the joint worker and optional workforce policy."""

    operation_entropy = _categorical_entropy(output.operation_log_probs)
    argument_entropy = _categorical_entropy(output.argument_log_probs)
    operation_probabilities = output.operation_log_probs.exp()
    worker_entropy = operation_entropy + (
        operation_probabilities * argument_entropy
    ).sum(dim=-1)
    team_entropy = torch.where(
        action_info.active_workers, worker_entropy, 0.0
    ).sum(dim=-1)
    if include_workforce:
        team_entropy = team_entropy + _categorical_entropy(
            output.workforce_log_probs
        )
    return team_entropy.mean()


__all__ = [
    "clipped_policy_loss",
    "generalized_advantage_estimate",
    "policy_entropy",
    "value_loss",
]
