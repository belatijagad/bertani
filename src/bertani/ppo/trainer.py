"""PPO minibatch optimization, separate from environment collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch.nn.utils import clip_grad_norm_

from ..models import ActorCritic
from .config import PPOConfig
from .math import (
    clipped_policy_loss,
    generalized_advantage_estimate,
    policy_entropy,
    value_loss,
)
from .storage import RolloutBatch, TrainingBatch


@dataclass(frozen=True, slots=True)
class PPOStats:
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    clip_fraction: float
    approximate_kl: float
    gradient_norm: float
    reward_mean: float
    advantage_mean: float
    return_mean: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class PPOTrainer:
    """Own optimizer state and update an actor-critic from CPU rollouts."""

    def __init__(
        self,
        model: ActorCritic,
        config: PPOConfig,
        *,
        device: torch.device | str,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)
        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            eps=config.adam_epsilon,
        )
        use_scaler = config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        self.updates = 0

    def update(self, rollout: RolloutBatch) -> PPOStats:
        rollout.validate()
        advantages, returns = generalized_advantage_estimate(
            rollout.values,
            rollout.rewards,
            rollout.dones,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
        training = rollout.training_batch(advantages, returns)
        sample_count = rollout.steps * rollout.environments
        accumulated: list[dict[str, float]] = []
        for _ in range(self.config.epochs_per_update):
            permutation = torch.randperm(sample_count)
            for start in range(0, sample_count, self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                minibatch = training.index(indices).to_device(self.device)
                accumulated.append(self._update_minibatch(minibatch))
        self.updates += 1

        def mean(key: str) -> float:
            return sum(batch[key] for batch in accumulated) / len(accumulated)

        return PPOStats(
            total_loss=mean("total_loss"),
            policy_loss=mean("policy_loss"),
            value_loss=mean("value_loss"),
            entropy=mean("entropy"),
            clip_fraction=mean("clip_fraction"),
            approximate_kl=mean("approximate_kl"),
            gradient_norm=mean("gradient_norm"),
            reward_mean=float(rollout.rewards.mean()),
            advantage_mean=float(advantages.mean()),
            return_mean=float(returns.mean()),
        )

    def _update_minibatch(self, batch: TrainingBatch) -> dict[str, float]:
        use_amp = self.scaler.is_enabled()
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            output = self.model(
                batch.observation,
                batch.action_info,
                operations=batch.actions.operations,
                arguments=batch.actions.arguments,
                target_hands=batch.actions.target_hands,
            )
            new_log_probs = output.joint_log_probs(
                batch.action_info.active_workers,
                include_workforce=self.config.include_workforce,
            )
            log_ratio = new_log_probs - batch.old_log_probs
            probability_ratio = log_ratio.exp()
            actor_loss = clipped_policy_loss(
                batch.advantages,
                probability_ratio,
                self.config.clip_coefficient,
            )
            critic_loss = value_loss(output.value, batch.returns)
            entropy = policy_entropy(
                output,
                batch.action_info,
                include_workforce=self.config.include_workforce,
            )
            total_loss = (
                actor_loss
                + self.config.value_coefficient * critic_loss
                - self.config.entropy_coefficient * entropy
            )

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        if self.config.max_gradient_norm is None:
            gradient_norm = self._gradient_norm()
        else:
            gradient_norm = clip_grad_norm_(
                self.model.parameters(), self.config.max_gradient_norm
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            clip_fraction = (
                (probability_ratio - 1.0).abs() > self.config.clip_coefficient
            ).float().mean()
            approximate_kl = ((probability_ratio - 1.0) - log_ratio).mean()
        return {
            "total_loss": float(total_loss.detach()),
            "policy_loss": float(actor_loss.detach()),
            "value_loss": float(critic_loss.detach()),
            "entropy": float(entropy.detach()),
            "clip_fraction": float(clip_fraction),
            "approximate_kl": float(approximate_kl),
            "gradient_norm": float(gradient_norm),
        }

    def _gradient_norm(self) -> torch.Tensor:
        squared = torch.zeros((), device=self.device)
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                squared += parameter.grad.detach().square().sum()
        return squared.sqrt()


__all__ = ["PPOStats", "PPOTrainer"]
