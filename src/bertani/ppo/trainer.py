"""PPO minibatch optimization, separate from environment collection."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

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
    explained_variance: float
    learning_rate: float
    samples_per_second: float
    prepare_seconds: float
    device_transfer_seconds: float
    forward_seconds: float
    backward_seconds: float
    optimizer_seconds: float
    update_seconds: float
    peak_gpu_memory_mb: float
    profile_synchronized: float

    def as_dict(self) -> dict[str, float]:
        return {
            "total_loss": self.total_loss,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "clip_fraction": self.clip_fraction,
            "approximate_kl": self.approximate_kl,
            "gradient_norm": self.gradient_norm,
            "reward_mean": self.reward_mean,
            "advantage_mean": self.advantage_mean,
            "return_mean": self.return_mean,
            "explained_variance": self.explained_variance,
            "learning_rate": self.learning_rate,
            "samples_per_second": self.samples_per_second,
            "prepare_seconds": self.prepare_seconds,
            "device_transfer_seconds": self.device_transfer_seconds,
            "forward_seconds": self.forward_seconds,
            "backward_seconds": self.backward_seconds,
            "optimizer_seconds": self.optimizer_seconds,
            "update_seconds": self.update_seconds,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
            "profile_synchronized": self.profile_synchronized,
        }


@dataclass(frozen=True, slots=True)
class _MinibatchResult:
    metrics: torch.Tensor
    forward_seconds: float
    backward_seconds: float
    optimizer_seconds: float


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
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
            torch.backends.cudnn.allow_tf32 = config.allow_tf32
            torch.backends.cudnn.benchmark = config.cudnn_benchmark
            if config.allow_tf32:
                torch.set_float32_matmul_precision("high")
            if config.channels_last:
                self.model.to(memory_format=torch.channels_last)
        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            eps=config.adam_epsilon,
            fused=config.fused_optimizer and self.device.type == "cuda",
        )
        use_scaler = config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        if config.compile_model and self.device.type == "cuda":
            self.model.compile(mode=config.compile_mode, dynamic=False)
        self.updates = 0

    def update(
        self,
        rollout: RolloutBatch,
        *,
        minibatch_callback: Callable[[], None] | None = None,
    ) -> PPOStats:
        update_started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        rollout.validate()
        prepare_started = time.perf_counter()
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
        prepare_seconds = time.perf_counter() - prepare_started
        sample_count = rollout.steps * rollout.environments
        accumulated: list[_MinibatchResult] = []
        device_transfer_seconds = 0.0
        on_device = self.config.preload_rollout and self.device.type == "cuda"
        if on_device:
            self._synchronize()
            transfer_started = time.perf_counter()
            training = training.to_device(
                self.device,
                channels_last=self.config.channels_last,
            )
            self._synchronize()
            device_transfer_seconds += time.perf_counter() - transfer_started
        for _ in range(self.config.epochs_per_update):
            permutation = torch.randperm(
                sample_count,
                device=self.device if on_device else "cpu",
            )
            for start in range(0, sample_count, self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                if on_device:
                    minibatch = training.index(indices)
                else:
                    self._synchronize()
                    transfer_started = time.perf_counter()
                    minibatch = training.index(indices).to_device(
                        self.device,
                        channels_last=(
                            self.config.channels_last and self.device.type == "cuda"
                        ),
                    )
                    self._synchronize()
                    device_transfer_seconds += time.perf_counter() - transfer_started
                accumulated.append(self._update_minibatch(minibatch))
                if minibatch_callback is not None:
                    minibatch_callback()
        self.updates += 1
        # One D2H copy replaces several implicit float(cuda_tensor)
        # synchronizations for every minibatch.
        metric_values = (
            torch.stack([batch.metrics for batch in accumulated])
            .float()
            .mean(dim=0)
            .cpu()
            .tolist()
        )
        update_seconds = time.perf_counter() - update_started
        (
            total_loss,
            actor_loss,
            critic_loss,
            entropy,
            clip_fraction,
            approximate_kl,
            gradient_norm,
        ) = metric_values

        flat_returns = returns.flatten()
        flat_values = rollout.values[:-1].flatten()
        return_variance = flat_returns.var(unbiased=False)
        explained_variance = (
            1.0 - (flat_returns - flat_values).var(unbiased=False) / return_variance
            if float(return_variance) > 0.0
            else torch.zeros(())
        )
        processed_samples = sample_count * self.config.epochs_per_update
        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
            if self.device.type == "cuda"
            else 0.0
        )

        return PPOStats(
            total_loss=total_loss,
            policy_loss=actor_loss,
            value_loss=critic_loss,
            entropy=entropy,
            clip_fraction=clip_fraction,
            approximate_kl=approximate_kl,
            gradient_norm=gradient_norm,
            reward_mean=float(rollout.rewards.mean()),
            advantage_mean=float(advantages.mean()),
            return_mean=float(returns.mean()),
            explained_variance=float(explained_variance),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            samples_per_second=processed_samples / max(update_seconds, 1e-9),
            prepare_seconds=prepare_seconds,
            device_transfer_seconds=device_transfer_seconds,
            forward_seconds=sum(batch.forward_seconds for batch in accumulated),
            backward_seconds=sum(batch.backward_seconds for batch in accumulated),
            optimizer_seconds=sum(batch.optimizer_seconds for batch in accumulated),
            update_seconds=update_seconds,
            peak_gpu_memory_mb=peak_gpu_memory_mb,
            profile_synchronized=float(self.config.profile),
        )

    def _update_minibatch(self, batch: TrainingBatch) -> _MinibatchResult:
        use_amp = self.scaler.is_enabled()
        # Release the preceding minibatch's gradient buffers before allocating
        # the next forward activations.
        self.optimizer.zero_grad(set_to_none=True)
        self._synchronize()
        forward_started = time.perf_counter()
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
        self._synchronize()
        forward_seconds = time.perf_counter() - forward_started

        self._synchronize()
        backward_started = time.perf_counter()
        self.scaler.scale(total_loss).backward()
        self._synchronize()
        backward_seconds = time.perf_counter() - backward_started

        optimizer_started = time.perf_counter()
        self.scaler.unscale_(self.optimizer)
        if self.config.max_gradient_norm is None:
            gradient_norm = self._gradient_norm()
        else:
            gradient_norm = clip_grad_norm_(
                self.model.parameters(), self.config.max_gradient_norm
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self._synchronize()
        optimizer_seconds = time.perf_counter() - optimizer_started

        with torch.no_grad():
            clip_fraction = (
                ((probability_ratio - 1.0).abs() > self.config.clip_coefficient)
                .float()
                .mean()
            )
            approximate_kl = ((probability_ratio - 1.0) - log_ratio).mean()
        return _MinibatchResult(
            metrics=torch.stack(
                (
                    total_loss.detach(),
                    actor_loss.detach(),
                    critic_loss.detach(),
                    entropy.detach(),
                    clip_fraction.detach(),
                    approximate_kl.detach(),
                    gradient_norm.detach(),
                )
            ),
            forward_seconds=forward_seconds,
            backward_seconds=backward_seconds,
            optimizer_seconds=optimizer_seconds,
        )

    def _synchronize(self) -> None:
        if self.config.profile and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _gradient_norm(self) -> torch.Tensor:
        squared = torch.zeros((), device=self.device)
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                squared += parameter.grad.detach().square().sum()
        return squared.sqrt()


__all__ = ["PPOStats", "PPOTrainer"]
