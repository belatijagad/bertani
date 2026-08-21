"""Typed rollout storage kept on CPU between collection and PPO epochs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from ..models import TorchActionInfo, TorchObservation


class PPOActions(NamedTuple):
    operations: torch.Tensor
    arguments: torch.Tensor
    target_hands: torch.Tensor

    def index(self, index: torch.Tensor | slice) -> PPOActions:
        return PPOActions(*(value[index] for value in self))

    def to_device(self, device: torch.device | str) -> PPOActions:
        return PPOActions(*(value.to(device, non_blocking=True) for value in self))


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    observation: TorchObservation
    action_info: TorchActionInfo
    actions: PPOActions
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def index(self, index: torch.Tensor | slice) -> TrainingBatch:
        return TrainingBatch(
            observation=self.observation.index(index),
            action_info=self.action_info.index(index),
            actions=self.actions.index(index),
            old_log_probs=self.old_log_probs[index],
            advantages=self.advantages[index],
            returns=self.returns[index],
        )

    def to_device(
        self,
        device: torch.device | str,
        *,
        channels_last: bool = False,
    ) -> TrainingBatch:
        return TrainingBatch(
            observation=self.observation.to_device(device, channels_last=channels_last),
            action_info=self.action_info.to_device(device),
            actions=self.actions.to_device(device),
            old_log_probs=self.old_log_probs.to(device, non_blocking=True),
            advantages=self.advantages.to(device, non_blocking=True),
            returns=self.returns.to(device, non_blocking=True),
        )


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    observation: TorchObservation
    action_info: TorchActionInfo
    actions: PPOActions
    old_log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor

    @property
    def steps(self) -> int:
        return self.rewards.shape[0]

    @property
    def environments(self) -> int:
        return self.rewards.shape[1]

    def validate(self) -> None:
        prefix = self.rewards.shape
        if self.dones.shape != prefix:
            raise ValueError("reward and done shapes differ")
        if self.values.shape != (prefix[0] + 1, prefix[1]):
            raise ValueError("values must have shape [steps + 1, environments]")
        for value in (*self.observation, *self.action_info, *self.actions):
            if value.shape[:2] != prefix:
                raise ValueError("rollout tensor has an incompatible prefix")
        if self.old_log_probs.shape != prefix:
            raise ValueError("old_log_probs has an incompatible shape")

    def training_batch(
        self,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> TrainingBatch:
        self.validate()
        if (
            advantages.shape != self.rewards.shape
            or returns.shape != self.rewards.shape
        ):
            raise ValueError("advantage and return shapes must match rewards")
        samples = self.steps * self.environments

        def flatten(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(samples, *value.shape[2:])

        return TrainingBatch(
            observation=TorchObservation(
                *(flatten(value) for value in self.observation)
            ),
            action_info=TorchActionInfo(
                *(flatten(value) for value in self.action_info)
            ),
            actions=PPOActions(*(flatten(value) for value in self.actions)),
            old_log_probs=self.old_log_probs.reshape(samples),
            advantages=advantages.reshape(samples),
            returns=returns.reshape(samples),
        )

    @classmethod
    def from_lists(
        cls,
        *,
        observations: list[TorchObservation],
        action_info: list[TorchActionInfo],
        actions: list[PPOActions],
        old_log_probs: list[torch.Tensor],
        values: list[torch.Tensor],
        rewards: list[torch.Tensor],
        dones: list[torch.Tensor],
    ) -> RolloutBatch:
        rollout = cls(
            observation=TorchObservation(
                *(torch.stack(values) for values in zip(*observations, strict=True))
            ),
            action_info=TorchActionInfo(
                *(torch.stack(values) for values in zip(*action_info, strict=True))
            ),
            actions=PPOActions(
                *(torch.stack(values) for values in zip(*actions, strict=True))
            ),
            old_log_probs=torch.stack(old_log_probs),
            values=torch.stack(values),
            rewards=torch.stack(rewards),
            dones=torch.stack(dones),
        )
        rollout.validate()
        return rollout


__all__ = ["PPOActions", "RolloutBatch", "TrainingBatch"]
