"""Actor-critic composition and sampled action representation."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from ..vec_env import UnitOp
from .heads import ValueHead, WorkerActorHead, WorkforceHead
from .types import TorchActionInfo, TorchObservation


class ActorCriticOutput(NamedTuple):
    """Policy distributions, sampled actions, and scalar value estimates."""

    operation_log_probs: torch.Tensor
    argument_log_probs: torch.Tensor
    operations: torch.Tensor
    arguments: torch.Tensor
    workforce_log_probs: torch.Tensor
    target_hands: torch.Tensor
    value: torch.Tensor

    def worker_log_probs(self, active_workers: torch.Tensor) -> torch.Tensor:
        """Return selected joint operation/argument log-probability per worker."""

        operation = self.operation_log_probs.gather(
            -1, self.operations.unsqueeze(-1)
        ).squeeze(-1)
        by_operation = self.argument_log_probs.gather(
            -2,
            self.operations[..., None, None].expand(
                -1, -1, 1, self.argument_log_probs.shape[-1]
            ),
        ).squeeze(-2)
        argument = by_operation.gather(-1, self.arguments.unsqueeze(-1)).squeeze(-1)
        return torch.where(active_workers, operation + argument, 0.0)

    def joint_log_probs(
        self,
        active_workers: torch.Tensor,
        *,
        include_workforce: bool = True,
    ) -> torch.Tensor:
        """Return the team-level log-probability of the sampled decisions."""

        result = self.worker_log_probs(active_workers).sum(dim=-1)
        if include_workforce:
            result = result + self.workforce_log_prob()
        return result

    def workforce_log_prob(self) -> torch.Tensor:
        """Return the selected global workforce log-probability."""

        return self.workforce_log_probs.gather(
            -1, self.target_hands.unsqueeze(-1)
        ).squeeze(-1)

    def to_unit_actions(self) -> torch.Tensor:
        """Convert sampled worker decisions to ``(op, argument, count)`` rows."""

        counts = (
            (self.operations == int(UnitOp.PICKUP))
            | (self.operations == int(UnitOp.PLACE))
        ).long()
        return torch.stack((self.operations, self.arguments, counts), dim=-1)


class ActorCritic(nn.Module):
    """Shared encoder with worker, workforce, and value heads."""

    def __init__(
        self,
        encoder: nn.Module,
        worker_head: WorkerActorHead,
        workforce_head: WorkforceHead,
        value_head: ValueHead,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.worker_head = worker_head
        self.workforce_head = workforce_head
        self.value_head = value_head

    def forward(
        self,
        observation: TorchObservation,
        action_info: TorchActionInfo,
        *,
        worker_temperature: float | None = None,
        workforce_temperature: float | None = None,
        omit_value: bool = False,
        operations: torch.Tensor | None = None,
        arguments: torch.Tensor | None = None,
        target_hands: torch.Tensor | None = None,
    ) -> ActorCriticOutput:
        # Sampled categorical IDs are stored/transferred as int16. Gather and
        # categorical indexing require int64, but only during this forward.
        if operations is not None:
            operations = operations.long()
        if arguments is not None:
            arguments = arguments.long()
        if target_hands is not None:
            target_hands = target_hands.long()
        encoded_map = self.encoder(observation)
        operation_log_probs, argument_log_probs, operations, arguments = (
            self.worker_head(
                encoded_map,
                observation,
                action_info,
                worker_temperature,
                operations=operations,
                arguments=arguments,
            )
        )
        workforce_log_probs, target_hands = self.workforce_head(
            encoded_map,
            workforce_temperature,
            target_hands=target_hands,
        )
        value = (
            torch.zeros(
                encoded_map.shape[0],
                device=encoded_map.device,
                dtype=encoded_map.dtype,
            )
            if omit_value
            else self.value_head(encoded_map)
        )
        return ActorCriticOutput(
            operation_log_probs=operation_log_probs,
            argument_log_probs=argument_log_probs,
            operations=operations,
            arguments=arguments,
            workforce_log_probs=workforce_log_probs,
            target_hands=target_hands,
            value=value,
        )
