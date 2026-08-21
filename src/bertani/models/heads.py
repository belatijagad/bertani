"""Actor and critic heads for the baseline neural policy."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional

from ..vec_env import ITEM_COUNT, UNIT_ACTION_COUNT, UnitOp
from .initialization import orthogonal_initialization_
from .types import TorchActionInfo, TorchObservation


def sample_log_probs(
    log_probs: torch.Tensor,
    temperature: float | None,
) -> torch.Tensor:
    """Sample the final categorical axis, or take its mode at temperature 0."""

    if temperature is not None and temperature < 0.0:
        raise ValueError("temperature cannot be negative")
    if temperature == 0.0:
        return log_probs.argmax(dim=-1)
    probabilities = (
        log_probs.exp()
        if temperature is None
        else functional.softmax(log_probs / temperature, dim=-1)
    )
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    return torch.multinomial(flat, num_samples=1).reshape(log_probs.shape[:-1])


def safely_mask_logits(
    logits: torch.Tensor,
    available: torch.Tensor,
    active_rows: torch.Tensor,
) -> torch.Tensor:
    """Apply categorical masks without producing all-``-inf`` padded rows."""

    usable = active_rows & available.any(dim=-1)
    masked = logits.masked_fill(~available, -torch.inf)
    return torch.where(usable.unsqueeze(-1), masked, torch.zeros_like(logits))


class WorkerActorHead(nn.Module):
    """One parameter-shared operation/argument policy for every worker."""

    def __init__(
        self,
        d_model: int,
        worker_channels: int,
        *,
        activation: Callable[[], nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.worker_input = nn.Sequential(
            nn.Linear(d_model + worker_channels, d_model),
            activation(),
        )
        self.operation_output = nn.Linear(d_model, UNIT_ACTION_COUNT)
        self.argument_output = nn.Linear(d_model, UNIT_ACTION_COUNT * ITEM_COUNT)
        self._init_weights()

    def _init_weights(self) -> None:
        worker_linear, _ = self.worker_input
        orthogonal_initialization_(worker_linear)
        orthogonal_initialization_(self.operation_output, scale=0.01)
        orthogonal_initialization_(self.argument_output, scale=0.01)

    def forward(
        self,
        encoded_map: torch.Tensor,
        observation: TorchObservation,
        action_info: TorchActionInfo,
        temperature: float | None,
        *,
        operations: torch.Tensor | None = None,
        arguments: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        local = self._gather_worker_locations(encoded_map, observation.worker_positions)
        worker_embedding = self.worker_input(
            torch.cat((local, observation.workers), dim=-1)
        )
        operation_logits = self.operation_output(worker_embedding)
        argument_logits = self.argument_output(worker_embedding).unflatten(
            -1, (UNIT_ACTION_COUNT, ITEM_COUNT)
        )

        operation_log_probs = functional.log_softmax(
            safely_mask_logits(
                operation_logits,
                action_info.unit_operation_mask,
                action_info.active_workers,
            ),
            dim=-1,
        )
        argument_log_probs = functional.log_softmax(
            safely_mask_logits(
                argument_logits,
                action_info.unit_argument_mask,
                action_info.active_workers.unsqueeze(-1),
            ),
            dim=-1,
        )
        if operations is None:
            operations = sample_log_probs(operation_log_probs, temperature)
        operations = torch.where(
            action_info.active_workers,
            operations,
            int(UnitOp.PASS),
        )
        selected_argument_log_probs = argument_log_probs.gather(
            -2,
            operations[..., None, None].expand(-1, -1, 1, ITEM_COUNT),
        ).squeeze(-2)
        if arguments is None:
            arguments = sample_log_probs(selected_argument_log_probs, temperature)
        arguments = torch.where(action_info.active_workers, arguments, 0)
        return operation_log_probs, argument_log_probs, operations, arguments

    @staticmethod
    def _gather_worker_locations(
        encoded_map: torch.Tensor,
        worker_positions: torch.Tensor,
    ) -> torch.Tensor:
        batch, workers, _ = worker_positions.shape
        batch_indices = (
            torch.arange(batch, device=encoded_map.device)
            .unsqueeze(-1)
            .expand(-1, workers)
        )
        # Rollout storage keeps tiny coordinates as int16. PyTorch indexing
        # requires long, so widen only at the gather boundary.
        x = worker_positions[..., 0].long()
        y = worker_positions[..., 1].long()
        return encoded_map[batch_indices, :, y, x]


class WorkforceHead(nn.Module):
    """Categorical target for the number of hired hands, excluding the farmer."""

    def __init__(
        self,
        d_model: int,
        max_hands: int,
        *,
        activation: Callable[[], nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.max_hands = max_hands
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_model),
            activation(),
            nn.Linear(d_model, max_hands + 1),
        )
        first, _, output = self.layers
        orthogonal_initialization_(first)
        orthogonal_initialization_(output, scale=0.01)

    def forward(
        self,
        encoded_map: torch.Tensor,
        temperature: float | None,
        *,
        target_hands: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = encoded_map.flatten(start_dim=-2).mean(dim=-1)
        log_probs = functional.log_softmax(self.layers(pooled), dim=-1)
        if target_hands is None:
            target_hands = sample_log_probs(log_probs, temperature)
        return log_probs, target_hands


class ValueHead(nn.Module):
    """Unbounded scalar team-value estimate."""

    def __init__(
        self,
        d_model: int,
        *,
        activation: Callable[[], nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1),
            activation(),
            nn.Conv2d(d_model, 1, kernel_size=1),
        )
        first, _, output = self.layers
        orthogonal_initialization_(first)
        orthogonal_initialization_(output)

    def forward(self, encoded_map: torch.Tensor) -> torch.Tensor:
        return self.layers(encoded_map).flatten(start_dim=-2).mean(dim=-1).squeeze(-1)
