"""Validated construction of the baseline actor-critic."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from .actor_critic import ActorCritic
from .encoder import SpatialGlobalEncoder
from .heads import ValueHead, WorkerActorHead, WorkforceHead


@dataclass(frozen=True, slots=True)
class ActorCriticConfig:
    """Small defaults intended for iteration before expensive scaling runs."""

    spatial_channels: int = 48
    global_channels: int = 77
    worker_channels: int = 29
    d_model: int = 64
    n_blocks: int = 5
    kernel_size: int = 3
    dropout: float = 0.0
    max_hands: int = 16

    def __post_init__(self) -> None:
        positive = {
            "spatial_channels": self.spatial_channels,
            "global_channels": self.global_channels,
            "worker_channels": self.worker_channels,
            "d_model": self.d_model,
            "n_blocks": self.n_blocks,
            "kernel_size": self.kernel_size,
            "max_hands": self.max_hands,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-padded convolutions")
        if not 0.0 <= self.dropout <= 0.2:
            raise ValueError("dropout must be between 0 and 0.2")


def build_actor_critic(
    config: ActorCriticConfig | None = None,
) -> ActorCritic:
    """Build the baseline residual-CNN actor-critic."""

    config = config or ActorCriticConfig()
    activation = nn.GELU
    return ActorCritic(
        encoder=SpatialGlobalEncoder(
            spatial_channels=config.spatial_channels,
            global_channels=config.global_channels,
            d_model=config.d_model,
            n_blocks=config.n_blocks,
            activation=activation,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        ),
        worker_head=WorkerActorHead(
            d_model=config.d_model,
            worker_channels=config.worker_channels,
            activation=activation,
        ),
        workforce_head=WorkforceHead(
            d_model=config.d_model,
            max_hands=config.max_hands,
            activation=activation,
        ),
        value_head=ValueHead(
            d_model=config.d_model,
            activation=activation,
        ),
    )
