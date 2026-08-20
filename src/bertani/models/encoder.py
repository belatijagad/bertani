"""Shared spatial/global state encoder."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from .conv_blocks import ResidualConvBlock
from .initialization import orthogonal_initialization_
from .types import TorchObservation


class SpatialGlobalEncoder(nn.Module):
    """Fuse spatial and global observations into one contextual farm map."""

    def __init__(
        self,
        spatial_channels: int,
        global_channels: int,
        d_model: int,
        n_blocks: int,
        *,
        activation: Callable[[], nn.Module] = nn.GELU,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.spatial_input = nn.Sequential(
            nn.Conv2d(
                spatial_channels,
                d_model,
                kernel_size=kernel_size,
                padding="same",
            ),
            activation(),
            nn.Conv2d(
                d_model,
                d_model,
                kernel_size=kernel_size,
                padding="same",
            ),
        )
        self.global_input = nn.Sequential(
            nn.Linear(global_channels, d_model),
            activation(),
            nn.Linear(d_model, d_model),
        )
        self.residual_blocks = nn.Sequential(
            *(
                ResidualConvBlock(
                    d_model,
                    d_model,
                    activation=activation,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            )
        )
        self.apply(lambda module: orthogonal_initialization_(module, strict=False))

    def forward(self, observation: TorchObservation) -> torch.Tensor:
        spatial = self.spatial_input(observation.spatial)
        global_features = self.global_input(observation.global_features)
        return self.residual_blocks(
            spatial + global_features.unsqueeze(-1).unsqueeze(-1)
        )
