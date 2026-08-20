"""Residual convolutional blocks used by the spatial encoder."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

ActivationFactory = Callable[[], nn.Module]


class SqueezeExcitation(nn.Module):
    """Channel attention using globally pooled spatial context."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.layers = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.GELU(),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        weights = x.flatten(start_dim=-2).mean(dim=-1)
        weights = self.layers(weights).view(batch, channels, 1, 1)
        return x * weights


class ResidualConvBlock(nn.Module):
    """Two same-padded convolutions with squeeze-excitation and a skip."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation: ActivationFactory = nn.GELU,
        kernel_size: int = 3,
        dropout: float = 0.0,
        squeeze_excitation: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.activation1 = activation()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.squeeze_excitation = (
            SqueezeExcitation(out_channels) if squeeze_excitation else nn.Identity()
        )
        self.dropout = nn.Dropout2d(dropout)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation2 = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.activation1(self.conv1(x))
        x = self.conv2(x)
        x = self.squeeze_excitation(x)
        return self.activation2(self.dropout(x) + residual)
