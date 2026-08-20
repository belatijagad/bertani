"""Model initialization helpers."""

from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def orthogonal_initialization_(
    module: nn.Module,
    *,
    scale: float = 1.0,
    strict: bool = True,
) -> None:
    """Orthogonally initialize a linear or convolutional layer in place."""

    if not isinstance(module, (nn.Linear, nn.Conv2d)):
        if strict:
            raise TypeError(f"unsupported module type {type(module).__name__}")
        return
    nn.init.orthogonal_(module.weight, gain=scale)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
