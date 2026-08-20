"""Typed Torch inputs derived from the stable vector-environment layout."""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import torch

from ..vec_env import Batch


class TorchObservation(NamedTuple):
    """Player-relative model inputs with environment and player axes flattened.

    ``spatial`` contains both farms in relative order: the acting player's 24
    tile channels followed by the opponent's 24 channels. ``workers`` contains
    only the acting player's units because the actor never controls the
    opponent. Worker positions remain in ``(x, y)`` order.
    """

    spatial: torch.Tensor
    """Float tensor shaped ``[batch * players, 48, board, board]``."""
    global_features: torch.Tensor
    """Float tensor shaped ``[batch * players, 65]``."""
    workers: torch.Tensor
    """Float tensor shaped ``[batch * players, max_units, 29]``."""
    worker_positions: torch.Tensor
    """Long tensor shaped ``[batch * players, max_units, 2]``."""

    @classmethod
    def from_batch(
        cls,
        batch: Batch,
        device: torch.device | str | None = None,
    ) -> TorchObservation:
        """Create model inputs from one reusable environment batch.

        The environment owns and overwrites its NumPy buffers. The permutes and
        concatenations here materialize model-ready tensors, so the returned
        observation remains stable when the environment advances.
        """

        views = batch.observation_views
        environments, players, relative_farms, board, _, tile_channels = (
            views.tiles.shape
        )
        flat_batch = environments * players

        tiles = torch.from_numpy(views.tiles).reshape(
            flat_batch,
            relative_farms,
            board,
            board,
            tile_channels,
        )
        spatial = tiles.permute(0, 1, 4, 2, 3).flatten(1, 2).contiguous()

        global_features = torch.cat(
            (
                torch.from_numpy(views.global_features).reshape(flat_batch, -1),
                torch.from_numpy(views.farms).reshape(flat_batch, -1),
                torch.from_numpy(views.private).reshape(flat_batch, -1),
            ),
            dim=-1,
        )
        workers = (
            torch.from_numpy(views.units[:, :, 0])
            .reshape(flat_batch, views.units.shape[-2], views.units.shape[-1])
            .contiguous()
        )

        # Unit channels 2 and 3 are normalized x and y coordinates. Inactive
        # rows may contain placeholder values; force them to the harmless
        # origin before the actor gathers from the spatial map.
        active = torch.from_numpy(batch.active_units).reshape(flat_batch, -1)
        positions = torch.round(workers[..., 2:4] * float(board - 1)).long()
        positions = positions.clamp(0, board - 1)
        positions = torch.where(active.unsqueeze(-1), positions, 0)

        observation = cls(
            spatial=spatial,
            global_features=global_features,
            workers=workers,
            worker_positions=positions,
        )
        return observation if device is None else observation.to_device(device)

    def to_device(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = True,
    ) -> TorchObservation:
        return TorchObservation(
            *(value.to(device, non_blocking=non_blocking) for value in self)
        )

    def index(self, index: torch.Tensor | slice) -> TorchObservation:
        return TorchObservation(*(value[index] for value in self))


class TorchActionInfo(NamedTuple):
    """Masks required to sample legal actions for a padded worker batch."""

    unit_operation_mask: torch.Tensor
    """Boolean tensor shaped ``[batch * players, max_units, 18]``."""
    unit_argument_mask: torch.Tensor
    """Boolean tensor shaped ``[batch * players, max_units, 18, 12]``."""
    active_workers: torch.Tensor
    """Boolean tensor shaped ``[batch * players, max_units]``."""

    @classmethod
    def from_batch(
        cls,
        batch: Batch,
        device: torch.device | str | None = None,
    ) -> TorchActionInfo:
        environments, players, max_units = batch.active_units.shape
        flat_batch = environments * players
        masks = batch.mask_views
        action_info = cls(
            unit_operation_mask=torch.from_numpy(masks.unit_ops).reshape(
                flat_batch, max_units, masks.unit_ops.shape[-1]
            ),
            unit_argument_mask=torch.from_numpy(masks.unit_args).reshape(
                flat_batch,
                max_units,
                masks.unit_args.shape[-2],
                masks.unit_args.shape[-1],
            ),
            active_workers=torch.from_numpy(batch.active_units).reshape(
                flat_batch, max_units
            ),
        )
        return action_info if device is None else action_info.to_device(device)

    def to_device(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = True,
    ) -> TorchActionInfo:
        return TorchActionInfo(
            *(value.to(device, non_blocking=non_blocking) for value in self)
        )

    def index(self, index: torch.Tensor | slice) -> TorchActionInfo:
        return TorchActionInfo(*(value[index] for value in self))


def iter_tensors(
    observation: TorchObservation,
    action_info: TorchActionInfo,
) -> Iterator[torch.Tensor]:
    """Yield every input tensor; useful for device and shape assertions."""

    yield from observation
    yield from action_info
