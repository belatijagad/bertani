"""Typed Torch inputs derived from the stable vector-environment layout."""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import numpy as np
import torch
from numpy.typing import NDArray

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
        environments, players = batch.active_units.shape[:2]
        return cls._from_arrays(
            tiles=views.tiles,
            global_features=views.global_features,
            farms=views.farms,
            private=views.private,
            units=views.units[:, :, 0],
            active_units=batch.active_units,
            flat_batch=environments * players,
            device=device,
        )

    @classmethod
    def from_batch_seats(
        cls,
        batch: Batch,
        seats: NDArray[np.int64],
        device: torch.device | str | None = None,
    ) -> TorchObservation:
        """Materialize one selected player perspective per environment."""

        environments = batch.active_units.shape[0]
        if seats.shape != (environments,):
            raise ValueError(f"seats must have shape ({environments},)")
        if np.any((seats < 0) | (seats > 1)):
            raise ValueError("seats must contain only 0 or 1")
        games = np.arange(environments, dtype=np.int64)
        views = batch.observation_views
        return cls._from_arrays(
            tiles=views.tiles[games, seats],
            global_features=views.global_features[games, seats],
            farms=views.farms[games, seats],
            private=views.private[games, seats],
            units=views.units[games, seats, 0],
            active_units=batch.active_units[games, seats],
            flat_batch=environments,
            device=device,
        )

    @classmethod
    def _from_arrays(
        cls,
        *,
        tiles: np.ndarray,
        global_features: np.ndarray,
        farms: np.ndarray,
        private: np.ndarray,
        units: np.ndarray,
        active_units: np.ndarray,
        flat_batch: int,
        device: torch.device | str | None,
    ) -> TorchObservation:
        relative_farms, board, _, tile_channels = tiles.shape[-4:]

        tile_tensor = torch.from_numpy(tiles).reshape(
            flat_batch,
            relative_farms,
            board,
            board,
            tile_channels,
        )
        spatial = tile_tensor.permute(0, 1, 4, 2, 3).flatten(1, 2).contiguous()

        global_tensor = torch.cat(
            (
                torch.from_numpy(global_features).reshape(flat_batch, -1),
                torch.from_numpy(farms).reshape(flat_batch, -1),
                torch.from_numpy(private).reshape(flat_batch, -1),
            ),
            dim=-1,
        )
        workers = torch.from_numpy(units).reshape(
            flat_batch, units.shape[-2], units.shape[-1]
        ).contiguous()

        # Unit channels 2 and 3 are normalized x and y coordinates. Inactive
        # rows may contain placeholder values; force them to the harmless
        # origin before the actor gathers from the spatial map.
        active = torch.from_numpy(active_units).reshape(flat_batch, -1)
        positions = torch.round(workers[..., 2:4] * float(board - 1)).long()
        positions = positions.clamp(0, board - 1)
        positions = torch.where(active.unsqueeze(-1), positions, 0)

        observation = cls(
            spatial=spatial,
            global_features=global_tensor,
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

    @classmethod
    def from_batch_seats(
        cls,
        batch: Batch,
        seats: NDArray[np.int64],
        device: torch.device | str | None = None,
    ) -> TorchActionInfo:
        """Materialize action masks for one selected seat per environment."""

        environments = batch.active_units.shape[0]
        if seats.shape != (environments,):
            raise ValueError(f"seats must have shape ({environments},)")
        if np.any((seats < 0) | (seats > 1)):
            raise ValueError("seats must contain only 0 or 1")
        games = np.arange(environments, dtype=np.int64)
        masks = batch.mask_views
        action_info = cls(
            unit_operation_mask=torch.from_numpy(masks.unit_ops[games, seats]),
            unit_argument_mask=torch.from_numpy(masks.unit_args[games, seats]),
            active_workers=torch.from_numpy(batch.active_units[games, seats]),
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
