"""Reusable vectorized controller for version-supplied opening books."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .vec_env import Batch, UnitOp


ActionRow = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class OpeningTurn:
    """Nominal unit and market rows for one opening turn."""

    units: tuple[ActionRow, ...]
    market: tuple[ActionRow, ...]


@dataclass(frozen=True, slots=True)
class OpeningDiagnostics:
    """Per-seat diagnostics from the most recent controller application."""

    active: NDArray[np.bool_]
    finished: NDArray[np.bool_]
    recovering: NDArray[np.bool_]
    invalid_nominal_action: NDArray[np.bool_]


class OpeningController:
    """Apply the opening book across a batch and repair its known weed branch."""

    def __init__(
        self,
        episode_steps: int,
        book: tuple[OpeningTurn, ...],
        pasture_recovery: tuple[int, int, int] | None = None,
    ) -> None:
        if episode_steps < len(book) + 1:
            raise ValueError(
                f"episode_steps must be at least {len(book) + 1} "
                "to run the complete opening"
            )
        self.episode_steps = episode_steps
        self.book = book
        self.pasture_recovery = pasture_recovery

    def apply(
        self,
        batch: Batch,
        unit_actions: NDArray[np.int64],
        market_actions: NDArray[np.int64],
        market_lengths: NDArray[np.int64],
    ) -> OpeningDiagnostics:
        """Overwrite actions for seats whose current step is in the opening."""

        self._validate_shapes(batch, unit_actions, market_actions, market_lengths)
        step = self.steps(batch)
        active = step < len(self.book)
        finished = ~active
        recovering = np.zeros_like(active)
        invalid = np.zeros_like(active)

        for turn_index, turn in enumerate(self.book):
            environments, players = np.nonzero(active & (step == turn_index))
            if environments.size == 0:
                continue
            unit_count = min(len(turn.units), unit_actions.shape[2])
            market_count = min(len(turn.market), market_actions.shape[2])
            unit_actions[environments, players] = 0
            market_actions[environments, players] = 0
            if unit_count:
                unit_actions[
                    environments[:, None],
                    players[:, None],
                    np.arange(unit_count),
                ] = np.asarray(turn.units[:unit_count], dtype=np.int64)
            if market_count:
                market_actions[
                    environments[:, None],
                    players[:, None],
                    np.arange(market_count),
                ] = np.asarray(turn.market[:market_count], dtype=np.int64)
            market_lengths[environments, players] = market_count
            if unit_count < len(turn.units) or market_count < len(turn.market):
                invalid[environments, players] = True
            represented = batch.active_units[environments, players].sum(axis=-1)
            invalid[environments, players] |= represented != len(turn.units)

        self._repair_pasture(batch, step, active, recovering, unit_actions)
        self._mark_invalid_unit_actions(batch, active, unit_actions, invalid)
        return OpeningDiagnostics(active, finished, recovering, invalid)

    def steps(self, batch: Batch) -> NDArray[np.int64]:
        """Recover integer simulator steps from the normalized batch clock."""

        return np.rint(
            batch.observation_views.global_features[..., 0]
            * (self.episode_steps - 1)
        ).astype(np.int64)

    def active_mask(self, batch: Batch) -> NDArray[np.bool_]:
        """Return seats still controlled by the opening."""

        return self.steps(batch) < len(self.book)

    def _repair_pasture(
        self,
        batch: Batch,
        step: NDArray[np.int64],
        active: NDArray[np.bool_],
        recovering: NDArray[np.bool_],
        unit_actions: NDArray[np.int64],
    ) -> None:
        if self.pasture_recovery is None:
            return
        pasture_x, pasture_y, window_start = self.pasture_recovery
        window = active & (step >= window_start)
        tile = batch.observation_views.tiles[
            :, :, 0, pasture_y, pasture_x
        ]
        weed = window & (tile[..., 2] > 0.5)
        empty = window & (tile[..., 0] > 0.5)
        weed_environments, weed_players = np.nonzero(weed)
        empty_environments, empty_players = np.nonzero(empty)
        unit_actions[weed_environments, weed_players, 0] = (UnitOp.DIG, 0, 0)
        unit_actions[empty_environments, empty_players, 0] = (
            UnitOp.BUILD_PASTURE,
            0,
            0,
        )
        recovering[...] = weed | (empty & (step > window_start))

    @staticmethod
    def _mark_invalid_unit_actions(
        batch: Batch,
        active: NDArray[np.bool_],
        unit_actions: NDArray[np.int64],
        invalid: NDArray[np.bool_],
    ) -> None:
        chosen = unit_actions[..., 0]
        allowed = np.take_along_axis(
            batch.mask_views.unit_ops, chosen[..., None], axis=-1
        )[..., 0]
        relevant = batch.active_units & active[..., None]
        invalid |= ((~allowed) & relevant).any(axis=-1)

    @staticmethod
    def _validate_shapes(
        batch: Batch,
        unit_actions: NDArray[np.int64],
        market_actions: NDArray[np.int64],
        market_lengths: NDArray[np.int64],
    ) -> None:
        n, players, units = batch.active_units.shape
        if unit_actions.shape != (n, players, units, 3):
            raise ValueError("unit action shape does not match the batch")
        if (
            market_actions.shape[:2] != (n, players)
            or market_actions.shape[3:] != (3,)
        ):
            raise ValueError("market action shape does not match the batch")
        if market_lengths.shape != (n, players):
            raise ValueError("market length shape does not match the batch")


__all__ = [
    "OpeningController",
    "OpeningDiagnostics",
    "OpeningTurn",
]
