"""Vectorized state-based controller for the observed three-day opening."""

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


# Replay steps 1..72 from submission 55463512. Tuple position zero is the
# action emitted from the initial step-0 observation. Unit slot zero is the
# farmer; later slots are farm hands in their stable insertion order.
OPENING_BOOK: tuple[OpeningTurn, ...] = (
    OpeningTurn(((14, 0, 0),), ((1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (5, 10, 2), (5, 11, 2), (3, 0, 7), (3, 4, 12), (4, 0, 6))),  # 0:00
    OpeningTurn(((5, 11, 1), (4, 0, 0), (1, 0, 0), (4, 0, 0), (4, 0, 0), (4, 0, 0)), ((6, 0, 3),)),  # 0:01
    OpeningTurn(((1, 0, 0), (5, 11, 1), (1, 0, 0), (1, 0, 0), (14, 0, 0), (4, 0, 0)), ()),  # 0:02
    OpeningTurn(((14, 0, 0), (1, 0, 0), (1, 0, 0), (5, 10, 1), (1, 0, 0), (1, 0, 0)), ()),  # 0:03
    OpeningTurn(((7, 11, 1), (4, 0, 0), (1, 0, 0), (7, 10, 1), (14, 0, 0), (4, 0, 0)), ()),  # 0:04
    OpeningTurn(((4, 0, 0), (7, 11, 1), (8, 0, 0), (5, 0, 1), (1, 0, 0), (1, 0, 0)), ()),  # 0:05
    OpeningTurn(((3, 0, 0), (4, 0, 0), (9, 0, 0), (1, 0, 0), (8, 4, 0), (8, 0, 0)), ((4, 0, 2),)),  # 0:06
    OpeningTurn(((2, 0, 0), (8, 4, 0), (4, 0, 0), (15, 0, 0), (9, 0, 0), (9, 0, 0)), ()),  # 0:07
    OpeningTurn(((5, 0, 1), (9, 0, 0), (1, 0, 0), (17, 0, 0), (1, 0, 0), (1, 0, 0)), ()),  # 0:08
    OpeningTurn(((4, 0, 0), (4, 0, 0), (8, 0, 0), (2, 0, 0), (8, 4, 0), (1, 0, 0)), ()),  # 0:09
    OpeningTurn(((1, 0, 0), (8, 4, 0), (9, 0, 0), (5, 10, 1), (9, 0, 0), (8, 0, 0)), ()),  # 0:10
    OpeningTurn(((15, 0, 0), (9, 0, 0), (4, 0, 0), (4, 0, 0), (4, 0, 0), (9, 0, 0)), ()),  # 0:11
    OpeningTurn(((17, 0, 0), (1, 0, 0), (4, 0, 0), (7, 10, 1), (4, 0, 0), (4, 0, 0)), ((4, 0, 1),)),  # 0:12
    OpeningTurn(((1, 0, 0), (8, 4, 0), (8, 0, 0), (4, 0, 0), (8, 0, 0), (4, 0, 0)), ()),  # 0:13
    OpeningTurn(((4, 0, 0), (9, 0, 0), (9, 0, 0), (4, 0, 0), (9, 0, 0), (8, 0, 0)), ()),  # 0:14
    OpeningTurn(((1, 0, 0), (4, 0, 0), (3, 0, 0), (8, 4, 0), (4, 0, 0), (9, 0, 0)), ()),  # 0:15
    OpeningTurn(((8, 4, 0), (8, 4, 0), (3, 0, 0), (9, 0, 0), (8, 4, 0), (2, 0, 0)), ()),  # 0:16
    OpeningTurn(((9, 0, 0), (9, 0, 0), (3, 0, 0), (4, 0, 0), (9, 0, 0), (2, 0, 0)), ()),  # 0:17
    OpeningTurn(((0, 0, 0), (2, 0, 0), (8, 4, 0), (8, 4, 0), (0, 0, 0), (0, 0, 0)), ()),  # 0:18
    OpeningTurn(((0, 0, 0), (8, 4, 0), (9, 0, 0), (9, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 0:19
    OpeningTurn(((0, 0, 0), (9, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 0:20
    OpeningTurn(((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 0:21
    OpeningTurn(((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 0:22
    OpeningTurn(((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 0:23
    OpeningTurn(((5, 0, 3),), ()),  # 1:00
    OpeningTurn(((15, 0, 0),), ()),  # 1:01
    OpeningTurn(((4, 0, 0),), ()),  # 1:02
    OpeningTurn(((15, 0, 0),), ()),  # 1:03
    OpeningTurn(((17, 0, 0),), ()),  # 1:04
    OpeningTurn(((1, 0, 0),), ()),  # 1:05
    OpeningTurn(((15, 0, 0),), ()),  # 1:06
    OpeningTurn(((17, 0, 0),), ()),  # 1:07
    OpeningTurn(((3, 0, 0),), ()),  # 1:08
    OpeningTurn(((2, 0, 0),), ()),  # 1:09
    OpeningTurn(((17, 0, 0),), ()),  # 1:10
    OpeningTurn(((16, 0, 0),), ()),  # 1:11
    OpeningTurn(((1, 0, 0),), ()),  # 1:12
    OpeningTurn(((16, 0, 0),), ()),  # 1:13
    OpeningTurn(((4, 0, 0),), ()),  # 1:14
    OpeningTurn(((16, 0, 0),), ()),  # 1:15
    OpeningTurn(((3, 0, 0),), ()),  # 1:16
    OpeningTurn(((2, 0, 0),), ()),  # 1:17
    OpeningTurn(((7, 8, 3),), ((6, 8, 3), (4, 0, 5))),  # 1:18
    OpeningTurn(((5, 0, 1),), ()),  # 1:19
    OpeningTurn(((1, 0, 0),), ()),  # 1:20
    OpeningTurn(((15, 0, 0),), ()),  # 1:21
    OpeningTurn(((17, 0, 0),), ()),  # 1:22
    OpeningTurn(((4, 0, 0),), ()),  # 1:23
    OpeningTurn(((5, 0, 4),), ((1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0))),  # 2:00
    OpeningTurn(((15, 0, 0), (4, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0)), ()),  # 2:01
    OpeningTurn(((17, 0, 0), (4, 0, 0), (4, 0, 0), (1, 0, 0), (16, 0, 0)), ((4, 0, 2),)),  # 2:02
    OpeningTurn(((1, 0, 0), (16, 0, 0), (1, 0, 0), (1, 0, 0), (2, 0, 0)), ()),  # 2:03
    OpeningTurn(((15, 0, 0), (1, 0, 0), (16, 0, 0), (4, 0, 0), (16, 0, 0)), ()),  # 2:04
    OpeningTurn(((17, 0, 0), (1, 0, 0), (4, 0, 0), (1, 0, 0), (4, 0, 0)), ()),  # 2:05
    OpeningTurn(((4, 0, 0), (9, 0, 0), (9, 0, 0), (9, 0, 0), (4, 0, 0)), ()),  # 2:06
    OpeningTurn(((15, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (4, 0, 0)), ()),  # 2:07
    OpeningTurn(((17, 0, 0), (9, 0, 0), (9, 0, 0), (9, 0, 0), (9, 0, 0)), ((4, 0, 2),)),  # 2:08
    OpeningTurn(((2, 0, 0), (1, 0, 0), (1, 0, 0), (4, 0, 0), (1, 0, 0)), ()),  # 2:09
    OpeningTurn(((15, 0, 0), (9, 0, 0), (9, 0, 0), (4, 0, 0), (9, 0, 0)), ()),  # 2:10
    OpeningTurn(((17, 0, 0), (4, 0, 0), (4, 0, 0), (9, 0, 0), (1, 0, 0)), ()),  # 2:11
    OpeningTurn(((4, 0, 0), (4, 0, 0), (9, 0, 0), (4, 0, 0), (9, 0, 0)), ()),  # 2:12
    OpeningTurn(((4, 0, 0), (9, 0, 0), (4, 0, 0), (4, 0, 0), (4, 0, 0)), ()),  # 2:13
    OpeningTurn(((4, 0, 0), (2, 0, 0), (9, 0, 0), (9, 0, 0), (9, 0, 0)), ()),  # 2:14
    OpeningTurn(((9, 0, 0), (2, 0, 0), (0, 0, 0), (0, 0, 0), (2, 0, 0)), ()),  # 2:15
    OpeningTurn(((3, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (9, 0, 0)), ()),  # 2:16
    OpeningTurn(((3, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 2:17
    OpeningTurn(((14, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 2:18
    OpeningTurn(((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 2:19
    OpeningTurn(((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 2:20
    OpeningTurn(((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 2:21
    OpeningTurn(((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 2:22
    OpeningTurn(((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ()),  # 2:23
)


class OpeningController:
    """Apply the opening book across a batch and repair its known weed branch."""

    pasture_x = 2
    pasture_y = 4
    pasture_window_start = 66

    def __init__(self, episode_steps: int = 720) -> None:
        if episode_steps < len(OPENING_BOOK) + 1:
            raise ValueError(
                f"episode_steps must be at least {len(OPENING_BOOK) + 1} "
                "to run the complete opening"
            )
        self.episode_steps = episode_steps

    def apply(
        self,
        batch: Batch,
        unit_actions: NDArray[np.int64],
        market_actions: NDArray[np.int64],
        market_lengths: NDArray[np.int64],
    ) -> OpeningDiagnostics:
        """Overwrite actions for seats whose current step is in the opening."""

        self._validate_shapes(batch, unit_actions, market_actions, market_lengths)
        step = np.rint(
            batch.observation_views.global_features[..., 0]
            * (self.episode_steps - 1)
        ).astype(np.int64)
        active = step < len(OPENING_BOOK)
        finished = ~active
        recovering = np.zeros_like(active)
        invalid = np.zeros_like(active)

        for turn_index, turn in enumerate(OPENING_BOOK):
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

    def _repair_pasture(
        self,
        batch: Batch,
        step: NDArray[np.int64],
        active: NDArray[np.bool_],
        recovering: NDArray[np.bool_],
        unit_actions: NDArray[np.int64],
    ) -> None:
        window = active & (step >= self.pasture_window_start)
        tile = batch.observation_views.tiles[
            :, :, 0, self.pasture_y, self.pasture_x
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
        recovering[...] = weed | (empty & (step > self.pasture_window_start))

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
    "OPENING_BOOK",
    "OpeningController",
    "OpeningDiagnostics",
    "OpeningTurn",
]
