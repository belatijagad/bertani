"""Competitive rewards for a learner playing frozen V9."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import torch

from ..v9_opponent import V9SelfPlayEnv
from ..vec_env import Batch


class RewardMode(StrEnum):
    MARGIN_DELTA = "margin_delta"
    TERMINAL_MARGIN = "terminal_margin"
    WIN_LOSS = "win_loss"


class CompetitiveReward:
    """Convert raw final coins into a stable learner-relative reward."""

    def __init__(
        self,
        mode: RewardMode = RewardMode.MARGIN_DELTA,
        *,
        starting_money: float = 3_000.0,
    ) -> None:
        if starting_money <= 0:
            raise ValueError("starting_money must be positive")
        self.mode = mode
        self.starting_money = starting_money
        self._margin: np.ndarray | None = None

    def reset(self, self_play: V9SelfPlayEnv, batch: Batch) -> None:
        self._margin = self._observation_margin(self_play, batch)

    def transition(self, self_play: V9SelfPlayEnv, batch: Batch) -> torch.Tensor:
        if self._margin is None:
            raise RuntimeError("reward must be reset before its first transition")
        games = self_play.games
        learner = self_play.learner_seats
        opponent = self_play.opponent_seats
        dones = batch.dones[games, learner]
        next_margin = self._observation_margin(self_play, batch)
        final_margin = (
            batch.rewards[games, learner] - batch.rewards[games, opponent]
        ) / self.starting_money

        if self.mode == RewardMode.MARGIN_DELTA:
            reached_margin = np.where(dones, final_margin, next_margin)
            reward = reached_margin - self._margin
        elif self.mode == RewardMode.TERMINAL_MARGIN:
            reward = np.where(dones, final_margin, 0.0)
        elif self.mode == RewardMode.WIN_LOSS:
            reward = np.where(dones, np.sign(final_margin), 0.0)
        else:  # pragma: no cover - exhaustive StrEnum guard
            raise ValueError(f"unsupported reward mode: {self.mode}")

        # With auto-reset, batch already contains the next episode's initial
        # observation at terminal transitions.
        self._margin = next_margin
        return torch.from_numpy(np.asarray(reward, dtype=np.float32))

    @staticmethod
    def _observation_margin(
        self_play: V9SelfPlayEnv, batch: Batch
    ) -> np.ndarray:
        farms = batch.observation_views.farms
        games = self_play.games
        seats = self_play.learner_seats
        # Farm zero is self and farm one is opponent in each viewer-relative row.
        return (
            farms[games, seats, 0, 0] - farms[games, seats, 1, 0]
        ).astype(np.float64, copy=True)


__all__ = ["CompetitiveReward", "RewardMode"]
