"""Competitive rewards for a learner playing a frozen opponent."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import torch

from ..self_play import SelfPlayEnv
from ..vec_env import Batch


class RewardMode(StrEnum):
    NET_WORTH_DELTA = "net_worth_delta"
    MARGIN_DELTA = "margin_delta"
    TERMINAL_MARGIN = "terminal_margin"
    WIN_LOSS = "win_loss"


class TerminalScore(StrEnum):
    BANK = "bank"
    NET_WORTH = "net_worth"


class CompetitiveReward:
    """Convert raw final coins into a stable learner-relative reward."""

    def __init__(
        self,
        mode: RewardMode = RewardMode.MARGIN_DELTA,
        *,
        starting_money: float = 3_000.0,
        reward_scale: float = 10_000.0,
        discount: float = 1.0,
        terminal_score: TerminalScore = TerminalScore.BANK,
    ) -> None:
        if starting_money <= 0:
            raise ValueError("starting_money must be positive")
        if reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be between zero and one")
        self.mode = mode
        self.starting_money = starting_money
        self.reward_scale = reward_scale
        self.discount = discount
        self.terminal_score = terminal_score
        self._margin: np.ndarray | None = None

    def reset(self, self_play: SelfPlayEnv, batch: Batch) -> None:
        self._margin = self._state_margin(self_play, batch)

    def transition(self, self_play: SelfPlayEnv, batch: Batch) -> torch.Tensor:
        if self._margin is None:
            raise RuntimeError("reward must be reset before its first transition")
        games = self_play.games
        learner = self_play.learner_seats
        opponent = self_play.opponent_seats
        dones = batch.dones[games, learner]
        next_margin = self._state_margin(self_play, batch)
        final_margin = self.terminal_margin(self_play, batch) / self.starting_money

        if self.mode == RewardMode.NET_WORTH_DELTA:
            final_margin = final_margin * self.starting_money / self.reward_scale
            reward = np.where(
                dones,
                final_margin - self._margin,
                self.discount * next_margin - self._margin,
            )
        elif self.mode == RewardMode.MARGIN_DELTA:
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

    def terminal_margin(
        self, self_play: SelfPlayEnv, batch: Batch
    ) -> np.ndarray:
        """Return the configured unscaled terminal score difference."""

        games = self_play.games
        learner = self_play.learner_seats
        opponent = self_play.opponent_seats
        values = (
            batch.rewards
            if self.terminal_score == TerminalScore.BANK
            else batch.terminal_economic_values
        )
        return values[games, learner] - values[games, opponent]

    def _state_margin(
        self, self_play: SelfPlayEnv, batch: Batch
    ) -> np.ndarray:
        if self.mode == RewardMode.NET_WORTH_DELTA:
            games = self_play.games
            learner = self_play.learner_seats
            opponent = self_play.opponent_seats
            return (
                batch.economic_values[games, learner]
                - batch.economic_values[games, opponent]
            ).astype(np.float64, copy=True) / self.reward_scale
        return self._observation_margin(self_play, batch)

    @staticmethod
    def _observation_margin(
        self_play: SelfPlayEnv, batch: Batch
    ) -> np.ndarray:
        farms = batch.observation_views.farms
        games = self_play.games
        seats = self_play.learner_seats
        # Farm zero is self and farm one is opponent in each viewer-relative row.
        return (
            farms[games, seats, 0, 0] - farms[games, seats, 1, 0]
        ).astype(np.float64, copy=True)


__all__ = ["CompetitiveReward", "RewardMode", "TerminalScore"]
