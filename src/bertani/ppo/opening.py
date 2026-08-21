"""Rule-opening warm starts kept outside the learned PPO policy."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..actions import ActionBatch
from ..self_play import SelfPlayEnv
from ..vec_env import Batch


class OpeningPolicy(Protocol):
    """Minimal interface required from a batched opening policy."""

    def act(
        self,
        batch: Batch,
        max_orders: int = 10,
        seat_mask: np.ndarray | None = None,
    ) -> ActionBatch: ...


class OpeningWarmStart:
    """Advance reset games to a fixed handoff step using scripted actions."""

    def __init__(
        self,
        policy: OpeningPolicy,
        *,
        handoff_step: int,
        episode_steps: int,
    ) -> None:
        if not 0 < handoff_step < episode_steps:
            raise ValueError("handoff_step must be inside the episode")
        self.policy = policy
        self.handoff_step = handoff_step
        self.episode_steps = episode_steps
        self.seat_mask: np.ndarray | None = None

    def advance(self, self_play: SelfPlayEnv) -> int:
        """Run the opening if the synchronized batch is before the handoff."""

        batch = self_play.batch
        steps = np.rint(
            batch.observation_views.global_features[
                self_play.games, self_play.learner_seats, 0
            ]
            * (self.episode_steps - 1)
        ).astype(np.int64)
        if np.all(steps == self.handoff_step):
            return 0
        if not np.all(steps == steps[0]):
            raise RuntimeError("opening warm start requires synchronized environments")
        if steps[0] > self.handoff_step:
            return 0

        if self.seat_mask is None or self.seat_mask.shape != (
            self_play.environment.num_envs,
            2,
        ):
            self.seat_mask = np.zeros(
                (self_play.environment.num_envs, 2), dtype=np.bool_
            )
            self.seat_mask[self_play.games, self_play.learner_seats] = True

        transitions = 0
        while steps[0] < self.handoff_step:
            actions = self.policy.act(
                self_play.batch,
                max_orders=self_play.environment.max_orders,
                seat_mask=self.seat_mask,
            )
            learner = (self_play.games, self_play.learner_seats)
            self_play.step(
                actions.unit_actions[learner],
                actions.market_actions[learner],
                actions.market_lengths[learner],
            )
            transitions += self_play.environment.num_envs
            steps += 1
        return transitions


__all__ = ["OpeningPolicy", "OpeningWarmStart"]
