"""Small, copyable Python rule-agent example."""

from __future__ import annotations

import numpy as np

from bertani import Item, RuleConfig, RuleFeatures, VecEnv
from bertani.kaggle_agent import make_agent
from bertani.rule_based import animal_counts_total
from bertani_rules.strategy import RulePlan, build_python_policy


def plan(features: RuleFeatures, targets: RulePlan) -> None:
    """Set economic targets with vectorized NumPy rules."""

    midgame = ~targets.liquidate
    targets.target_hands[midgame] = np.where(features.day[midgame] < 5, 4, 8)
    targets.cash_reserve[midgame] = 200
    targets.wheat_reserve[midgame] = np.maximum(
        4, animal_counts_total(features.animal_counts)[midgame] + 2
    )

    targets.crop(Item.WHEAT)[midgame] = 12
    targets.crop(Item.CARROT)[midgame] = 6
    targets.animal(Item.COW)[midgame] = 4


def build_policy(config: RuleConfig | None = None):
    """Create the native-backed policy used locally and on Kaggle."""

    return build_python_policy(plan, config)


agent = make_agent(build_policy)


def play_local(num_envs: int = 32, seed: int = 11):
    """Run a vector batch and return final rewards for quick experiments."""

    environment = VecEnv(num_envs, seed=seed, auto_reset=False)
    policy = build_policy()
    batch = environment.reset()
    for _ in range(policy.config.episode_steps - 1):
        actions = policy.act(batch, max_orders=environment.max_orders)
        batch = environment.step(
            actions.unit_actions,
            actions.market_actions,
            actions.market_lengths,
        )
    return batch.rewards.copy()


__all__ = ["agent", "build_policy", "plan", "play_local"]
