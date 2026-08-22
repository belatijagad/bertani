"""The repository's current competitive rule strategy.

Its implementation remains in ``bertani_rules.agent`` for compatibility with
existing submissions and tools. This module places it beside copyable custom
strategy examples.
"""

from bertani.kaggle_agent import make_agent
from bertani_rules.agent import IntentPlanner, build_policy

agent = make_agent(build_policy)


__all__ = ["IntentPlanner", "agent", "build_policy"]
