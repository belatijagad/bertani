"""Submission-compatible entry point for the current rule policy."""

from bertani.kaggle_agent import make_agent
from bertani_rules.agent import build_policy


agent = make_agent(build_policy)


__all__ = ["agent"]
