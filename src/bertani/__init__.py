"""Fast Rust tooling for the Kaggriculture reinforcement-learning competition."""

from .vec_env import (
    Batch,
    Item,
    MarketOp,
    MaskViews,
    ObservationViews,
    UnitOp,
    VecEnv,
)
from .rule_based import (
    RuleActions,
    RuleConfig,
    RuleFeatures,
    RulePhase,
    StrategicIntent,
    VectorRulePolicy,
)

__all__ = [
    "Batch",
    "Item",
    "MarketOp",
    "MaskViews",
    "ObservationViews",
    "RuleActions",
    "RuleConfig",
    "RuleFeatures",
    "RulePhase",
    "StrategicIntent",
    "UnitOp",
    "VecEnv",
    "VectorRulePolicy",
]


def main() -> None:
    """Print a short pointer to the library API."""

    print("bertani: use bertani.VecEnv for batched Kaggriculture simulations")
