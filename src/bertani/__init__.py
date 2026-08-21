"""Fast Rust tooling for the Kaggriculture reinforcement-learning competition."""

from .actions import ActionBatch
from .market import MarketPlanBatch, MarketRule
from .opening import (
    OpeningController,
    OpeningDiagnostics,
    OpeningTurn,
)
from .rule_based import (
    RuleConfig,
    RuleFeatures,
    RulePhase,
    StrategicIntent,
    VectorRulePolicy,
)
from .self_play import SelfPlayEnv
from .tasks import (
    TaskAssignments,
    TaskBatch,
    TaskKind,
    TaskRule,
    TaskScheduler,
    WorkforcePlan,
    WorkforcePlanner,
    WorkRole,
    WorkZone,
)
from .v16 import V16CacheStats, V16OpponentPolicy
from .v9_opponent import V9CacheStats, V9OpponentPolicy
from .vec_env import (
    Batch,
    Item,
    MarketOp,
    MaskViews,
    ObservationViews,
    UnitOp,
    VecEnv,
)

__all__ = [
    "Batch",
    "ActionBatch",
    "Item",
    "MarketOp",
    "MarketPlanBatch",
    "MarketRule",
    "MaskViews",
    "ObservationViews",
    "OpeningController",
    "OpeningDiagnostics",
    "OpeningTurn",
    "RuleConfig",
    "RuleFeatures",
    "RulePhase",
    "StrategicIntent",
    "TaskAssignments",
    "TaskBatch",
    "TaskKind",
    "TaskRule",
    "TaskScheduler",
    "UnitOp",
    "SelfPlayEnv",
    "V16CacheStats",
    "V16OpponentPolicy",
    "V9CacheStats",
    "V9OpponentPolicy",
    "VecEnv",
    "VectorRulePolicy",
    "WorkRole",
    "WorkZone",
    "WorkforcePlan",
    "WorkforcePlanner",
]


def main() -> None:
    """Print a short pointer to the library API."""

    print("bertani: use bertani.VecEnv for batched Kaggriculture simulations")
