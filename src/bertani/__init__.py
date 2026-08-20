"""Fast Rust tooling for the Kaggriculture reinforcement-learning competition."""

from .market import MarketPlanBatch, MarketRule
from .opening import (
    OpeningController,
    OpeningDiagnostics,
    OpeningTurn,
)
from .rule_based import (
    RuleActions,
    RuleConfig,
    RuleFeatures,
    RulePhase,
    StrategicIntent,
    VectorRulePolicy,
)
from .tasks import (
    TaskAssignments,
    TaskBatch,
    TaskExecutor,
    TaskKind,
    TaskRule,
    TaskScheduler,
    WorkforcePlan,
    WorkforcePlanner,
    WorkRole,
    WorkZone,
)
from .v9_opponent import V9CacheStats, V9OpponentPolicy, V9SelfPlayEnv
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
    "Item",
    "MarketOp",
    "MarketPlanBatch",
    "MarketRule",
    "MaskViews",
    "ObservationViews",
    "OpeningController",
    "OpeningDiagnostics",
    "OpeningTurn",
    "RuleActions",
    "RuleConfig",
    "RuleFeatures",
    "RulePhase",
    "StrategicIntent",
    "TaskAssignments",
    "TaskBatch",
    "TaskExecutor",
    "TaskKind",
    "TaskRule",
    "TaskScheduler",
    "UnitOp",
    "V9CacheStats",
    "V9OpponentPolicy",
    "V9SelfPlayEnv",
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
