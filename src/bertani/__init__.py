"""Fast Rust tooling for the Kaggriculture reinforcement-learning competition."""

from .market import MarketPlanBatch, MarketRule
from .opening import (
    OPENING_BOOK,
    OpeningController,
    OpeningDiagnostics,
    OpeningTurn,
)
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
from .tasks import (
    MaintenanceTaskRule,
    TaskAssignments,
    TaskBatch,
    TaskExecutor,
    TaskKind,
    TaskRule,
    TaskScheduler,
)

__all__ = [
    "Batch",
    "Item",
    "MarketOp",
    "MarketPlanBatch",
    "MarketRule",
    "MaskViews",
    "ObservationViews",
    "OPENING_BOOK",
    "OpeningController",
    "OpeningDiagnostics",
    "OpeningTurn",
    "RuleActions",
    "RuleConfig",
    "RuleFeatures",
    "RulePhase",
    "StrategicIntent",
    "MaintenanceTaskRule",
    "TaskAssignments",
    "TaskBatch",
    "TaskExecutor",
    "TaskKind",
    "TaskRule",
    "TaskScheduler",
    "UnitOp",
    "VecEnv",
    "VectorRulePolicy",
]


def main() -> None:
    """Print a short pointer to the library API."""

    print("bertani: use bertani.VecEnv for batched Kaggriculture simulations")
