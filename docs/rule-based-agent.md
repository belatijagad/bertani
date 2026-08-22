# Rule-based agent architecture

## Recommended Python interface

Use `bertani_rules.strategy.build_python_policy` for new agents. It gives
Python a compact strategic surface while retaining the native runtime for the
expensive work:

```text
Rust VecEnv observation
    -> Rust feature reduction
    -> one Python plan(features, targets) call per batch
    -> Rust market and farm-task generation
    -> Rust workforce assignment, routing, and action encoding
    -> Rust simulation
```

The callback receives `RuleFeatures`, whose arrays all begin with
`[environment, player]`, and a reusable `RulePlan`. Fill targets with NumPy
masks rather than looping over games:

```python
import numpy as np

from bertani import Item, RuleConfig, RuleFeatures
from bertani_rules.strategy import RulePlan, build_python_policy


def plan(features: RuleFeatures, targets: RulePlan) -> None:
    active = ~targets.liquidate
    targets.target_hands[active] = np.where(features.day[active] < 5, 4, 8)
    targets.cash_reserve[active] = 200
    targets.wheat_reserve[active] = 6
    targets.crop(Item.WHEAT)[active] = 12
    targets.animal(Item.COW)[active] = 4


def build_policy(config: RuleConfig | None = None):
    return build_python_policy(plan, config)
```

Every plan is cleared before the callback, and the final day is marked for
liquidation by default. Available read-only features are `step`, `day`, `hour`,
`money`, crop and animal counts, shed and seed counts, unlocked-shop counts,
opponent crop counts, and market price ratios. Writable targets are phase,
hands, cash and wheat reserves, crop and animal counts, and liquidation.

Put team strategies under `src/bertani_rules/strategies/`. Start from
`simple.py`; `current.py` exposes the existing competitive strategy beside it.
Pass `use_current_opening=True` to `build_python_policy` only when a strategy is
compatible with the current fixed 72-turn opening.

This interface is fast because Python is not called once per environment,
farmer, or farm hand. One callback handles the entire batch, and its arrays are
reused on following turns.

Rule-based policies are vectorizable. Vectorization means evaluating the same
operations over a batch of environments; it does not require a neural network.
Dense decisions such as phase selection, resource counting, price comparison,
reserve calculation, and action masking fit NumPy well. Ragged decisions such
as assigning a varying number of hands to a varying number of farm tasks may
still use a small per-environment loop after the expensive state evaluation has
been batched.

`bertani.VectorRulePolicy` establishes three boundaries. It remains
strategy-free unless task and market rules are passed to it:

```text
VecEnv Batch
    -> batched feature extraction
    -> StrategicIntent (rules now, learned planner later)
    -> TaskRule proposals
    -> TaskScheduler assignments
    -> masked TaskExecutor
    -> VecEnv action tensors
```

`StrategicIntent` contains phase, hiring targets, cash and wheat reserves,
crop/animal targets, and liquidation flags. `VectorRulePolicy` supplies a
neutral no-op intent by default. The current strategy supplies its opening,
midgame, workforce, production, reserve, and liquidation choices through
`IntentPlanner`. A
custom callable passed as `intent_planner=` changes strategy without replacing
action legality and serialization. Version factories may offer
`use_opening=False` to train or evaluate from the initial state.

## Version boundaries

All current strategy decisions are colocated in `src/bertani_rules/agent.py`:

- the 72-turn opening book and its pasture-recovery parameters;
- phase, workforce, reserve, crop, animal, and liquidation targets;
- maintenance, production, logistics, and market rule classes;
- `build_policy()`, which composes the strategy on the reusable engine.

`src/bertani/` contains only reusable representations, feature extraction,
opening-book execution, task arbitration, scheduling, action serialization,
the Kaggle observation adapter, and the vector environment wrapper. Experimental
strategies should remain outside that abstraction package. Preserve competitive
submission snapshots under `baselines/` when a stable comparison point is
needed.

The opening controller owns steps 0–71 (days 0–2). Its nominal action book is
the sequence observed in submission `55463512`, but it inspects the planned
pasture tile `(2, 4)` during the final six turns. If that tile contains a weed,
the farmer digs it and builds the pasture on the following turn. The controller
therefore reconverges to the intended opening position instead of blindly
advancing the tape. `last_opening_diagnostics` exposes per-seat `active`,
`finished`, `recovering`, and `invalid_nominal_action` arrays after every
`act()` call.

Outside the opening, rules no longer emit raw simulator actions. They write
typed objectives into a fixed-capacity `TaskBatch`. The first `B*B` slots map
directly to board tiles and extra slots represent global logistics work. A task
contains:

- `TaskKind` and target coordinates;
- optional item and quantity;
- priority, deadline, and estimated value;
- required inventory item/count;
- whether the task must be assigned exclusively.

Rules may compete for the same tile. `TaskBatch.propose_tiles()` retains the
highest-priority proposal, making arbitration independent of rule ordering when
priorities differ. `TaskBatch.set_global()` adds named logistics tasks without
inventing fake board tiles.

The current strategy in `src/bertani_rules/agent.py` defines a
`MaintenanceTaskRule` that proposes work in this priority order:

1. feed;
2. harvest;
3. water;
4. collect fertilizer;
5. care.

If animals need food and the carried wheat is insufficient, it also proposes a
`FETCH_ITEM(WHEAT)` prerequisite at the shed. The policy replans every turn, so
the inferred workflow naturally progresses from fetch, through movement, to
feed without persisting a brittle imperative script.

`TaskScheduler` computes batched priority-minus-distance scores, checks required
inventory, and then performs a small per-seat conflict-resolution loop. Each
unit receives at most one task and exclusive tasks receive at most one unit.
`TaskExecutor` handles Manhattan movement, local action masks, item arguments,
and raw tensor serialization. This is the main extension seam: a new rule only
needs to propose tasks.

For example:

```python
class BuildPastureRule:
    def propose(self, batch, intent, tasks):
        candidates = ...  # bool[N, 2, B, B]
        tasks.propose_tiles(
            TaskKind.BUILD_PASTURE,
            candidates,
            priority=250.0,
        )

from bertani_rules.agent import MaintenanceTaskRule

policy = VectorRulePolicy(task_rules=(MaintenanceTaskRule(), BuildPastureRule()))
```

The `intent` argument lets production rules respond to the current strategic
targets without coupling those targets to movement or action encoding.

The strategy's `ProductionTaskRule` fills the observed wheat/melon footprint after harvest,
clears weeds that obstruct future production, and routes units carrying
harvests or fertilizer back to the nearest shed access tile. One-time crops are
harvested on their maximum-yield day rather than at first maturity.

The strategy's `EconomyMarketRule` sells deposited products when their price is healthy or
the shed is under pressure, protects the livestock wheat reserve, buys missing
feed and replacement seeds, and hires toward the daily workforce target. The
executor sells every remaining shed product during liquidation.

Market rules use a separate `MarketPlanBatch`. `append()` preserves order and
the active-prefix length required by simultaneous market processing. Rules can
reserve cash or shed items before later rules append purchases or sales, and an
overflow flag records when proposals exceed the configured order limit. This
keeps order sequencing and shared-resource arbitration out of raw action-array
code.

```python
class OpeningHireRule:
    def propose(self, batch, intent, plan):
        selected = intent.phase == RulePhase.OPENING
        plan.reserve_cash(selected, 100)
        plan.append(selected, MarketOp.HIRE)

policy = VectorRulePolicy(market_rules=(OpeningHireRule(),))
```

Use it with the vector environment:

```python
from bertani import VecEnv
from bertani_rules.agent import build_policy

env = VecEnv(num_envs=256, seed=11)
policy = build_policy()  # Opening controller enabled by default.
batch = env.reset()

while True:
    actions = policy.act(batch, max_orders=env.max_orders)
    batch = env.step(
        actions.unit_actions,
        actions.market_actions,
        actions.market_lengths,
    )
```

The action buffers are allocated once per batch shape and reused. `act()` must
therefore be called again before each environment step, and callers must copy
actions they need to retain.

## Planned extensions

The scaffold is not yet a competitive baseline. The next layers are:

- livestock and land expansion rules driven by `StrategicIntent`;
- animal placement and crop fertilization workflows;
- deadline-aware scheduling when the current workforce cannot finish all work;
- joint cash/item reservations across field tasks and market plans;
- market-order construction from reserves, targets, prices, and remaining time;
- batch benchmarks and replay comparisons against the observed opening.

Those layers belong in the executor and rule planner respectively. A future
neural policy should replace high-level intent or task scores first, while the
deterministic executor continues to enforce action shape, masks, and logistics.
