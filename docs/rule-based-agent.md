# Rule-based agent architecture

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
    -> fused TaskScheduler assignment and masked execution
    -> VecEnv action tensors
```

`StrategicIntent` contains phase, hiring targets, cash and wheat reserves,
crop/animal targets, and liquidation flags. `VectorRulePolicy` supplies a
neutral no-op intent by default. The concrete strategy supplies its opening,
midgame, workforce, production, reserve, and liquidation choices through
`IntentPlanner`. A custom callable passed as `intent_planner=` changes strategy
without replacing action legality and serialization. `build_policy()` accepts
`use_opening=False` to train or evaluate without the opening.

## Strategy boundary

All current decisions are colocated in `src/bertani_rules/agent.py`:

- the 72-turn opening book and its pasture-recovery parameters;
- phase, workforce, reserve, crop, animal, and liquidation targets;
- maintenance, production, logistics, and market rule classes;
- `build_policy()`, which composes the strategy on the reusable engine.

`src/bertani/` contains only reusable representations, feature extraction,
opening-book execution, task arbitration, scheduling, action serialization,
the Kaggle observation adapter, and the vector environment wrapper.

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
priorities differ. Native task proposal kernels fill the extra logistics slots
without inventing fake board tiles.

The native maintenance portion of `FarmTaskRule` proposes work in this
priority order:

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
`TaskScheduler.assign_and_execute()` handles assignment, Manhattan movement,
local action masks, item arguments, and raw tensor serialization in one native
call. A new rule only needs to propose tasks.

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

from bertani_rules.agent import FarmTaskRule

policy = VectorRulePolicy(task_rules=(FarmTaskRule(), BuildPastureRule()))
```

The `intent` argument lets production rules respond to the current strategic
targets without coupling those targets to movement or action encoding.

The production portion of `FarmTaskRule` fills the observed wheat/melon
footprint after harvest, clears weeds that obstruct future production, and
routes units carrying harvests or fertilizer back to the nearest shed access
tile. One-time crops are harvested on their maximum-yield day rather than at
first maturity.

`EconomyMarketRule` sells deposited products when their price is healthy or
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
