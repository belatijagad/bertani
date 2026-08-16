# Rule-based agent architecture

Rule-based policies are vectorizable. Vectorization means evaluating the same
operations over a batch of environments; it does not require a neural network.
Dense decisions such as phase selection, resource counting, price comparison,
reserve calculation, and action masking fit NumPy well. Ragged decisions such
as assigning a varying number of hands to a varying number of farm tasks may
still use a small per-environment loop after the expensive state evaluation has
been batched.

`bertani.VectorRulePolicy` establishes three boundaries:

```text
VecEnv Batch
    -> batched feature extraction
    -> StrategicIntent (rules now, learned planner later)
    -> masked deterministic executor
    -> VecEnv action tensors
```

`StrategicIntent` contains phase, hiring targets, cash and wheat reserves,
crop/animal targets, and liquidation flags. The default rules identify the
observed three-day opening, a midgame, and the final liquidation window. A
custom callable passed as `intent_planner=` can replace those strategic rules
without replacing action legality and serialization.

The initial executor is intentionally conservative. It performs useful actions
available on each unit's current tile, in this priority order:

1. harvest;
2. feed;
3. water;
4. collect fertilizer;
5. care;
6. drop carried inventory;
7. pass.

It also sells shed products during liquidation. It never assigns a generic
`DIG`, because the action mask allows digging plants as well as weeds and a
context-free priority could destroy a healthy crop.

Use it with the vector environment:

```python
from bertani import VecEnv, VectorRulePolicy

env = VecEnv(num_envs=256, seed=11)
policy = VectorRulePolicy()
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

- a state-based opening controller reproducing the observed day 0–2 position;
- a task map for watering, feeding, care, harvesting, weeds, and structures;
- deterministic unit-to-task assignment and shortest-path movement;
- inventory logistics around the shed;
- market-order construction from reserves, targets, prices, and remaining time;
- batch benchmarks and replay comparisons against the observed opening.

Those layers belong in the executor and rule planner respectively. A future
neural policy should replace high-level intent or task scores first, while the
deterministic executor continues to enforce action shape, masks, and logistics.
