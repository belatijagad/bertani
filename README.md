# Bertani

Fast, reproducible tooling for the [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) reinforcement-learning competition.

Bertani mirrors the advanced Kaggriculture environment in a pure Rust rules engine, then layers a parallel, NumPy-first vector environment on top. Python and serialization stay outside the simulation hot path.

## Current status

- Pure Rust `reset`/`step` engine with typed state and actions
- Dynamic boards, hands, and market-order lists
- Crops, animals, structures, shed behavior, land, hiring, town demand, and all market rules
- CPython-compatible Mersenne Twister for exact weed/shop replay behavior
- Full-state, every-transition differential tests against the installed Python oracle
- Focused tests for high-risk edge cases
- Rayon-parallel vector environment exposed through PyO3
- Reused caller-owned NumPy buffers with player-relative observations and action masks
- Fixed-shape unit and market actions with slot-exact simultaneous-market behavior
- Deterministic auto-reset with retained terminal snapshots

The oracle is pinned to `kaggle-environments==1.32.7`. The exact `kaggriculture.py` used during development has SHA-256:

```text
bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e
```

This matters because the C++ notebook in `references/` targets 1.32.6. Version 1.32.7 changed the scarcity curves for carrot, tomato, and egg to the new `hinge` function, so copying that port's constants produces incorrect rewards.

## Use the Rust core

```rust
use kaggriculture_core::{Action, Config, Sim};

let mut sim = Sim::new(Config {
    seed: 11,
    ..Config::default()
});
let pass = Action::pass();

while !sim.state.done {
    sim.step([&pass, &pass]);
}

assert_eq!(sim.state.step, 719);
assert_eq!(sim.reward(0), 3000.0);
```

## Use the Python vector environment

```python
import numpy as np

from bertani import Item, MarketOp, VecEnv

env = VecEnv(num_envs=256, seed=11)
batch = env.reset()

# Reuse the environment-owned action arrays instead of allocating each step.
unit_actions, market_actions, market_lengths = env.clear_actions()
market_actions[:, 0, 0] = (MarketOp.BUY_SEED, Item.CARROT, 1)
market_lengths[:, 0] = 1

batch = env.step(unit_actions, market_actions, market_lengths)
assert batch.observations.dtype == np.float32
assert batch.rewards.shape == (256, 2)
```

`Batch` arrays and their named views are overwritten by the next `reset` or `step`; copy only values that must survive. The default unit dimension is 231, the exact observable bound implied by the default turn and market-order limits. See [the vector-environment API](docs/vector-environment.md) for layouts, masks, auto-reset semantics, and seeding.

Rule-based policies can use the same batched boundary. `VectorRulePolicy`
extracts features and evaluates strategic rules across every environment with
NumPy, arbitrates typed `TaskBatch` objectives, assigns units, and serializes
masked actions through a reusable executor. Ordered `MarketPlanBatch` objects
track market actions and resource reservations. The intent planner is
replaceable, so a learned planner can later retain deterministic legality,
logistics, and action encoding. See
[the rule-based agent architecture](docs/rule-based-agent.md).

Run validation with:

```bash
uv run cargo test --workspace
uv run cargo clippy --workspace --all-targets -- -D warnings
uv run pytest -q
```

Run the benchmark scopes with:

```bash
cargo run --release -p kaggriculture-core --example benchmark -- 10000
uv run python scripts/benchmark_python.py 3
uv run maturin develop --release
uv run python scripts/benchmark_vec_env.py
```

Run submission-compatible agents on common seeds in both seat orders with:

```bash
uv run python scripts/pit_agents.py \
  baselines/v16_rc5/main.py path/to/rule_agent/main.py \
  --seeds 11 12 13
```

Use the parity-checked native V16 harness for high-throughput rule development:

```bash
# Deterministic panel of 100 generated seeds, both seat orders (200 games).
uv run python scripts/pit_v16_native.py --num-seeds 100 --seed-source 2026

# A named regression panel.
uv run python scripts/pit_v16_native.py --seeds 11 451781128 874717982
```

The native adapter reproduces V16's trace, weed recovery, and market
front-running directly in tensors. Its fixed three-seed results match the
submission-compatible Python runner exactly. Continue using `pit_agents.py`
as the final Kaggle-format check; use `pit_v16_native.py` for broad searches.

Render one full local game and immediately open the standalone replay in the
default browser with:

```bash
uv run python scripts/render_game.py --seed 11
```

Use `--rule-seat 1` to swap seats, `--opponent starter` to use a Kaggle built-in
agent, or `--no-open` when generating a replay non-interactively. Replays are
written under `outputs/replays/` by default.

The preserved V16-RC5 baseline and its provenance are documented in
[`baselines/v16_rc5/README.md`](baselines/v16_rc5/README.md).

Build the current rule-based policy as a portable Kaggle archive with:

```bash
uv run python scripts/package_rule_agent.py
```

This writes `dist/rule_based_submission.tar.gz` with `main.py` at the archive
root. All concrete decisions live in `src/bertani_rules/agent.py`; reusable
planning and execution abstractions remain under `src/bertani`. Improvements
overwrite this single development policy until it beats the preserved baseline.
The bundled observation adapter uses NumPy but does not require the local Rust
extension.

On this development machine, the typed Rust core ran about 4,350 pass/pass episodes/second (0.230 ms/episode), while the full Python Kaggle framework ran about 1.15 episodes/second (872 ms/episode). That ratio is useful for capacity planning but is deliberately not presented as a core-to-core comparison: the Python timing also includes framework, schema, and agent orchestration.

Regenerate the Python reference trace after an intentional oracle update with:

```bash
uv run python scripts/export_reference_trace.py --agents starter,pass --seed 11
```

See [the rules-engine notes](docs/rules-engine.md) for the exact transition order and parity contract. The fixed-shape training boundary is documented in [the vector-environment API](docs/vector-environment.md).

## Layout

```text
crates/kaggriculture-core/  deterministic Rust simulator
crates/bertani-python/      PyO3 vector environment and encoders
scripts/                    Python-oracle fixture generation
baselines/                  immutable submission-compatible opponents
src/bertani/                typed Python wrapper and NumPy buffer views
references/                 local competition references; gitignored
```
