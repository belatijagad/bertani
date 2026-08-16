# Bertani

Fast, reproducible tooling for the [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) reinforcement-learning competition.

The first milestone is a pure Rust rules engine. It mirrors the advanced Kaggriculture environment while keeping Python and serialization out of the simulation hot path. A Python vector environment and RL-facing observation/action encoders will sit on top of this core after transition parity is locked down.

## Current status

- Pure Rust `reset`/`step` engine with typed state and actions
- Dynamic boards, hands, and market-order lists
- Crops, animals, structures, shed behavior, land, hiring, town demand, and all market rules
- CPython-compatible Mersenne Twister for exact weed/shop replay behavior
- Full-state, every-transition differential tests against the installed Python oracle
- Focused tests for high-risk edge cases

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

Run validation with:

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Run the two benchmark scopes with:

```bash
cargo run --release -p kaggriculture-core --example benchmark -- 10000
uv run python scripts/benchmark_python.py 3
```

On this development machine, the typed Rust core ran about 4,350 pass/pass episodes/second (0.230 ms/episode), while the full Python Kaggle framework ran about 1.15 episodes/second (872 ms/episode). That ratio is useful for capacity planning but is deliberately not presented as a core-to-core comparison: the Python timing also includes framework, schema, and agent orchestration.

Regenerate the Python reference trace after an intentional oracle update with:

```bash
uv run python scripts/export_reference_trace.py --agents starter,pass --seed 11
```

See [the rules-engine notes](docs/rules-engine.md) for the exact transition order, parity contract, and current boundary between the strict Rust core and the future Kaggle/Python compatibility layer.

## Layout

```text
crates/kaggriculture-core/  deterministic Rust simulator
scripts/                    Python-oracle fixture generation
src/bertani/                Python package (RL bindings will live here)
references/                 local competition references; gitignored
```
