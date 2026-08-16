# Kaggriculture rules engine

## Oracle contract

The implementation targets the advanced environment in `kaggle-environments==1.32.7`, not `kaggriculture_beginner` and not the 1.32.6 implementation used by the local C++ reference notebook.

The test exporter refuses to generate fixtures from another package version or from a source file whose hash differs from the expected hash. Each checked-in trace also records the resolved configuration, seed, package version, source hash, and action/state alignment.

## Transition order

For acting step `t`, the engine performs:

1. Player 0 unit actions, then player 1 unit actions
2. Market orders by slot, with both players quoted from the same pre-commit inventory
3. Town shop and town-center consumption
4. Per-turn crop decay
5. End-of-day refresh when `(t + 1) % turns_per_day == 0`
6. Clock update
7. Terminal check

End-of-day refresh processes plants, animals, weed draws, inventory flush, and unit reset for player 0 and then player 1. A shop unlock draw happens afterward. One freshly seeded CPython-compatible RNG stream is shared by all of those draws for that day.

The default Kaggle episode has 720 recorded states but 719 acting transitions. The last processed action uses old step 718; the resulting step 719 state is terminal.

## Exactness traps covered by the design

- Per-unit inventories preserve Python dictionary insertion order. When the shed reaches capacity, that order determines which carried items survive.
- Plant over-demand validation counts actions submitted for nonexistent hands and cancels every plant request for an over-subscribed crop.
- Locked land is passable. Shed operations also work from a locked central access tile, while ordinary tile operations do not.
- Market quantities resolve one unit at a time and by order-slot index across players. Buy-product quotes use post-buy inventory; sells use pre-sell inventory.
- Sales at the $1 price floor pay the seller but do not add market supply.
- Market prices use Python-compatible ties-to-even rounding and all 1.32.7 curve shapes, including `hinge`.
- Daily randomness reproduces CPython integer seeding, 53-bit `random()`, and `getrandbits` rejection sampling for `choice()`.
- Animal production, care-bank consumption, fertilizer availability, escape, crop production, crop death, and decay retain the Python ordering.

## API boundary

The core accepts typed, valid-domain Rust actions. This keeps Kaggle's permissive JSON parsing and silent malformed-action behavior out of the hot transition function. The Python binding should provide the compatibility adapter and a separate fixed-shape action representation for vectorized RL.

The current numeric boundary is intentionally native Rust rather than arbitrary-precision Python:

- `Config.seed` is an already-resolved, nonnegative `u64`. The future Python adapter should resolve Kaggle's `null` seed before constructing the core.
- Counts, market inventory, and resolved market integer parameters must fit in `i64`; hire-cost calculation uses `i128` with saturation once the value is already far beyond reachable competition money.
- In-game days must fit in `i32`.

The official defaults are many orders of magnitude inside these limits. Deliberately extreme custom market values (for example a base price above `i64::MAX`) are unsupported rather than reproduced with Python big integers; a binding should validate and reject them explicitly.

The intended next layer follows the useful separation in the Orbit Wars reference project:

```text
Python/Torch buffers
        |
PyO3 vector environment and encoders
        |
typed kaggriculture-core reset/step
```

Independent simulations can then be sharded across threads while observations and actions remain in caller-owned contiguous buffers. JSON is reserved for parity fixtures and diagnostics, not training.

## Updating the oracle

An environment upgrade is a rules change until proven otherwise:

1. Pin the new package version in `pyproject.toml`.
2. Review the Python source diff.
3. Update the recorded source hash in the exporter.
4. Regenerate reference traces.
5. Run full-state transition parity and focused rule tests.
6. Benchmark only after parity is restored.
