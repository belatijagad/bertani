# Vector-environment API

`bertani.VecEnv` owns a batch of independent Rust simulations. One native call decodes every action, advances the simulations in parallel with Rayon, and overwrites preallocated NumPy observation, mask, reward, and termination buffers.

## Lifetime and reset contract

`reset()` and `step()` return the same `Batch` object and reuse its arrays. A later call overwrites all fields and all named views. This is intentional: a learner can create Torch views once and avoid per-step allocation. Copy episode statistics or terminal data that must outlive the next call.

Rewards are `float64[N, 2]`: zero during an active episode and each player's raw final bank balance on the terminal transition. Reward shaping belongs in the learner wrapper rather than the rules engine.

With `auto_reset=True`, a terminal call returns terminal rewards and `dones=True`, but its observation is already the next episode's initial observation. `episode_ids` likewise identify that new current episode. `terminal_snapshot(i)` retains the terminal state, seed, and previous episode ID; an explicit `reset()` clears it.

With `auto_reset=False`, the terminal call returns the terminal observation and reward once. Further `step()` calls fail until `reset()` is called, which prevents accidental repeated counting of a terminal reward. Changing `auto_reset` after termination does not bypass that requirement.

Constructor seed `s` assigns initial seeds `s + i` (wrapping `u64`) to vector slot `i`. Each auto-reset increments that slot's episode ID and advances its seed by the vector width: `base_seed + episode_id*N`. This keeps automatically assigned seeds disjoint across slots and episodes until integer wrapping. Passing a contiguous `uint64[N]` array to `reset(seeds)` installs new per-slot base seeds and returns all episode IDs to zero; explicitly supplied bases remain the caller's responsibility.

## Actions

All action arrays are C-contiguous `int64`. Unit actions have shape `[N, 2, U, 3]`; the final fields are `(operation, argument, count)`. Slot zero is the farmer and later slots are current farm hands in stable order. Inactive padded slots are ignored.

Unit operation IDs are:

```text
0 PASS       1 NORTH       2 SOUTH       3 EAST
4 WEST       5 PICKUP      6 DROP        7 PLACE
8 PLANT      9 WATER      10 HARVEST    11 FERTILIZE
12 DIG      13 BUILD_COOP 14 BUILD_PASTURE
15 FEED     16 COLLECT_FERTILIZER       17 CARE
```

`PICKUP` and `PLACE` take an item ID `0..11`; `PLANT` takes a crop ID `0..4`. Operations without an argument require zero. Counts pass through as signed `i64`; the rules engine treats invalid or nonpositive state-dependent quantities as no-ops.

Market actions have shape `[N, 2, M, 3]` with IDs:

```text
0 NONE  1 HIRE  2 BUY_LAND  3 BUY_SEED
4 BUY_PRODUCT  5 BUY_ANIMAL  6 SELL
```

Market lengths have shape `[N, 2]` and select the active prefix independently for each player. An internal `NONE` is retained as a no-op slot. This is essential because both players' market orders resolve by slot and share pre-commit quotes; filtering a gap would change game results. The Python wrapper can infer lengths through the last non-`NONE` row, but explicit lengths are preferable when constructing policies.

Item IDs are wheat, carrot, tomato, strawberry, melon, egg, milk, wool, fertilizer, goose, cow, sheep in `0..12`. `BUY_PRODUCT` accepts only wheat or fertilizer, `BUY_ANIMAL` accepts `9..12`, and `SELL` accepts products `0..9`.

The default unit bound is computed as:

```text
transitions = max(1, episode_steps - 1)
U = 1 + min(turns_per_day - 1, transitions) * max_market_orders
```

It is 231 under the official defaults. Hands hired on the end-of-day transition are cleared before the next observation, so they never need action slots. A nonzero `max_units` override smaller than the bound is rejected rather than silently dropping units.

## Observation layout

Observations are player-relative `float32[N, 2, O]`. For each viewer, relative farm zero is self and relative farm one is the opponent. Opponent private and carried inventory channels are zero. `Batch.observation_views` exposes zero-copy shaped views over these contiguous sections:

| View | Shape after `[N, 2]` | Channels |
| --- | --- | --- |
| `global_features` | `[42]` | clock 4, market inventory/price pairs 18, shop counts 8, per-product shop demand 9, event countdowns 3 |
| `farms` | `[2, 9]` | money, farmer coordinates, hands, four land flags, hires |
| `tiles` | `[2, B, B, 24]` | kind 9, crop 5, lifecycle values 10 |
| `units` | `[2, U, 29]` | active/farmer/position/visibility 5, inventory 12, insertion order 12 |
| `private` | `[17]` | own shed 12 and seeds 5 |

Thus `O = 77 + 48*B*B + 58*U`, or 18,275 with default `B=10` and `U=231`.

Continuous values use stable game-scale normalization. Values are not clipped and can exceed one. Coordinates divide by `B-1`; inventory divides by shed capacity; item insertion-order IDs divide by 11 and unused positions are `-1`. Product-demand channels divide the units consumed at the next shop tick by the maximum possible demand of 16. The final three global channels are normalized turns until the next shop tick, town-center tick, and shop-unlock transition; zero means the event follows the current action. The 24 tile channels distinguish empty, locked, weed, plant, empty coop/pasture, and each occupied animal type, then encode crop and lifecycle state. `buffer_specs` is the authoritative source for offsets and channel counts.

## Action masks

`Batch.mask_views` exposes:

| View | Shape after `[N, 2]` |
| --- | --- |
| `unit_ops` | `[U, 18]` |
| `unit_args` | `[U, 18, 12]` |
| `market_ops` | `[7]` |
| `market_args` | `[7, 12]` |

Inactive unit rows allow `PASS` only and are separately marked false in `active_units`. Argument-free operations use argument zero. The flattened mask length is `234*U + 91` (54,145 by default).

Masks mean that one unit of an action can affect the pre-step state. They are policy hints, not validation and not a proof that a complete factorized joint action succeeds. Multiple units may compete for one seed; multiple market orders compete for money and shed space; and the opponent can change shared quotes. Domain-valid masked-off actions still reach the simulator and follow its normal no-op semantics. Malformed operation or argument IDs are rejected before any environment in the batch is mutated.

## Diagnostics and native boundary

`state_snapshot(i)` and `terminal_snapshot(i)` return complete JSON-compatible debug state. They are intentionally outside the hot path. `NativeVecEnv` also exposes `reset_into` and `step_into` for integrations that manage their own buffers; `buffer_specs()` is the source of truth for their shapes and dtypes. Native boolean outputs use `uint8` storage restricted to zero and one, while the supported Python wrapper exposes zero-copy `np.bool_` views. This avoids constructing Rust `bool` references over arbitrary uninitialized NumPy bytes.

The extension exports `RL_API_VERSION = 1`; the Python wrapper refuses a mismatched native module. Increment this value whenever action IDs or buffer semantics change incompatibly.

The current Python constructor exposes the official scalar configuration and uses the pinned default market curves. Custom sparse `marketParams` remain available in the Rust core but are not yet part of the fixed Python training boundary.
