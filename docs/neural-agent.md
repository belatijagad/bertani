# Neural baseline architecture

`bertani.models` provides a small actor-critic baseline modeled after the
shared-map/shared-entity design used by Frog Parade's Lux AI Season 3 agent.
It is intentionally a model boundary rather than a training framework.

```text
both farms' tile channels [48, B, B] -> spatial projection --+
                                                               +-> residual CNN
global/farm/private channels [77]       -> global projection ---+        |
                                                                        +-> value
                                                                        +-> workforce
                                                                        +-> gather at worker positions
                                                                                |
acting workers [U, 29] ----------------------------------------------------------+
                                                                                |
                                                                        shared worker head
```

The model has one parameter-shared worker head. `U` is only a padded tensor
dimension; inactive slots are suppressed by `active_workers`. The same model
therefore accepts the three-unit buffers of a small test environment, the 13
active units seen in the downloaded leader replays, and the official default
231-slot environment buffer.

The default encoder uses a conventional 64-channel width and five residual
blocks. Together with all actor and critic heads, it has 475,580 trainable
parameters. The worker dimension changes activation memory and compute but not
the parameter count.

The worker actor factorizes each decision into an operation and an argument.
It emits `[batch, U, 18]` operation log-probabilities and
`[batch, U, 18, 12]` conditional argument log-probabilities. Environment masks
are applied before sampling. Multiple workers can still compete for a shared
seed, item, or tile because the simulator masks describe individual legality,
not joint feasibility; a learned task-assignment layer can be placed before
the existing deterministic scheduler later.

The workforce head predicts a target number of hired hands from zero through
`max_hands`. It does not emit `HIRE` market orders. Market order construction
is a separate global decision problem and should be added as its own decoder.

Install the optional Torch dependency and run a forward pass:

```bash
uv sync --extra rl
```

```python
from bertani import VecEnv
from bertani.models import (
    TorchActionInfo,
    TorchObservation,
    build_actor_critic,
)

env = VecEnv(64)
batch = env.reset()
observation = TorchObservation.from_batch(batch, device="cuda")
action_info = TorchActionInfo.from_batch(batch, device="cuda")
model = build_actor_critic().to("cuda")

output = model(observation, action_info)
unit_actions = output.to_unit_actions()
joint_log_probs = output.joint_log_probs(action_info.active_workers)
```

`TorchObservation.from_batch()` materializes model-ready tensors because the
environment overwrites its NumPy buffers on every step. For a high-throughput
trainer, preallocate device staging buffers and copy into them asynchronously
instead of retaining these convenience objects across steps.

Categorical action IDs and worker coordinates are stored and transferred as
`int16`. The model widens only the minibatch values required by PyTorch
indexing to `int64`. The simulator action ABI remains `int64`, because order
quantities are general game-state values rather than small categorical IDs.

## Frozen rollout opponents

PPO defaults to the batched V16-RC5 opponent. Its fixed expert trace, weed
repair, and market front-running run in an independent Rust module over typed
simulator state, making it appropriate for high-throughput pure-RL experiments.
Select it in YAML with
`opponent: v16` and `opponent_path: baselines/v16_rc5/main.py`.

V16 is intentionally isolated from both neural encoding and rule planning:

- `crates/bertani-python/src/v16_opponent.rs` owns native V16 execution/state
- `src/bertani/v16/trace.py` only loads and encodes the preserved trace
- `src/bertani/v16/opponent.py` is the thin Python/native boundary
- `src/bertani/actions.py` owns the policy-neutral action tensor container
- `src/bertani/self_play.py` composes any opponent with the learner

Removing the rule-based package therefore does not require changing V16.

V9 remains available for harder evaluation and later curriculum stages.

Use `V9SelfPlayEnv` to train against `references/v9_main_restarted.py`. The
wrapper alternates learner seats across vector slots and accepts compact
learner-only action tensors; it inserts V9 into the other seat before stepping
the native environment.

```python
import numpy as np

from bertani import V9SelfPlayEnv, VecEnv

environment = VecEnv(128, auto_reset=True)
self_play = V9SelfPlayEnv.from_path(environment)
batch = self_play.reset()

# Learner output only: [environment, worker, (operation, argument, count)].
learner_actions = np.zeros(
    (environment.num_envs, environment.max_units, 3), dtype=np.int64
)
batch = self_play.step(learner_actions)
rewards = self_play.learner_rewards()
```

V9 is a 22 MB replay ensemble whose decoded Python bank is much larger. The
adapter imports that bank once, keeps only its small trajectory state per
environment, restores Python cyclic garbage collection after import, reuses the
native action buffers, and caches identical states within a vector step. The
cache key is generated for the whole batch directly from Rust simulator state
and mirrors V9's feature tuple, so cache hits never construct Python snapshots.
Only one representative of each unique V9 state crosses into the preserved
Python replay selector. Do not instantiate one V9 module per environment or put
`NativeFileAgentPolicy` in the PPO hot loop.

## PPO training

The PPO implementation follows the same separation used in Isaiah Pressman's
Lux agent: rollout collection, typed experience storage, GAE/loss equations,
and minibatch optimization are independent modules under `bertani.ppo`.

Start a training run with:

```bash
uv run python scripts/train_ppo.py
```

The default experiment is defined in
`scripts/config/ppo_default.yaml`, following Isaiah's grouped PPO config
layout. Copy it for an experiment and pass the new file explicitly:

```bash
cp scripts/config/ppo_default.yaml scripts/config/ppo_worker_ablation.yaml
uv run python scripts/train_ppo.py --config scripts/config/ppo_worker_ablation.yaml
```

Command-line options still override the loaded file for quick one-off changes,
for example `--updates 10 --num-envs 8 --device cpu --no-progress`.

The default GPU section enables mixed precision, TF32, channels-last
convolutions, fused Adam, and `torch.compile`. PPO rollout tensors are uploaded
once per update and shuffled/indexed directly on the GPU instead of being
copied once per minibatch. Edit `gpu_config` in the experiment YAML to disable
individual optimizations when comparing hardware or debugging. Compilation is
lazy, so the first rollout and optimizer pass include graph-compilation time;
judge throughput from later updates.

The default `margin_delta` reward is the per-turn change in normalized bank
margin. It telescopes to the final margin without discarding information about
how strongly a game was won or lost. `terminal_margin` and `win_loss` can be
selected with `--reward`.

The current neural baseline has worker and workforce heads but no complete
market decoder. By default, `--market rule-scaffold` retains the rule policy's
buy, sell, land, seed, and animal orders while replacing its hires with the
network's workforce target. This gives worker PPO a functioning economy without
pretending those economic actions were sampled by the neural policy. Use
`--market workforce-only` to remove that scaffold.

Training displays an overall `tqdm` update bar plus nested rollout-step and
optimizer-minibatch bars. The active phase is shown explicitly, with rollout
and training throughput, loss, and recent win rate in the main postfix. Disable
the bars for batch jobs with `--no-progress`.

Every update is also appended as JSON to `outputs/ppo-v16/metrics.jsonl`. Metrics
include:

- Observation packing, device transfer, neural forward, and action transfer
- Rule-market generation, opponent inference, action composition, and Rust stepping
- PPO preparation, device transfer, forward, backward, and optimizer time
- Rollout and optimization throughput
- Win/tie/loss counts, final margin, explained variance, KL, and clip fraction
- Opponent cache deltas, process RSS, and CUDA memory

CUDA launches are asynchronous during normal training. Pass `--profile` to
synchronize at timing boundaries and obtain accurate component timings; this
intentionally reduces throughput. For call-level Python profiling, write a
standard `pstats` file:

```bash
uv run python scripts/train_ppo.py \
  --updates 2 --profile \
  --cprofile outputs/ppo/training.prof

uv run python -m pstats outputs/ppo/training.prof
```

The opponent and environment timers surround the policy/native boundary separately,
which makes it straightforward to tell whether an optimization belongs in the
replay policy adapter, tensor preparation, or native simulator.
