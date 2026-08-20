# Neural baseline architecture

`bertani.models` provides a small actor-critic baseline modeled after the
shared-map/shared-entity design used by Frog Parade's Lux AI Season 3 agent.
It is intentionally a model boundary rather than a training framework.

```text
both farms' tile channels [48, B, B] -> spatial projection --+
                                                               +-> residual CNN
global/farm/private channels [65]       -> global projection ---+        |
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

## Frozen V9 rollout opponent

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
native action buffers, and caches identical states within a vector step. Do not
instantiate one V9 module per environment or put `NativeFileAgentPolicy` in the
PPO hot loop.
