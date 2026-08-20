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
