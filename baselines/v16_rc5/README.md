# V16-RC5 baseline

[`main.py`](main.py) is the executable `V16-RC5-PremiumMarketLead` agent
extracted from `references/v16-rc5-high-score-8c-4s-premium-market-lead.ipynb`.
It is intentionally kept as the notebook emitted it so that it remains a
stable comparison target.

- Source submission reconstructed by the notebook: `55440039`
- Source episodes: `92165990`, `92185587`, `92223213`
- Size: `18,946` bytes
- SHA-256: `f029fa0cb66a9eb509afbe44e3f59b800332d0419db91607183410e4089c4d19`
- Notebook engine: `kaggle-environments==1.32.6`

The agent implements a fixed 720-turn 8-cow/4-sheep route, hand-count
alignment, recovery when a weed blocks a scheduled plant/build action, and a
one-turn market lead for melon, milk, strawberry, and wool.

The module has mutable episode state. Load a fresh module for every seat in
every game; `scripts/pit_agents.py` does this automatically. For example:

```bash
uv run python scripts/pit_agents.py \
  baselines/v16_rc5/main.py path/to/rule_agent/main.py \
  --seeds 11 12 13
```

Each seed is played twice with seats swapped. This controls for seat order and
gives the future rule-based agent a stable, submission-compatible benchmark.

