#!/usr/bin/env python3
"""Train the baseline actor-critic against frozen v9_main_restarted."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from bertani import V9SelfPlayEnv, VecEnv
from bertani.models import ActorCriticConfig, build_actor_critic
from bertani.ppo import (
    CompetitiveReward,
    PPOConfig,
    PPOTrainer,
    RewardMode,
    WorkforceMarketPolicy,
    collect_rollout,
)
from bertani.rule_based import RuleConfig
from bertani.v9_opponent import DEFAULT_V9_PATH
from bertani_rules.agent import build_policy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "outputs" / "ppo" / "latest.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=1_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps-per-update", type=int, default=32)
    parser.add_argument("--epochs-per-update", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--v9", type=Path, default=DEFAULT_V9_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument(
        "--reward",
        type=RewardMode,
        choices=list(RewardMode),
        default=RewardMode.MARGIN_DELTA,
    )
    parser.add_argument(
        "--market",
        choices=("rule-scaffold", "workforce-only"),
        default="rule-scaffold",
        help="Use rule buying/selling around learned hires, or train hires alone.",
    )
    parser.add_argument("--max-hires-per-turn", type=int, default=2)
    parser.add_argument("--no-mixed-precision", action="store_true")
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    trainer: PPOTrainer,
    ppo_config: PPOConfig,
    model_config: ActorCriticConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "updates": trainer.updates,
            "ppo_config": ppo_config.as_dict(),
            "model_config": asdict(model_config),
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if min(args.updates, args.num_envs, args.steps_per_update) <= 0:
        raise ValueError("updates, num-envs, and steps-per-update must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ppo_config = PPOConfig(
        steps_per_update=args.steps_per_update,
        epochs_per_update=args.epochs_per_update,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        mixed_precision=not args.no_mixed_precision,
    )
    model_config = ActorCriticConfig()
    model = build_actor_critic(model_config).train()
    trainer = PPOTrainer(model, ppo_config, device=device)
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.updates = int(checkpoint["updates"])

    environment = VecEnv(args.num_envs, seed=args.seed, auto_reset=True)
    self_play = V9SelfPlayEnv.from_path(environment, args.v9)
    generator = np.random.default_rng(args.seed)
    seeds = generator.integers(
        0,
        np.iinfo(np.uint32).max,
        size=args.num_envs,
        dtype=np.uint64,
    )
    batch = self_play.reset(seeds)
    reward = CompetitiveReward(args.reward)
    reward.reset(self_play, batch)

    economy = None
    if args.market == "rule-scaffold":
        economy = build_policy(
            RuleConfig(), use_opening=True, liquidation_days=1
        )
    market = WorkforceMarketPolicy(
        economy, max_hires_per_turn=args.max_hires_per_turn
    )

    for _ in range(args.updates):
        started = time.perf_counter()
        rollout = collect_rollout(
            self_play,
            model,
            market,
            reward,
            ppo_config,
            device=device,
        )
        collected = time.perf_counter()
        stats = trainer.update(rollout)
        finished = time.perf_counter()
        metrics = {
            "update": trainer.updates,
            "environment_steps": (
                trainer.updates * ppo_config.steps_per_update * args.num_envs
            ),
            "collect_seconds": collected - started,
            "update_seconds": finished - collected,
            "transitions_per_second": (
                ppo_config.steps_per_update
                * args.num_envs
                / max(collected - started, 1e-9)
            ),
            "v9_cache_hits": self_play.opponent.cache_stats.hits,
            "v9_cache_misses": self_play.opponent.cache_stats.misses,
            **stats.as_dict(),
        }
        print(json.dumps(metrics, sort_keys=True), flush=True)
        if trainer.updates % args.checkpoint_every == 0:
            save_checkpoint(
                args.checkpoint, model, trainer, ppo_config, model_config
            )

    save_checkpoint(args.checkpoint, model, trainer, ppo_config, model_config)


if __name__ == "__main__":
    main()
