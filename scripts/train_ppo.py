#!/usr/bin/env python3
"""Train the baseline actor-critic against frozen v9_main_restarted."""

from __future__ import annotations

import argparse
import cProfile
import json
import math
import resource
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

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
DEFAULT_METRICS = ROOT / "outputs" / "ppo" / "metrics.jsonl"


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
    parser.add_argument("--metrics-file", type=Path, default=DEFAULT_METRICS)
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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Synchronize CUDA around detailed rollout and optimizer timings.",
    )
    parser.add_argument(
        "--cprofile",
        type=Path,
        help="Write a Python cProfile data file for the complete run.",
    )
    parser.add_argument("--no-progress", action="store_true")
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


def process_peak_rss_mb() -> float:
    # Linux reports KiB; macOS reports bytes. Kaggle and the supported training
    # environment are Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run(args: argparse.Namespace) -> None:
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
        profile=args.profile,
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

    args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    progress_disabled = args.no_progress or not sys.stderr.isatty()
    cumulative_games = cumulative_wins = cumulative_ties = cumulative_losses = 0
    cumulative_margin = 0.0
    update_progress = tqdm(
        total=args.updates,
        desc="PPO",
        unit="update",
        dynamic_ncols=True,
        disable=progress_disabled,
    )
    for _ in range(args.updates):
        update_number = trainer.updates + 1
        update_progress.set_description(f"update {update_number}: rollout")
        if progress_disabled:
            print(
                json.dumps({"update": update_number, "phase": "rollout"}),
                flush=True,
            )
        rollout_progress = tqdm(
            total=ppo_config.steps_per_update,
            desc="  rollout",
            unit="step",
            leave=False,
            dynamic_ncols=True,
            disable=progress_disabled,
        )
        collection = collect_rollout(
            self_play,
            model,
            market,
            reward,
            ppo_config,
            device=device,
            step_callback=rollout_progress.update,
        )
        rollout_progress.close()

        sample_count = ppo_config.steps_per_update * args.num_envs
        minibatches = ppo_config.epochs_per_update * math.ceil(
            sample_count / ppo_config.minibatch_size
        )
        update_progress.set_description(f"update {update_number}: optimize")
        if progress_disabled:
            print(
                json.dumps({"update": update_number, "phase": "optimize"}),
                flush=True,
            )
        optimize_progress = tqdm(
            total=minibatches,
            desc="  optimize",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            disable=progress_disabled,
        )
        stats = trainer.update(
            collection.rollout,
            minibatch_callback=optimize_progress.update,
        )
        optimize_progress.close()

        episodes = collection.episodes
        profile = collection.profile
        cumulative_games += episodes.completed
        cumulative_wins += episodes.wins
        cumulative_ties += episodes.ties
        cumulative_losses += episodes.losses
        cumulative_margin += episodes.final_margin_sum
        checkpoint_saved = trainer.updates % args.checkpoint_every == 0
        if checkpoint_saved:
            save_checkpoint(
                args.checkpoint, model, trainer, ppo_config, model_config
            )

        rollout_attributed = sum(
            (
                profile.observation_seconds,
                profile.device_transfer_seconds,
                profile.policy_forward_seconds,
                profile.action_transfer_seconds,
                profile.market_seconds,
                profile.v9_seconds,
                profile.action_composition_seconds,
                profile.environment_seconds,
                profile.reward_seconds,
            )
        )
        train_attributed = sum(
            (
                stats.prepare_seconds,
                stats.device_transfer_seconds,
                stats.forward_seconds,
                stats.backward_seconds,
                stats.optimizer_seconds,
            )
        )
        metrics = {
            "update": trainer.updates,
            "environment_steps": (
                trainer.updates * ppo_config.steps_per_update * args.num_envs
            ),
            "phase": "complete",
            "checkpoint_saved": checkpoint_saved,
            "rollout/total_seconds": profile.total_seconds,
            "rollout/transitions_per_second": profile.transitions_per_second,
            "rollout/observation_seconds": profile.observation_seconds,
            "rollout/device_transfer_seconds": profile.device_transfer_seconds,
            "rollout/policy_forward_seconds": profile.policy_forward_seconds,
            "rollout/action_transfer_seconds": profile.action_transfer_seconds,
            "rollout/market_seconds": profile.market_seconds,
            "rollout/v9_seconds": profile.v9_seconds,
            "rollout/action_composition_seconds": (
                profile.action_composition_seconds
            ),
            "rollout/environment_seconds": profile.environment_seconds,
            "rollout/reward_seconds": profile.reward_seconds,
            "rollout/unattributed_seconds": max(
                profile.total_seconds - rollout_attributed, 0.0
            ),
            "rollout/profile_synchronized": profile.synchronized,
            "v9/cache_hits": profile.v9_cache_hits,
            "v9/cache_misses": profile.v9_cache_misses,
            "episodes/completed": episodes.completed,
            "episodes/wins": episodes.wins,
            "episodes/ties": episodes.ties,
            "episodes/losses": episodes.losses,
            "episodes/win_rate": episodes.win_rate,
            "episodes/mean_final_margin": episodes.mean_final_margin,
            "episodes/cumulative_completed": cumulative_games,
            "episodes/cumulative_wins": cumulative_wins,
            "episodes/cumulative_ties": cumulative_ties,
            "episodes/cumulative_losses": cumulative_losses,
            "episodes/cumulative_win_rate": (
                cumulative_wins / cumulative_games if cumulative_games else 0.0
            ),
            "episodes/cumulative_mean_final_margin": (
                cumulative_margin / cumulative_games if cumulative_games else 0.0
            ),
            "memory/process_peak_rss_mb": process_peak_rss_mb(),
            "memory/gpu_allocated_mb": (
                torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
                if device.type == "cuda"
                else 0.0
            ),
            "memory/gpu_reserved_mb": (
                torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)
                if device.type == "cuda"
                else 0.0
            ),
            "train/unattributed_seconds": max(
                stats.update_seconds - train_attributed, 0.0
            ),
            **{
                f"train/{name}": value
                for name, value in stats.as_dict().items()
            },
        }
        line = json.dumps(metrics, sort_keys=True)
        with args.metrics_file.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(line + "\n")
        if progress_disabled:
            print(line, flush=True)
        else:
            tqdm.write(line)
        update_progress.set_description(f"update {trainer.updates}: complete")
        update_progress.set_postfix(
            rollout_sps=f"{profile.transitions_per_second:.0f}",
            train_sps=f"{stats.samples_per_second:.0f}",
            loss=f"{stats.total_loss:.3g}",
            win=f"{episodes.win_rate:.1%}" if episodes.completed else "n/a",
        )
        update_progress.update()

    update_progress.close()
    save_checkpoint(args.checkpoint, model, trainer, ppo_config, model_config)


def main() -> None:
    args = parse_args()
    if args.cprofile is None:
        run(args)
        return
    args.cprofile.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    profiler.runcall(run, args)
    profiler.dump_stats(args.cprofile)
    print(f"wrote cProfile data to {args.cprofile}", flush=True)


if __name__ == "__main__":
    main()
