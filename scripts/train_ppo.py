#!/usr/bin/env python3
"""Train the baseline actor-critic against frozen v9_main_restarted."""

from __future__ import annotations

import argparse
import cProfile
import json
import math
import resource
import sys
from dataclasses import asdict, replace
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
    load_experiment_config,
)
from bertani.rule_based import RuleConfig
from bertani_rules.agent import build_policy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_FILE = Path(__file__).parent / "config" / "ppo_default.yaml"


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE)
    config_args, _ = config_parser.parse_known_args()
    experiment = load_experiment_config(config_args.config, root=ROOT)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=config_args.config)
    parser.add_argument("--updates", type=int, default=experiment.max_updates)
    parser.add_argument("--num-envs", type=int, default=experiment.n_envs)
    parser.add_argument(
        "--steps-per-update", type=int, default=experiment.ppo.steps_per_update
    )
    parser.add_argument(
        "--epochs-per-update", type=int, default=experiment.ppo.epochs_per_update
    )
    parser.add_argument(
        "--minibatch-size", type=int, default=experiment.ppo.minibatch_size
    )
    parser.add_argument(
        "--learning-rate", type=float, default=experiment.ppo.learning_rate
    )
    parser.add_argument("--device", default=experiment.device)
    parser.add_argument("--seed", type=int, default=experiment.seed)
    parser.add_argument("--v9", type=Path, default=experiment.opponent_path)
    parser.add_argument("--checkpoint", type=Path, default=experiment.checkpoint_path)
    parser.add_argument("--metrics-file", type=Path, default=experiment.metrics_file)
    parser.add_argument("--resume", type=Path, default=experiment.resume)
    parser.add_argument(
        "--checkpoint-every", type=int, default=experiment.checkpoint_every
    )
    parser.add_argument(
        "--reward",
        type=RewardMode,
        choices=list(RewardMode),
        default=experiment.reward,
    )
    parser.add_argument(
        "--market",
        choices=("rule-scaffold", "workforce-only"),
        default=experiment.market,
        help="Use rule buying/selling around learned hires, or train hires alone.",
    )
    parser.add_argument(
        "--max-hires-per-turn",
        type=int,
        default=experiment.max_hires_per_turn,
    )
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=experiment.ppo.mixed_precision,
    )
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=experiment.ppo.profile,
        help="Synchronize CUDA around detailed rollout and optimizer timings.",
    )
    parser.add_argument(
        "--cprofile",
        type=Path,
        default=experiment.cprofile,
        help="Write a Python cProfile data file for the complete run.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=experiment.progress,
    )
    args = parser.parse_args()
    args.ppo_config = replace(
        experiment.ppo,
        steps_per_update=args.steps_per_update,
        epochs_per_update=args.epochs_per_update,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        mixed_precision=args.mixed_precision,
        profile=args.profile,
    )
    args.model_config = experiment.model
    return args


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    trainer: PPOTrainer,
    ppo_config: PPOConfig,
    model_config: ActorCriticConfig,
    experiment_config: dict[str, object],
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
            "experiment_config": experiment_config,
        },
        temporary,
    )
    temporary.replace(path)


def process_peak_rss_mb() -> float:
    # Linux reports KiB; macOS reports bytes. Kaggle and the supported training
    # environment are Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def effective_experiment_config(
    args: argparse.Namespace,
    ppo_config: PPOConfig,
    model_config: ActorCriticConfig,
) -> dict[str, object]:
    """Return the fully resolved settings, including command-line overrides."""

    return {
        "max_updates": args.updates,
        "optimizer_kwargs": {
            "lr": ppo_config.learning_rate,
            "eps": ppo_config.adam_epsilon,
        },
        "clip_gradients": ppo_config.max_gradient_norm,
        "steps_per_update": ppo_config.steps_per_update,
        "epochs_per_update": ppo_config.epochs_per_update,
        "train_batch_size": ppo_config.minibatch_size,
        "use_mixed_precision": ppo_config.mixed_precision,
        "gpu_config": {
            "compile_model": ppo_config.compile_model,
            "compile_mode": ppo_config.compile_mode,
            "channels_last": ppo_config.channels_last,
            "fused_optimizer": ppo_config.fused_optimizer,
            "preload_rollout": ppo_config.preload_rollout,
            "allow_tf32": ppo_config.allow_tf32,
            "cudnn_benchmark": ppo_config.cudnn_benchmark,
        },
        "gamma": ppo_config.gamma,
        "gae_lambda": ppo_config.gae_lambda,
        "clip_coefficient": ppo_config.clip_coefficient,
        "loss_coefficients": {
            "value": ppo_config.value_coefficient,
            "entropy": ppo_config.entropy_coefficient,
        },
        "normalize_advantages": ppo_config.normalize_advantages,
        "include_workforce": ppo_config.include_workforce,
        "env_config": {"n_envs": args.num_envs, "seed": args.seed},
        "self_play_config": {
            "opponent_path": str(args.v9),
            "reward": args.reward.value,
            "market": args.market,
            "max_hires_per_turn": args.max_hires_per_turn,
        },
        "rl_model_config": asdict(model_config),
        "device": args.device,
        "checkpoint_path": str(args.checkpoint),
        "resume": None if args.resume is None else str(args.resume),
        "checkpoint_every": args.checkpoint_every,
        "metrics_file": str(args.metrics_file),
        "progress": args.progress,
        "profile": ppo_config.profile,
        "cprofile": None if args.cprofile is None else str(args.cprofile),
    }


def run(args: argparse.Namespace) -> None:
    if min(args.updates, args.num_envs, args.steps_per_update) <= 0:
        raise ValueError("updates, num-envs, and steps-per-update must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ppo_config: PPOConfig = args.ppo_config
    model_config: ActorCriticConfig = args.model_config
    experiment_config = effective_experiment_config(args, ppo_config, model_config)
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
        economy = build_policy(RuleConfig(), use_opening=True, liquidation_days=1)
    market = WorkforceMarketPolicy(economy, max_hires_per_turn=args.max_hires_per_turn)

    args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    progress_disabled = not args.progress or not sys.stderr.isatty()
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
                args.checkpoint,
                model,
                trainer,
                ppo_config,
                model_config,
                experiment_config,
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
            "rollout/action_composition_seconds": (profile.action_composition_seconds),
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
            **{f"train/{name}": value for name, value in stats.as_dict().items()},
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
    save_checkpoint(
        args.checkpoint,
        model,
        trainer,
        ppo_config,
        model_config,
        experiment_config,
    )


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
