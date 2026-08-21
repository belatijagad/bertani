#!/usr/bin/env python3
"""Train the baseline actor-critic against a frozen batched opponent."""

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

from bertani import SelfPlayEnv, V16OpponentPolicy, V9OpponentPolicy, VecEnv
from bertani.models import ActorCriticConfig, build_actor_critic
from bertani.ppo import (
    CompetitiveReward,
    OpeningWarmStart,
    PPOConfig,
    PPOTrainer,
    RewardMode,
    TerminalScore,
    WorkforceMarketPolicy,
    collect_rollout,
    load_experiment_config,
)
from bertani.rule_based import RuleConfig
from bertani_rules.agent import OPENING_BOOK, build_policy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_FILE = Path(__file__).parent / "config" / "ppo_14d.yaml"


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
        "--episode-steps", type=int, default=experiment.episode_steps
    )
    parser.add_argument(
        "--turns-per-day", type=int, default=experiment.turns_per_day
    )
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
    parser.add_argument(
        "--opponent",
        choices=("v9", "v16"),
        default=experiment.opponent,
    )
    parser.add_argument(
        "--opponent-path",
        type=Path,
        default=experiment.opponent_path,
        help="Frozen opponent submission or trace file.",
    )
    parser.add_argument("--v9", type=Path, default=None, help=argparse.SUPPRESS)
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
        "--reward-scale",
        type=float,
        default=experiment.reward_scale,
        help="Coin scale used by net-worth reward shaping.",
    )
    parser.add_argument(
        "--terminal-score",
        type=TerminalScore,
        choices=list(TerminalScore),
        default=experiment.terminal_score,
        help="Score completed episodes by bank balance or economic net worth.",
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
    if args.v9 is not None:
        args.opponent = "v9"
        args.opponent_path = args.v9
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
        "env_config": {
            "n_envs": args.num_envs,
            "seed": args.seed,
            "episode_steps": args.episode_steps,
            "turns_per_day": args.turns_per_day,
        },
        "self_play_config": {
            "opponent": args.opponent,
            "opponent_path": str(args.opponent_path),
            "reward": args.reward.value,
            "reward_scale": args.reward_scale,
            "terminal_score": args.terminal_score.value,
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
    if min(
        args.updates,
        args.num_envs,
        args.steps_per_update,
        args.episode_steps,
        args.turns_per_day,
    ) <= 0:
        raise ValueError("training and environment sizes must be positive")
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

    environment = VecEnv(
        args.num_envs,
        seed=args.seed,
        max_units=model_config.max_hands + 1,
        auto_reset=True,
        episode_steps=args.episode_steps,
        turns_per_day=args.turns_per_day,
    )
    if args.opponent == "v16":
        opponent = V16OpponentPolicy.from_path(
            args.opponent_path,
            max_orders=environment.max_orders,
        )
        self_play = SelfPlayEnv(environment, opponent)
    else:
        opponent = V9OpponentPolicy.from_path(
            args.opponent_path,
            configuration={
                "episodeSteps": args.episode_steps,
                "turnsPerDay": args.turns_per_day,
                "boardSize": environment.board_size,
                "startingMoney": 3_000,
                "shedCapacity": 100,
                "maxMarketOrdersPerTurn": environment.max_orders,
            },
            max_orders=environment.max_orders,
        )
        self_play = SelfPlayEnv(environment, opponent)
    generator = np.random.default_rng(args.seed)
    seeds = generator.integers(
        0,
        np.iinfo(np.uint32).max,
        size=args.num_envs,
        dtype=np.uint64,
    )
    batch = self_play.reset(seeds)
    reward = CompetitiveReward(
        args.reward,
        reward_scale=args.reward_scale,
        discount=ppo_config.gamma,
        terminal_score=args.terminal_score,
    )
    reward.reset(self_play, batch)

    rule_policy = build_policy(
        RuleConfig(
            episode_steps=args.episode_steps,
            turns_per_day=args.turns_per_day,
        ),
        use_opening=True,
        liquidation_days=1,
    )
    market = WorkforceMarketPolicy(
        rule_policy, max_hires_per_turn=args.max_hires_per_turn
    )
    opening = OpeningWarmStart(
        rule_policy,
        handoff_step=len(OPENING_BOOK),
        episode_steps=args.episode_steps,
    )

    args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    progress_disabled = not args.progress or not sys.stderr.isatty()
    cumulative_games = cumulative_wins = cumulative_ties = cumulative_losses = 0
    cumulative_margin = 0.0
    cumulative_bank_margin = 0.0
    target_update = trainer.updates + args.updates
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
                f"update {update_number}/{target_update}: rollout",
                file=sys.stderr,
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
            opening=opening,
        )
        rollout_progress.close()

        sample_count = ppo_config.steps_per_update * args.num_envs
        minibatches = ppo_config.epochs_per_update * math.ceil(
            sample_count / ppo_config.minibatch_size
        )
        update_progress.set_description(f"update {update_number}: optimize")
        if progress_disabled:
            print(
                f"update {update_number}/{target_update}: optimize",
                file=sys.stderr,
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
        workforce = collection.workforce
        market_stats = collection.market
        profile = collection.profile
        cumulative_games += episodes.completed
        cumulative_wins += episodes.wins
        cumulative_ties += episodes.ties
        cumulative_losses += episodes.losses
        cumulative_margin += episodes.final_margin_sum
        cumulative_bank_margin += episodes.final_bank_margin_sum
        cumulative_win_rate = (
            cumulative_wins / cumulative_games if cumulative_games else 0.0
        )
        cumulative_mean_margin = (
            cumulative_margin / cumulative_games if cumulative_games else 0.0
        )
        cumulative_mean_bank_margin = (
            cumulative_bank_margin / cumulative_games if cumulative_games else 0.0
        )
        batch_margin_text = (
            f"{episodes.mean_final_margin:+,.0f}"
            if episodes.completed
            else "n/a"
        )
        cumulative_margin_text = (
            f"{cumulative_mean_margin:+,.0f}" if cumulative_games else "n/a"
        )
        batch_win_text = (
            f"{episodes.win_rate:.1%}" if episodes.completed else "n/a"
        )
        bank_margin_text = (
            f"{episodes.mean_final_bank_margin:+,.0f}"
            if episodes.completed
            else "n/a"
        )
        cumulative_win_text = (
            f"{cumulative_win_rate:.1%}" if cumulative_games else "n/a"
        )
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
                profile.opponent_seconds,
                profile.action_composition_seconds,
                profile.environment_seconds,
                profile.reward_seconds,
                profile.opening_seconds,
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
            "rollout/opponent_seconds": profile.opponent_seconds,
            "rollout/action_composition_seconds": (profile.action_composition_seconds),
            "rollout/environment_seconds": profile.environment_seconds,
            "rollout/reward_seconds": profile.reward_seconds,
            "rollout/opening_seconds": profile.opening_seconds,
            "rollout/opening_transitions": profile.opening_transitions,
            "rollout/unattributed_seconds": max(
                profile.total_seconds - rollout_attributed, 0.0
            ),
            "rollout/profile_synchronized": profile.synchronized,
            "opponent/cache_hits": profile.opponent_cache_hits,
            "opponent/cache_misses": profile.opponent_cache_misses,
            "episodes/completed": episodes.completed,
            "episodes/wins": episodes.wins,
            "episodes/ties": episodes.ties,
            "episodes/losses": episodes.losses,
            "episodes/win_rate": episodes.win_rate,
            "episodes/mean_final_margin": episodes.mean_final_margin,
            "episodes/terminal_score": args.terminal_score.value,
            "episodes/mean_final_bank_margin": episodes.mean_final_bank_margin,
            "episodes/cumulative_completed": cumulative_games,
            "episodes/cumulative_wins": cumulative_wins,
            "episodes/cumulative_ties": cumulative_ties,
            "episodes/cumulative_losses": cumulative_losses,
            "episodes/cumulative_win_rate": cumulative_win_rate,
            "episodes/cumulative_mean_final_margin": cumulative_mean_margin,
            "episodes/cumulative_mean_final_bank_margin": cumulative_mean_bank_margin,
            "workforce/mean_target_hands": workforce.mean_target_hands,
            "workforce/mean_current_hands": workforce.mean_current_hands,
            "workforce/target_met_rate": workforce.target_met_rate,
            "workforce/hire_orders": workforce.hire_orders,
            "workforce/observed_hires": workforce.observed_hires,
            "market/mean_economic_orders": market_stats.mean_economic_orders,
            "market/mean_buy_orders": market_stats.mean_buy_orders,
            "market/mean_sell_orders": market_stats.mean_sell_orders,
            "market/economic_orders": market_stats.economic_orders,
            "market/buy_orders": market_stats.buy_orders,
            "market/sell_orders": market_stats.sell_orders,
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
        result = (
            f"u{trainer.updates:03d} | "
            f"margin={batch_margin_text} run={cumulative_margin_text} "
            f"bank={bank_margin_text} | "
            f"win={batch_win_text} run={cumulative_win_text} | "
            f"reward={stats.reward_mean:.3g}±{stats.reward_std:.3g} "
            f"KL={stats.approximate_kl:.3g} EV={stats.explained_variance:.2f} | "
            f"hands={workforce.mean_target_hands:.1f} | "
            f"market={market_stats.mean_economic_orders:.1f} "
            f"(buy={market_stats.mean_buy_orders:.1f}, "
            f"sell={market_stats.mean_sell_orders:.1f}) | "
            f"speed={profile.transitions_per_second:.0f}/"
            f"{stats.samples_per_second:.0f}"
        )
        if progress_disabled:
            print(result, file=sys.stderr, flush=True)
        else:
            update_progress.write(result, file=sys.stderr)
        update_progress.set_description(f"update {trainer.updates}: complete")
        update_progress.set_postfix(
            margin=batch_margin_text,
            margin_run=cumulative_margin_text,
            win_run=cumulative_win_text,
            KL=f"{stats.approximate_kl:.2g}",
            market=f"{market_stats.mean_economic_orders:.1f}",
        )
        update_progress.update()

        # Do not carry the completed CPU rollout into the next iteration.
        # Python evaluates the right-hand side of ``collection =
        # collect_rollout(...)`` before replacing the previous collection, so
        # leaving this reference alive makes update N+1 overlap two complete
        # rollout buffers. Large vector batches can otherwise exhaust host RAM
        # while the new rollout is being stacked.
        del collection

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
