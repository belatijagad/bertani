"""Strict YAML configuration for reproducible PPO experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..models import ActorCriticConfig
from .config import PPOConfig
from .rewards import RewardMode


@dataclass(frozen=True, slots=True)
class PPOExperimentConfig:
    max_updates: int
    n_envs: int
    seed: int
    device: str
    opponent_path: Path
    reward: RewardMode
    market: str
    max_hires_per_turn: int
    checkpoint_path: Path
    resume: Path | None
    checkpoint_every: int
    metrics_file: Path
    progress: bool
    cprofile: Path | None
    ppo: PPOConfig
    model: ActorCriticConfig

    def __post_init__(self) -> None:
        positive = {
            "max_updates": self.max_updates,
            "n_envs": self.n_envs,
            "checkpoint_every": self.checkpoint_every,
            "max_hires_per_turn": self.max_hires_per_turn,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.market not in {"rule-scaffold", "workforce-only"}:
            raise ValueError("market must be rule-scaffold or workforce-only")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a YAML mapping")
    return dict(value)


def _exact_fields(mapping: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(mapping.keys() - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {', '.join(unknown)}")
    missing = sorted(allowed - mapping.keys())
    if missing:
        raise ValueError(f"missing {name} fields: {', '.join(missing)}")


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be true or false")
    return value


def _path(value: object, root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _optional_path(value: object, root: Path, name: str) -> Path | None:
    return None if value is None else _path(value, root, name)


def load_experiment_config(
    path: Path,
    *,
    root: Path,
) -> PPOExperimentConfig:
    """Load one Isaiah-style PPO YAML file with strict field validation."""

    with path.open(encoding="utf-8") as config_file:
        raw = _mapping(yaml.safe_load(config_file), "config")

    top_level = {
        "max_updates",
        "optimizer_kwargs",
        "clip_gradients",
        "steps_per_update",
        "epochs_per_update",
        "train_batch_size",
        "use_mixed_precision",
        "gpu_config",
        "gamma",
        "gae_lambda",
        "clip_coefficient",
        "loss_coefficients",
        "normalize_advantages",
        "include_workforce",
        "env_config",
        "self_play_config",
        "rl_model_config",
        "device",
        "checkpoint_path",
        "resume",
        "checkpoint_every",
        "metrics_file",
        "progress",
        "profile",
        "cprofile",
    }
    _exact_fields(raw, top_level, "config")

    optimizer = _mapping(raw["optimizer_kwargs"], "optimizer_kwargs")
    _exact_fields(optimizer, {"lr", "eps"}, "optimizer_kwargs")
    losses = _mapping(raw["loss_coefficients"], "loss_coefficients")
    _exact_fields(losses, {"value", "entropy"}, "loss_coefficients")
    gpu = _mapping(raw["gpu_config"], "gpu_config")
    gpu_fields = {
        "compile_model",
        "compile_mode",
        "channels_last",
        "fused_optimizer",
        "preload_rollout",
        "allow_tf32",
        "cudnn_benchmark",
    }
    _exact_fields(gpu, gpu_fields, "gpu_config")
    environment = _mapping(raw["env_config"], "env_config")
    _exact_fields(environment, {"n_envs", "seed"}, "env_config")
    self_play = _mapping(raw["self_play_config"], "self_play_config")
    _exact_fields(
        self_play,
        {"opponent_path", "reward", "market", "max_hires_per_turn"},
        "self_play_config",
    )
    model_values = _mapping(raw["rl_model_config"], "rl_model_config")
    _exact_fields(
        model_values,
        {"d_model", "n_blocks", "kernel_size", "dropout", "max_hands"},
        "rl_model_config",
    )

    ppo = PPOConfig(
        steps_per_update=int(raw["steps_per_update"]),
        epochs_per_update=int(raw["epochs_per_update"]),
        minibatch_size=int(raw["train_batch_size"]),
        learning_rate=float(optimizer["lr"]),
        adam_epsilon=float(optimizer["eps"]),
        gamma=float(raw["gamma"]),
        gae_lambda=float(raw["gae_lambda"]),
        clip_coefficient=float(raw["clip_coefficient"]),
        value_coefficient=float(losses["value"]),
        entropy_coefficient=float(losses["entropy"]),
        max_gradient_norm=(
            None if raw["clip_gradients"] is None else float(raw["clip_gradients"])
        ),
        normalize_advantages=_boolean(
            raw["normalize_advantages"], "normalize_advantages"
        ),
        include_workforce=_boolean(raw["include_workforce"], "include_workforce"),
        mixed_precision=_boolean(raw["use_mixed_precision"], "use_mixed_precision"),
        compile_model=_boolean(gpu["compile_model"], "gpu_config.compile_model"),
        compile_mode=str(gpu["compile_mode"]),
        channels_last=_boolean(gpu["channels_last"], "gpu_config.channels_last"),
        fused_optimizer=_boolean(gpu["fused_optimizer"], "gpu_config.fused_optimizer"),
        preload_rollout=_boolean(gpu["preload_rollout"], "gpu_config.preload_rollout"),
        allow_tf32=_boolean(gpu["allow_tf32"], "gpu_config.allow_tf32"),
        cudnn_benchmark=_boolean(gpu["cudnn_benchmark"], "gpu_config.cudnn_benchmark"),
        profile=_boolean(raw["profile"], "profile"),
    )
    model = ActorCriticConfig(**model_values)
    return PPOExperimentConfig(
        max_updates=int(raw["max_updates"]),
        n_envs=int(environment["n_envs"]),
        seed=int(environment["seed"]),
        device=str(raw["device"]),
        opponent_path=_path(
            self_play["opponent_path"], root, "self_play_config.opponent_path"
        ),
        reward=RewardMode(str(self_play["reward"])),
        market=str(self_play["market"]),
        max_hires_per_turn=int(self_play["max_hires_per_turn"]),
        checkpoint_path=_path(raw["checkpoint_path"], root, "checkpoint_path"),
        resume=_optional_path(raw["resume"], root, "resume"),
        checkpoint_every=int(raw["checkpoint_every"]),
        metrics_file=_path(raw["metrics_file"], root, "metrics_file"),
        progress=_boolean(raw["progress"], "progress"),
        cprofile=_optional_path(raw["cprofile"], root, "cprofile"),
        ppo=ppo,
        model=model,
    )


__all__ = ["PPOExperimentConfig", "load_experiment_config"]
