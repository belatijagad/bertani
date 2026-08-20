"""Validated PPO hyperparameters with conservative baseline defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class PPOConfig:
    steps_per_update: int = 32
    epochs_per_update: int = 1
    minibatch_size: int = 128
    learning_rate: float = 1e-4
    adam_epsilon: float = 3e-4
    gamma: float = 0.9999
    gae_lambda: float = 0.85
    clip_coefficient: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 1e-4
    max_gradient_norm: float | None = 10.0
    normalize_advantages: bool = True
    include_workforce: bool = True
    mixed_precision: bool = True

    def __post_init__(self) -> None:
        positive = {
            "steps_per_update": self.steps_per_update,
            "epochs_per_update": self.epochs_per_update,
            "minibatch_size": self.minibatch_size,
            "learning_rate": self.learning_rate,
            "adam_epsilon": self.adam_epsilon,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in {
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if not 0.0 < self.clip_coefficient < 1.0:
            raise ValueError("clip_coefficient must be between zero and one")
        if min(self.value_coefficient, self.entropy_coefficient) < 0.0:
            raise ValueError("loss coefficients cannot be negative")
        if self.max_gradient_norm is not None and self.max_gradient_norm <= 0.0:
            raise ValueError("max_gradient_norm must be positive or None")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = ["PPOConfig"]
