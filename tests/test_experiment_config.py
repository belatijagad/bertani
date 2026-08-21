from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bertani.ppo import RewardMode, TerminalScore, load_experiment_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "scripts" / "config"
PPO_14D_CONFIG = CONFIG_DIR / "ppo_14d.yaml"


def test_14_day_ppo_config_loads_the_supported_experiment() -> None:
    config = load_experiment_config(PPO_14D_CONFIG, root=ROOT)

    assert config.max_updates == 300
    assert config.n_envs == 256
    assert config.episode_steps == 14 * 24
    assert config.turns_per_day == 24
    assert config.opponent == "v16"
    assert config.opponent_path == ROOT / "baselines" / "v16_rc5" / "main.py"
    assert config.reward is RewardMode.NET_WORTH_DELTA
    assert config.reward_scale == 10_000
    assert config.terminal_score is TerminalScore.NET_WORTH
    assert config.max_hires_per_turn == 2
    assert config.resume is None
    output = ROOT / "outputs" / "ppo-14d-fast-original"
    assert config.checkpoint_path == output / "latest.pt"
    assert config.metrics_file == output / "metrics.jsonl"
    assert config.ppo.learning_rate == 1e-4
    assert config.ppo.adam_epsilon == 3e-4
    assert config.ppo.max_gradient_norm == 10.0
    assert config.ppo.normalize_advantages
    assert config.ppo.include_workforce
    assert config.ppo.compile_model
    assert config.ppo.compile_mode == "default"
    assert config.ppo.channels_last
    assert config.ppo.fused_optimizer
    assert config.ppo.preload_rollout
    assert config.model.d_model == 64
    assert config.model.n_blocks == 5
    assert config.model.max_hands == 16
    assert config.model.workforce_prior_hands == 5


def test_only_one_ppo_experiment_config_is_kept() -> None:
    assert list(CONFIG_DIR.glob("*.yaml")) == [PPO_14D_CONFIG]


def test_experiment_config_rejects_unknown_fields(tmp_path: Path) -> None:
    with PPO_14D_CONFIG.open(encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file)
    values["typo_field"] = 1
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown config fields: typo_field"):
        load_experiment_config(path, root=ROOT)
