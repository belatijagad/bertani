from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bertani.ppo import RewardMode, load_experiment_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "scripts" / "config" / "ppo_default.yaml"
HIRE4_CONFIG = ROOT / "scripts" / "config" / "ppo_hire4.yaml"


def test_default_ppo_experiment_config_loads_all_sections() -> None:
    config = load_experiment_config(DEFAULT_CONFIG, root=ROOT)

    assert config.max_updates > 0
    # Experiment-scale knobs are intentionally edited in this file between
    # runs; this test verifies they parse instead of pinning one machine size.
    assert config.n_envs > 0
    assert config.ppo.steps_per_update > 0
    assert config.ppo.minibatch_size > 0
    assert config.reward is RewardMode.MARGIN_DELTA
    assert config.opponent == "v16"
    assert config.opponent_path == ROOT / "baselines" / "v16_rc5" / "main.py"
    assert config.ppo.learning_rate == 1e-4
    assert config.ppo.adam_epsilon == 3e-4
    assert config.ppo.max_gradient_norm == 10.0
    assert config.ppo.compile_model
    assert config.ppo.compile_mode == "default"
    assert config.ppo.channels_last
    assert config.ppo.fused_optimizer
    assert config.ppo.preload_rollout
    assert config.model.global_channels == 77
    assert config.model.d_model == 64


def test_experiment_config_rejects_unknown_fields(tmp_path: Path) -> None:
    with DEFAULT_CONFIG.open(encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file)
    values["typo_field"] = 1
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown config fields: typo_field"):
        load_experiment_config(path, root=ROOT)


def test_hire4_experiment_is_an_isolated_ablation() -> None:
    baseline = load_experiment_config(DEFAULT_CONFIG, root=ROOT)
    experiment = load_experiment_config(HIRE4_CONFIG, root=ROOT)

    assert experiment.max_updates == 100
    assert experiment.max_hires_per_turn == 4
    assert baseline.max_hires_per_turn == 2
    assert experiment.metrics_file != baseline.metrics_file
    assert experiment.checkpoint_path != baseline.checkpoint_path
