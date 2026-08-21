from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bertani._rust", reason="the maturin extension has not been built")

from bertani import UnitOp, VecEnv
from bertani.models import (
    ActorCriticConfig,
    TorchActionInfo,
    TorchObservation,
    build_actor_critic,
)


def _model():
    torch.manual_seed(7)
    return build_actor_critic(
        ActorCriticConfig(
            d_model=32,
            n_blocks=2,
            max_hands=16,
        )
    ).eval()


def test_default_model_stays_near_half_a_million_parameters() -> None:
    model = build_actor_critic()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert parameter_count == 476_348
    assert model.encoder.spatial_input[0].out_channels == 64
    assert len(model.encoder.residual_blocks) == 5


@torch.no_grad()
def test_workforce_prior_starts_near_five_instead_of_uniform_eight() -> None:
    model = build_actor_critic(
        ActorCriticConfig(
            d_model=16,
            n_blocks=1,
            workforce_prior_hands=5,
            workforce_prior_std=2.0,
        )
    )
    encoded = torch.zeros((4, 16, 10, 10))

    log_probs, targets = model.workforce_head(encoded, temperature=0.0)
    choices = torch.arange(17)
    expected = (log_probs.exp() * choices).sum(dim=-1)

    torch.testing.assert_close(targets, torch.full((4,), 5))
    assert torch.all((expected > 4.9) & (expected < 5.1))


def test_batch_adapter_matches_the_vector_environment_layout() -> None:
    env = VecEnv(
        2,
        max_market_orders=2,
        turns_per_day=2,
        weed_spawn_chance=0.0,
    )
    batch = env.reset()

    observation = TorchObservation.from_batch(batch)
    action_info = TorchActionInfo.from_batch(batch)

    assert observation.spatial.shape == (4, 48, 10, 10)
    assert observation.global_features.shape == (4, 77)
    assert observation.workers.shape == (4, env.max_units, 29)
    assert observation.worker_positions.shape == (4, env.max_units, 2)
    assert action_info.unit_operation_mask.shape == (
        4,
        env.max_units,
        18,
    )
    assert action_info.unit_argument_mask.shape == (
        4,
        env.max_units,
        18,
        12,
    )
    assert action_info.active_workers.shape == (4, env.max_units)
    assert observation.worker_positions.dtype == torch.int16
    assert action_info.active_workers.dtype == torch.bool


@torch.no_grad()
def test_actor_samples_only_masked_actions_and_ignores_padding() -> None:
    env = VecEnv(
        2,
        max_market_orders=2,
        turns_per_day=2,
        weed_spawn_chance=0.0,
    )
    batch = env.reset()
    observation = TorchObservation.from_batch(batch)
    action_info = TorchActionInfo.from_batch(batch)

    output = _model()(
        observation,
        action_info,
        worker_temperature=0.0,
        workforce_temperature=0.0,
    )

    flat_batch, max_units = action_info.active_workers.shape
    assert output.operation_log_probs.shape == (flat_batch, max_units, 18)
    assert output.argument_log_probs.shape == (
        flat_batch,
        max_units,
        18,
        12,
    )
    assert output.workforce_log_probs.shape == (flat_batch, 17)
    assert output.target_hands.shape == (flat_batch,)
    assert output.value.shape == (flat_batch,)

    selected_operations_are_legal = action_info.unit_operation_mask.gather(
        -1, output.operations.unsqueeze(-1)
    ).squeeze(-1)
    assert selected_operations_are_legal[action_info.active_workers].all()

    selected_argument_masks = action_info.unit_argument_mask.gather(
        -2,
        output.operations[..., None, None].expand(-1, -1, 1, 12),
    ).squeeze(-2)
    selected_arguments_are_legal = selected_argument_masks.gather(
        -1, output.arguments.unsqueeze(-1)
    ).squeeze(-1)
    assert selected_arguments_are_legal[action_info.active_workers].all()

    inactive = ~action_info.active_workers
    assert (output.operations[inactive] == int(UnitOp.PASS)).all()
    assert (output.arguments[inactive] == 0).all()
    assert torch.isfinite(output.joint_log_probs(action_info.active_workers)).all()

    unit_actions = output.to_unit_actions()
    assert unit_actions.shape == (flat_batch, max_units, 3)
    counted = (output.operations == int(UnitOp.PICKUP)) | (
        output.operations == int(UnitOp.PLACE)
    )
    torch.testing.assert_close(unit_actions[..., 2].bool(), counted)


@torch.no_grad()
def test_one_shared_worker_head_accepts_different_worker_dimensions() -> None:
    model = _model()

    for max_units, active_count in ((3, 1), (13, 9)):
        active = torch.zeros((2, max_units), dtype=torch.bool)
        active[:, :active_count] = True
        operation_mask = torch.zeros((2, max_units, 18), dtype=torch.bool)
        operation_mask[..., int(UnitOp.PASS)] = True
        argument_mask = torch.zeros((2, max_units, 18, 12), dtype=torch.bool)
        argument_mask[..., int(UnitOp.PASS), 0] = True
        observation = TorchObservation(
            spatial=torch.zeros((2, 48, 10, 10)),
            global_features=torch.zeros((2, 77)),
            workers=torch.zeros((2, max_units, 29)),
            worker_positions=torch.zeros((2, max_units, 2), dtype=torch.long),
        )
        action_info = TorchActionInfo(
            unit_operation_mask=operation_mask,
            unit_argument_mask=argument_mask,
            active_workers=active,
        )

        output = model(
            observation,
            action_info,
            worker_temperature=0.0,
            workforce_temperature=0.0,
        )

        assert output.operations.shape == (2, max_units)
        assert (output.operations == int(UnitOp.PASS)).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"d_model": 0}, "d_model must be positive"),
        ({"kernel_size": 2}, "kernel_size must be odd"),
        ({"dropout": 0.3}, "dropout must be between"),
    ],
)
def test_model_config_rejects_invalid_values(
    kwargs: dict[str, int | float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ActorCriticConfig(**kwargs)


def test_batch_adapter_preserves_relative_farm_channel_order() -> None:
    env = VecEnv(
        1,
        max_market_orders=1,
        turns_per_day=2,
        weed_spawn_chance=0.0,
    )
    batch = env.reset()
    batch.observation_views.tiles[0, 0, 0, 2, 3, 4] = 0.25
    batch.observation_views.tiles[0, 0, 1, 2, 3, 4] = 0.75

    observation = TorchObservation.from_batch(batch)

    assert observation.spatial[0, 4, 2, 3].item() == pytest.approx(0.25)
    assert observation.spatial[0, 24 + 4, 2, 3].item() == pytest.approx(0.75)
    assert np.isfinite(observation.global_features.numpy()).all()
