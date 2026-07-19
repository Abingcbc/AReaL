# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from omegaconf import OmegaConf

from areal.api.cli_args import (
    GRPOConfig,
    InferenceEngineConfig,
    PPOActorConfig,
    TeacherConfig,
    to_structured_cfg,
)
from areal.infra.rpc.rtensor import RTensor, TensorShardInfo
from areal.trainer.ppo.actor import PPOActor, grpo_loss_fn
from areal.trainer.rl_trainer import _align_rollout_teacher_logps


def _teacher_config(**kwargs) -> TeacherConfig:
    return TeacherConfig(rollout=InferenceEngineConfig(), **kwargs)


def test_teacher_opd_defaults_preserve_joint_loss():
    """Legacy teacher configurations should continue to select joint loss."""
    config = _teacher_config()

    assert config.distillation_mode == "joint_loss"
    assert config.opd_kl_coef == 1.0
    assert config.opd_student_logp_source == "recompute"


def test_teacher_opd_overrides_merge_into_structured_config():
    """Teacher OPD overrides should remain compatible with OmegaConf structured configs."""
    raw_config = OmegaConf.create(
        {
            "teacher": {
                "engine_type": "rollout",
                "rollout": {"backend": "vllm:d1p1t1"},
                "distillation_mode": "kl_penalty",
                "opd_kl_coef": 1.0,
                "opd_student_logp_source": "recompute",
            }
        }
    )

    config = to_structured_cfg(raw_config, GRPOConfig)

    assert config.teacher.distillation_mode == "kl_penalty"
    assert config.teacher.opd_kl_coef == 1.0
    assert config.teacher.opd_student_logp_source == "recompute"


@pytest.mark.parametrize("source", ["recompute", "rollout"])
def test_teacher_opd_accepts_student_logp_sources(source):
    config = _teacher_config(opd_student_logp_source=source)

    assert config.opd_student_logp_source == source


def test_teacher_opd_rejects_unknown_student_logp_source():
    """OPD should reject unknown student logp sources during config parsing."""
    with pytest.raises(ValueError, match="opd_student_logp_source must be"):
        _teacher_config(opd_student_logp_source="unknown")


def test_teacher_opd_kl_penalty_allows_zero_rl_weight():
    """Advantage OPD should support pure distillation without task advantages."""
    config = _teacher_config(distillation_mode="kl_penalty", rl_loss_weight=0.0)

    assert config.rl_loss_weight == 0.0


def test_teacher_rejects_unknown_distillation_mode():
    """Teacher config should reject unsupported distillation modes early."""
    with pytest.raises(ValueError, match="distillation_mode must be"):
        _teacher_config(distillation_mode="unknown")


def test_actor_applies_opd_before_advantage_normalization_and_keeps_returns():
    actor = PPOActor(PPOActorConfig(), engine=None)
    normalized_input = None

    def capture_normalization(advantages, loss_mask, group_sizes=None):
        nonlocal normalized_input
        normalized_input = advantages.clone()
        return advantages

    actor.adv_norm = capture_normalization
    data = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4),
        "loss_mask": torch.tensor([[0.0, 1.0, 1.0, 1.0]]),
        "logprobs": torch.zeros(1, 4),
        "prox_logp": torch.zeros(1, 4),
        "opd_student_logp": torch.tensor([[0.4, 0.3, 0.2, 0.0]]),
        "teacher_logp": torch.tensor([[0.1, 0.1, 0.1, 0.0]]),
        "rewards": torch.tensor([1.0]),
        "opd_mode": "kl_penalty",
        "opd_kl_coef": 2.0,
        "rl_loss_weight": 0.5,
    }

    result = actor._compute_advantages(data)

    assert normalized_input is not None
    expected_reverse_kl = torch.tensor([[0.3, 0.2, 0.1, 0.0]])
    torch.testing.assert_close(
        result["opd_reverse_kl"], expected_reverse_kl, rtol=0.0, atol=1e-6
    )
    torch.testing.assert_close(
        normalized_input,
        0.5 * result["returns"] - 2.0 * expected_reverse_kl,
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        result["advantages"], normalized_input, rtol=0.0, atol=1e-6
    )


@pytest.mark.parametrize(
    ("source", "expected_reverse_kl"),
    [
        ("recompute", torch.tensor([[0.3, 0.2, 0.1, 0.0]])),
        ("rollout", torch.tensor([[0.5, 0.4, 0.3, 0.0]])),
    ],
)
def test_actor_selects_configured_opd_student_logp_source(source, expected_reverse_kl):
    """OPD should independently select recomputed or rollout student logp."""
    actor = PPOActor(PPOActorConfig(recompute_logprob=True), engine=None)
    data = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4),
        "loss_mask": torch.tensor([[0.0, 1.0, 1.0, 1.0]]),
        "logprobs": torch.tensor([[0.0, 0.6, 0.5, 0.4]]),
        "prox_logp": torch.zeros(1, 4),
        "opd_student_logp": torch.tensor([[0.4, 0.3, 0.2, 0.0]]),
        "teacher_logp": torch.tensor([[0.1, 0.1, 0.1, 0.0]]),
        "rewards": torch.tensor([1.0]),
        "opd_mode": "kl_penalty",
        "opd_kl_coef": 1.0,
        "opd_student_logp_source": source,
    }

    result = actor._compute_advantages(data)

    torch.testing.assert_close(
        result["opd_reverse_kl"], expected_reverse_kl, rtol=0.0, atol=1e-6
    )


def test_actor_recompute_opd_source_requires_scored_logp():
    """Recompute OPD should fail when trainer did not attach actor-scored logp."""
    actor = PPOActor(PPOActorConfig(), engine=None)
    data = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3),
        "loss_mask": torch.tensor([[0.0, 1.0, 1.0]]),
        "logprobs": torch.zeros(1, 3),
        "rewards": torch.tensor([1.0]),
        "teacher_logp": torch.zeros(1, 3),
        "opd_mode": "kl_penalty",
        "opd_student_logp_source": "recompute",
    }

    with pytest.raises(ValueError, match="requires actor-scored opd_student_logp"):
        actor._compute_advantages(data)


def test_store_actor_logps_keeps_opd_score_out_of_loglinear_proximal_policy():
    """An OPD-only actor score must not disable log-linear prox approximation."""
    from areal.trainer.rl_trainer import _store_actor_logps

    config = PPOActorConfig(
        use_decoupled_loss=True,
        prox_logp_method="loglinear",
    )
    trajectories = [{}]
    scored_logp = torch.tensor([[0.1, 0.2]])

    _store_actor_logps(
        trajectories,
        [scored_logp],
        store_prox_logp=config.should_compute_prox_logp(),
        store_opd_student_logp=True,
    )

    assert "prox_logp" not in trajectories[0]
    assert trajectories[0]["opd_student_logp"] is scored_logp


def test_rollout_teacher_logps_localize_rtensors_before_alignment():
    """Remote teacher scores must be materialized before tensor operations."""
    score = RTensor(
        shard=TensorShardInfo(shard_id="teacher-score", node_addr="teacher:1234"),
        data=torch.tensor([[0.1, 0.2, 0.3]]),
    )

    aligned, score_refs = _align_rollout_teacher_logps([score])

    assert score_refs[0] is score
    torch.testing.assert_close(
        aligned[0], torch.tensor([[0.2, 0.3, 0.0]]), rtol=0.0, atol=0.0
    )


def test_kl_penalty_loss_does_not_rescale_adjusted_advantages():
    """rl_loss_weight is applied to base advantages, not the combined actor loss."""
    from unittest.mock import patch

    from areal.utils.stats_tracker import DistributedStatsTracker

    logprobs = torch.zeros(1, 2)
    input_data = {
        "input_ids": torch.tensor([[1, 2]]),
        "logprobs": torch.zeros(1, 2),
        "prox_logp": torch.zeros(1, 2),
        "advantages": torch.ones(1, 2),
        "loss_mask": torch.ones(1, 2, dtype=torch.bool),
        "opd_mode": "kl_penalty",
        "rl_loss_weight": 0.25,
    }

    with patch("areal.trainer.ppo.actor.stats_tracker", DistributedStatsTracker()):
        loss = grpo_loss_fn(
            logprobs=logprobs,
            entropy=torch.zeros_like(logprobs),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
        )

    torch.testing.assert_close(loss, torch.tensor(-1.0))
