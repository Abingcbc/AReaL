# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from areal.api.cli_args import (
    InferenceEngineConfig,
    PPOActorConfig,
    TeacherConfig,
    TeacherOPDConfig,
)
from areal.trainer.ppo.actor import PPOActor


def _teacher_config(**kwargs) -> TeacherConfig:
    return TeacherConfig(rollout=InferenceEngineConfig(), **kwargs)


def test_teacher_opd_defaults_preserve_joint_loss():
    """Legacy teacher configurations should continue to select joint loss."""
    config = _teacher_config()

    assert config.opd.enabled is False
    assert config.opd.mode == "joint_loss"
    assert config.opd.kl_coef == 1.0
    assert config.opd.student_logp_source == "recompute"


@pytest.mark.parametrize("source", ["recompute", "rollout"])
def test_teacher_opd_accepts_student_logp_sources(source):
    """OPD should accept both slime-compatible student logp sources."""
    config = TeacherOPDConfig(student_logp_source=source)

    assert config.student_logp_source == source


def test_teacher_opd_rejects_unknown_student_logp_source():
    """OPD should reject unknown student logp sources during config parsing."""
    with pytest.raises(ValueError, match="student_logp_source must be"):
        TeacherOPDConfig(student_logp_source="unknown")


def test_teacher_opd_kl_penalty_requires_positive_rl_weight():
    """Advantage OPD should reject configurations without an RL objective."""
    opd = TeacherOPDConfig(enabled=True, mode="kl_penalty")

    with pytest.raises(ValueError, match="rl_loss_weight must be positive"):
        _teacher_config(opd=opd, rl_loss_weight=0.0)


def test_teacher_opd_disabled_rejects_advantage_mode():
    """Selecting advantage OPD should require its explicit enable flag."""
    with pytest.raises(ValueError, match="requires teacher.opd.enabled=true"):
        TeacherOPDConfig(enabled=False, mode="kl_penalty")


def test_actor_applies_opd_before_advantage_normalization_and_keeps_returns():
    """Actor advantages should match slime ordering without changing critic returns."""
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
    }

    result = actor._compute_advantages(data)

    assert normalized_input is not None
    expected_reverse_kl = torch.tensor([[0.3, 0.2, 0.1, 0.0]])
    torch.testing.assert_close(
        result["opd_reverse_kl"], expected_reverse_kl, rtol=0.0, atol=1e-6
    )
    torch.testing.assert_close(
        normalized_input,
        result["returns"] - 2.0 * expected_reverse_kl,
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
