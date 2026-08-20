from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from omegaconf import OmegaConf

from vagen.joint_policy.integration_gate import (
    K4_ID186_CONTINUE_GATE_IMPLEMENTATION,
    JointIntegrationGate,
)
from vagen.main_ppo import _configure_joint_actor_extension


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "vagen/configs/joint_id186_continue.yaml"
TRAIN = ROOT / "examples/train/navigation/train_navigation_joint_id186.yaml"
ID184_TRAIN = ROOT / "examples/train/navigation/train_navigation_joint_id184.yaml"
VAL = ROOT / "examples/train/navigation/val_navigation_joint_id186.yaml"
RUN_NAME = (
    "186_continue_k4schemeb_jointupdate_dp8_tp8_u40_from20_"
    "train3x60_b24_t20_s100_c1_a1_b85p78297006578457_t1_"
    "cot07p095_val5x8"
)


def _env() -> dict[str, str]:
    root = "/project/peilab/atst/nimloth/outputs/experiments"
    return {
        "ID186_TRAIN_CONFIG": str(TRAIN),
        "ID186_VAL_CONFIG": str(VAL),
        "ID186_HEAD_IP": "10.0.0.1",
        "SLURM_JOB_ID": "123",
        "VAGEN_REMOTE_ENV_BASE_URL_OVERRIDE": "http://10.0.0.1:19823",
        "VAGEN_REMOTE_ENV_BASE_URL_OVERRIDE_SCOPE": (
            "id186_exact_continuation_v1"
        ),
        "ID186_ACTOR_MODEL": (
            f"{root}/training/sft2/2026-08-15/"
            "176_id74_action_head_repair_balanced271x8_val40x8/checkpoint"
        ),
        "ID186_PLANNING_CHECKPOINT": (
            f"{root}/vagen_legacy_wm_k16_grid/2026-08-02/sft2/"
            "74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_"
            "ws16n3g844lw844_px100352/train_ws16/epoch_001"
        ),
        "ID186_SOURCE_CHECKPOINT": (
            f"{root}/training/rl/2026-08-17/"
            "184_continue_k4schemeb_jointupdate_dp8_tp8_u20_from10_"
            "train3x60_b24_t20_s100_c1_a1_b85p78297006578457_t1_"
            "cot07p095_val5x8_retry1/checkpoints/global_step_20"
        ),
        "ID186_RUN_OUT": f"/tmp/{RUN_NAME}",
        "ID186_RUN_NAME": RUN_NAME,
        "ID186_AGENT_CONFIG": "/tmp/agent.yaml",
    }


def _config_source():
    source = OmegaConf.load(CONFIG)
    return OmegaConf.merge(
        OmegaConf.create({"actor_rollout_ref": {"actor": {"optim": {}}}}),
        source,
    )


def test_id186_gate_has_two_exact_resume_phases() -> None:
    phase1 = JointIntegrationGate(
        implementation=K4_ID186_CONTINUE_GATE_IMPLEMENTATION,
        experiment_id=186,
        phase="resume_20_to_30",
    )
    phase2 = JointIntegrationGate(
        implementation=K4_ID186_CONTINUE_GATE_IMPLEMENTATION,
        experiment_id=186,
        phase="resume_30_to_40",
    )
    assert phase1.expected_total_training_steps == 30
    assert phase2.expected_total_training_steps == 40
    assert phase1.expected_resume_mode == phase2.expected_resume_mode == "resume_path"
    with pytest.raises(ValueError, match="phase"):
        JointIntegrationGate(
            implementation=K4_ID186_CONTINUE_GATE_IMPLEMENTATION,
            experiment_id=186,
            phase="resume_20_to_40",
        )


def test_id186_phase1_accepts_only_exact_step20_continuation() -> None:
    with patch.dict(os.environ, _env()):
        config = _config_source()
        training = _configure_joint_actor_extension(config)
        assert training.run_seed == 42179
        assert config.trainer.total_training_steps == 30
        assert config.trainer.total_epochs == 30
        assert config.trainer.val_before_train is True
        assert config.trainer.test_freq == 5
        assert config.trainer.save_freq == 5
        assert config.trainer.joint_dataloader_resume_policy == "exact"
        assert config.trainer.resume_from_path.endswith("global_step_20")

        drift = _config_source()
        drift.trainer.joint_dataloader_resume_policy = "reset"
        with pytest.raises(ValueError, match="ID186.*dataloader"):
            _configure_joint_actor_extension(drift)

        drift = _config_source()
        drift.trainer.resume_from_path = "/tmp/global_step_20"
        with pytest.raises(ValueError, match="ID186.*checkpoint"):
            _configure_joint_actor_extension(drift)

    missing_override = _env()
    del missing_override["VAGEN_REMOTE_ENV_BASE_URL_OVERRIDE"]
    with patch.dict(os.environ, missing_override, clear=True):
        with pytest.raises(ValueError, match="ID186.*transport"):
            _configure_joint_actor_extension(_config_source())


def test_id186_phase2_accepts_only_fresh_step30_resume() -> None:
    with patch.dict(os.environ, _env()):
        config = _config_source()
        config.joint_integration_gate.phase = "resume_30_to_40"
        config.trainer.total_training_steps = 40
        config.trainer.total_epochs = 40
        config.trainer.val_before_train = False
        config.trainer.resume_from_path = f"/tmp/{RUN_NAME}/checkpoints/global_step_30"
        _configure_joint_actor_extension(config)

        drift = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        drift.trainer.val_before_train = True
        with pytest.raises(ValueError, match="ID186.*validation"):
            _configure_joint_actor_extension(drift)

        drift = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        drift.trainer.total_training_steps = 30
        with pytest.raises(ValueError, match="phase runtime"):
            _configure_joint_actor_extension(drift)


def test_id186_reuses_the_defined_training_set_and_small_validation() -> None:
    assert yaml.safe_load(TRAIN.read_text()) == yaml.safe_load(ID184_TRAIN.read_text())
    train = yaml.safe_load(TRAIN.read_text())["envs"]
    val = yaml.safe_load(VAL.read_text())["envs"]
    assert len(train) == 3 and all(row["n_envs"] == 60 for row in train)
    assert {row["config"]["eval_set"] for row in train} == {
        "base_train",
        "common_sense_train",
        "long_horizon_train",
    }
    assert len(val) == 5 and all(row["n_envs"] == 8 for row in val)
    assert {row["config"]["eval_set"] for row in val} == {
        "base",
        "common_sense",
        "complex_instruction",
        "visual_appearance",
        "long_horizon",
    }


def test_id186_restore_runtime_is_explicit_for_both_boundaries() -> None:
    source = (ROOT / "vagen/ray_trainer.py").read_text()
    assert "ID186_TRAINING_CONTRACT_PATH_MIGRATION_OK" in source
    assert "ID186_K4_CONTINUE_RESUME_OK" in source
    assert '"resume_20_to_30"' in source
    assert '"resume_30_to_40"' in source
