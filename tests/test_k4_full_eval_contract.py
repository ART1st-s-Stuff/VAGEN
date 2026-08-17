from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from omegaconf import OmegaConf

from vagen.joint_policy.integration_gate import (
    K4_ID185_FULL_EVAL_GATE_IMPLEMENTATION,
    JointIntegrationGate,
)
from vagen.main_ppo import _configure_joint_actor_extension


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "vagen/configs/joint_id185_full_eval.yaml"
TRAIN = ROOT / "examples/train/navigation/train_navigation_joint_id185.yaml"
VAL = ROOT / "examples/train/navigation/val_navigation_joint_id185.yaml"


def _env() -> dict[str, str]:
    root = "/project/peilab/atst/nimloth/outputs/experiments"
    return {
        "ID185_TRAIN_CONFIG": "/tmp/train_navigation_joint_id185.yaml",
        "ID185_VAL_CONFIG": "/tmp/val_navigation_joint_id185.yaml",
        "ID185_ACTOR_MODEL": (
            f"{root}/training/sft2/2026-08-15/"
            "176_id74_action_head_repair_balanced271x8_val40x8/checkpoint"
        ),
        "ID185_PLANNING_CHECKPOINT": (
            f"{root}/vagen_legacy_wm_k16_grid/2026-08-02/sft2/"
            "74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_"
            "ws16n3g844lw844_px100352/train_ws16/epoch_001"
        ),
        "ID185_SOURCE_CHECKPOINT": (
            f"{root}/training/rl/2026-08-17/"
            "184_continue_k4schemeb_jointupdate_dp8_tp8_u20_from10_"
            "train3x60_b24_t20_s100_c1_a1_b85p78297006578457_t1_"
            "cot07p095_val5x8_retry1/checkpoints/global_step_20"
        ),
        "ID185_RUN_OUT": "/tmp/185_full_eval",
        "ID185_RUN_NAME": (
            "185_eval_k4schemeb_dp8_tp8_source20_test5x60_"
            "t20_s100_c1_a1_b85p78297006578457_t1_cot07p095"
        ),
        "ID185_AGENT_CONFIG": "/tmp/agent.yaml",
    }


def _config_source():
    source = OmegaConf.load(CONFIG)
    return OmegaConf.merge(
        OmegaConf.create({"actor_rollout_ref": {"actor": {"optim": {}}}}),
        source,
    )


def test_id185_gate_is_eval_only_from_complete_step20() -> None:
    gate = JointIntegrationGate(
        implementation=K4_ID185_FULL_EVAL_GATE_IMPLEMENTATION,
        experiment_id=185,
        phase="full_eval_test300",
    )
    assert gate.expected_total_training_steps == 20
    assert gate.expected_resume_mode == "resume_path"
    with pytest.raises(ValueError, match="phase"):
        JointIntegrationGate(
            implementation=K4_ID185_FULL_EVAL_GATE_IMPLEMENTATION,
            experiment_id=185,
            phase="train",
        )


def test_id185_config_accepts_only_exact_val_only_restore() -> None:
    with patch.dict(os.environ, _env()):
        config = _config_source()
        training = _configure_joint_actor_extension(config)
        assert training.run_seed == 42179
        assert config.trainer.total_training_steps == 20
        assert config.trainer.total_epochs == 20
        assert config.trainer.resume_mode == "resume_path"
        assert config.trainer.resume_from_path.endswith("global_step_20")
        assert config.trainer.joint_dataloader_resume_policy == "exact"
        assert config.trainer.val_before_train is True
        assert config.trainer.val_only is True
        assert config.trainer.test_freq == -1
        assert config.trainer.save_freq == 5
        assert config.data.val_batch_size == 40
        assert config.trainer.nnodes == 4
        assert config.trainer.n_gpus_per_node == 2

        drift = _config_source()
        drift.trainer.val_only = False
        with pytest.raises(ValueError, match="ID185.*validation"):
            _configure_joint_actor_extension(drift)

        drift = _config_source()
        drift.trainer.resume_from_path = "/tmp/global_step_20"
        with pytest.raises(ValueError, match="ID185.*checkpoint"):
            _configure_joint_actor_extension(drift)

        drift = _config_source()
        drift.trainer.joint_dataloader_resume_policy = "reset"
        with pytest.raises(ValueError, match="ID185.*dataloader"):
            _configure_joint_actor_extension(drift)


def test_id185_restore_migrates_transport_without_training() -> None:
    source = (ROOT / "vagen/ray_trainer.py").read_text()
    assert "ID185_TRAINING_CONTRACT_PATH_MIGRATION_OK" in source
    assert "ID185_K4_FULL_EVAL_RESTORE_OK global_step=20" in source
    assert "experiment_id in {183, 184, 185}" in source


def test_id185_uses_all_historical_vagen_test_tasks() -> None:
    train = yaml.safe_load(TRAIN.read_text())["envs"]
    val = yaml.safe_load(VAL.read_text())["envs"]
    assert len(train) == 3
    assert len(val) == 5
    assert {row["config"]["eval_set"] for row in val} == {
        "base",
        "common_sense",
        "long_horizon",
        "complex_instruction",
        "visual_appearance",
    }
    for row in val:
        assert row["n_envs"] == 60
        assert row["seed_list"][:60] == list(range(1, 61))
        assert len(row["seed_list"]) > row["n_envs"]
        assert row["max_turns"] == 20
        assert row["config"]["prompt_format"] == "nimloth"
        assert row["config"]["latent_token_count"] == 16
        assert row["config"]["per_turn_format_reward"] == 0.01
        assert row["config"]["format_reward"] == 0.0
        assert row["config"]["success_reward"] == 1.0
