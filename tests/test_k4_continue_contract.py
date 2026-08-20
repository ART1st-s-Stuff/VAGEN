from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from omegaconf import OmegaConf

from vagen.joint_policy.integration_gate import (
    K4_ID184_CONTINUE_GATE_IMPLEMENTATION,
    JointIntegrationGate,
)
from vagen.main_ppo import _configure_joint_actor_extension


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "vagen/configs/joint_id184_continue.yaml"
TRAIN = ROOT / "examples/train/navigation/train_navigation_joint_id184.yaml"
VAL = ROOT / "examples/train/navigation/val_navigation_joint_id184.yaml"


def _env() -> dict[str, str]:
    root = "/project/peilab/atst/nimloth/outputs/experiments"
    return {
        "ID184_TRAIN_CONFIG": "/tmp/train_navigation_joint_id184.yaml",
        "ID184_VAL_CONFIG": "/tmp/val_navigation_joint_id184.yaml",
        "ID184_ACTOR_MODEL": (
            f"{root}/training/sft2/2026-08-15/"
            "176_id74_action_head_repair_balanced271x8_val40x8/checkpoint"
        ),
        "ID184_PLANNING_CHECKPOINT": (
            f"{root}/vagen_legacy_wm_k16_grid/2026-08-02/sft2/"
            "74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_"
            "ws16n3g844lw844_px100352/train_ws16/epoch_001"
        ),
        "ID184_SOURCE_CHECKPOINT": (
            f"{root}/training/rl/2026-08-16/"
            "183_canary_k4schemeb_jointupdate_dp8_tp8_u10_r5_"
            "train3x8_t20_s100_c1_a1_b85p78297006578457_t1_"
            "cot07p095_val5x8_retry2/checkpoints/global_step_10"
        ),
        "ID184_RUN_OUT": "/tmp/184_continue",
        "ID184_RUN_NAME": (
            "184_continue_k4schemeb_jointupdate_dp8_tp8_u20_"
            "from10_train3x60_b24_test"
        ),
        "ID184_AGENT_CONFIG": "/tmp/agent.yaml",
    }


def _config_source():
    source = OmegaConf.load(CONFIG)
    return OmegaConf.merge(
        OmegaConf.create({"actor_rollout_ref": {"actor": {"optim": {}}}}),
        source,
    )


def test_id184_gate_is_one_fresh_resume_from_ten_to_twenty() -> None:
    gate = JointIntegrationGate(
        implementation=K4_ID184_CONTINUE_GATE_IMPLEMENTATION,
        experiment_id=184,
        phase="resume_10_to_20",
    )
    assert gate.expected_total_training_steps == 20
    assert gate.expected_resume_mode == "resume_path"
    with pytest.raises(ValueError, match="phase"):
        JointIntegrationGate(
            implementation=K4_ID184_CONTINUE_GATE_IMPLEMENTATION,
            experiment_id=184,
            phase="train_from_scratch",
        )


def test_id184_full_config_accepts_only_the_approved_continuation() -> None:
    with patch.dict(os.environ, _env()):
        config = _config_source()
        training = _configure_joint_actor_extension(config)
        assert training.run_seed == 42179
        assert training.checkpoint_frequency == 5
        assert config.trainer.total_training_steps == 20
        assert config.trainer.total_epochs == 20
        assert config.trainer.resume_mode == "resume_path"
        assert config.trainer.resume_from_path.endswith("global_step_10")
        assert config.trainer.joint_dataloader_resume_policy == "reset"
        assert config.trainer.val_before_train is True
        assert config.trainer.test_freq == 5
        assert config.trainer.save_freq == 5
        assert config.data.train_batch_size == 24
        assert config.data.gen_batch_size == 24
        assert config.data.shuffle is True
        assert config.data.seed == 42184
        assert config.trainer.nnodes == 4
        assert config.trainer.n_gpus_per_node == 2

        drift = _config_source()
        drift.trainer.joint_dataloader_resume_policy = "exact"
        with pytest.raises(ValueError, match="ID184.*dataloader"):
            _configure_joint_actor_extension(drift)

        drift = _config_source()
        drift.trainer.resume_from_path = "/tmp/global_step_10"
        with pytest.raises(ValueError, match="ID184.*checkpoint"):
            _configure_joint_actor_extension(drift)

        drift = _config_source()
        drift.data.train_batch_size = 48
        with pytest.raises(ValueError, match="ID184.*batch"):
            _configure_joint_actor_extension(drift)


def test_id184_restore_migrates_only_the_snapshot_transport_path() -> None:
    source = (ROOT / "vagen/ray_trainer.py").read_text()
    assert "ID184_TRAINING_CONTRACT_PATH_MIGRATION_OK" in source
    assert "source_snapshot_root = str(Path(active_transport).parents[1])" in source
    assert "source_world_model = replace(" in source
    assert "snapshot_transport_root=source_snapshot_root" in source
    assert "ID184_DATALOADER_RESET_OK global_step=10" in source
    assert 'joint_dataloader_resume_policy",' in source
    assert "id184_reset = (" in source
    assert "ID186_DATALOADER_RESET_OK global_step=20" in source


def test_id184_expands_only_training_to_three_by_sixty_unique_tasks() -> None:
    train = yaml.safe_load(TRAIN.read_text())["envs"]
    val = yaml.safe_load(VAL.read_text())["envs"]
    assert len(train) == 3
    assert len(val) == 5
    assert {row["config"]["eval_set"] for row in train} == {
        "base_train",
        "common_sense_train",
        "long_horizon_train",
    }
    for row in train:
        assert row["n_envs"] == 60
        assert row["seed"] == [0, 1199, 1]
        assert row["max_turns"] == 20
    for row in val:
        assert row["n_envs"] == 8
        assert row["seed_list"][:8] == list(range(8))
        assert len(row["seed_list"]) > row["n_envs"]
    for row in [*train, *val]:
        assert row["config"]["per_turn_format_reward"] == 0.01
        assert row["config"]["format_reward"] == 0.0
        assert row["config"]["success_reward"] == 1.0
