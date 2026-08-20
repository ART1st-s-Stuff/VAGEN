from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from vagen.main_ppo import _configure_joint_actor_extension


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "vagen/configs"


def _env() -> dict[str, str]:
    return {
        "ID188_TRAIN_CONFIG": "/tmp/train_navigation_joint_id188.yaml",
        "ID188_VAL_CONFIG": "/tmp/val_navigation_joint_id188.yaml",
        "ID188_ACTOR_MODEL": (
            "/project/peilab/atst/nimloth/outputs/experiments/training/sft2/"
            "2026-08-15/176_id74_action_head_repair_balanced271x8_val40x8/checkpoint"
        ),
        "ID188_PLANNING_CHECKPOINT": (
            "/project/peilab/atst/nimloth/outputs/experiments/"
            "vagen_legacy_wm_k16_grid/2026-08-02/sft2/"
            "74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_"
            "ws16n3g844lw844_px100352/train_ws16/epoch_001"
        ),
        "ID188_AGENT_CONFIG": "/tmp/agent.yaml",
        "ID188_RUN_NAME": (
            "188_smoke_rollout_browser_k4_dp8_tp8_step0_base_seed2_t20_s100"
        ),
        "ID188_RUN_OUT": "/tmp/id188",
        "ID188_SEED": "2",
    }


def _config():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="joint_id188_step0_visualize_one")


def test_id188_step0_browser_gate_accepts_only_frozen_val_only_bootstrap() -> None:
    with patch.dict(os.environ, _env(), clear=False):
        config = _config()
        training = _configure_joint_actor_extension(config)
        assert training.initial_snapshot_source_step == 776
        assert config.trainer.nnodes == 1
        assert config.trainer.n_gpus_per_node == 8
        assert list(config.trainer.joint_process_on_nodes) == [8]
        assert config.trainer.val_before_train is True
        assert config.trainer.val_only is True
        assert config.trainer.resume_mode == "disable"
        assert config.trainer.total_training_steps == 1
        assert config.trainer.validation_rollout_browser_expected_rows == 1
        assert config.trainer.validation_rollout_browser_capture_mcts_process is True
        assert config.trainer.validation_rollout_browser_source_step == 776
        assert config.trainer.validation_rollout_browser_checkpoint_identity == _env()[
            "ID188_ACTOR_MODEL"
        ]

        drift = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        drift.trainer.resume_mode = "resume_path"
        drift.trainer.resume_from_path = "/tmp/global_step_20"
        with pytest.raises(ValueError, match="ID188.*resume"):
            _configure_joint_actor_extension(drift)

        drift = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        drift.trainer.validation_rollout_browser_expected_rows = 40
        with pytest.raises(ValueError, match="ID188.*browser"):
            _configure_joint_actor_extension(drift)

        drift = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        drift.trainer.joint_process_on_nodes = [4, 4]
        with pytest.raises(ValueError, match="heterogeneous Ray pool"):
            _configure_joint_actor_extension(drift)

        drift = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        drift.trainer.validation_rollout_browser_capture_mcts_process = False
        with pytest.raises(ValueError, match="ID188.*browser"):
            _configure_joint_actor_extension(drift)

        drift = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        drift.trainer.val_only = False
        with pytest.raises(ValueError, match="ID188.*validation"):
            _configure_joint_actor_extension(drift)
