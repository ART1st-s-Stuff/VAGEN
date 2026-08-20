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
SOURCE = (
    "/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-17/"
    "184_continue_k4schemeb_jointupdate_dp8_tp8_u20_from10_train3x60_b24_"
    "t20_s100_c1_a1_b85p78297006578457_t1_cot07p095_val5x8_retry1/"
    "checkpoints/global_step_20"
)


def _env() -> dict[str, str]:
    return {
        "ID187_TRAIN_CONFIG": "/tmp/train_navigation_joint_id187.yaml",
        "ID187_VAL_CONFIG": "/tmp/val_navigation_joint_id187.yaml",
        "ID187_ACTOR_MODEL": (
            "/project/peilab/atst/nimloth/outputs/experiments/training/sft2/"
            "2026-08-15/176_id74_action_head_repair_balanced271x8_val40x8/checkpoint"
        ),
        "ID187_PLANNING_CHECKPOINT": (
            "/project/peilab/atst/nimloth/outputs/experiments/"
            "vagen_legacy_wm_k16_grid/2026-08-02/sft2/"
            "74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_"
            "ws16n3g844lw844_px100352/train_ws16/epoch_001"
        ),
        "ID187_SOURCE_CHECKPOINT": SOURCE,
        "ID187_AGENT_CONFIG": "/tmp/agent.yaml",
        "ID187_RUN_NAME": (
            "187_smoke_rollout_browser_k4_dp8_tp8_source20_base_seed2_"
            "t20_s100_preempt_retry6"
        ),
        "ID187_RUN_OUT": "/tmp/id187",
        "ID187_SEED": "2",
    }


def _config():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="joint_id187_source20_visualize_one")


def test_id187_source20_browser_gate_requires_exact_read_only_restore() -> None:
    with patch.dict(os.environ, _env(), clear=False):
        config = _config()
        _configure_joint_actor_extension(config)
        assert config.trainer.resume_mode == "resume_path"
        assert config.trainer.nnodes == 2
        assert config.trainer.n_gpus_per_node == 4
        assert list(config.trainer.joint_process_on_nodes) == [6, 2]
        assert config.trainer.resume_from_path == SOURCE
        assert config.trainer.val_only is True
        assert config.trainer.validation_rollout_browser_expected_rows == 1
        assert config.trainer.validation_rollout_browser_capture_mcts_process is True
        assert config.trainer.validation_rollout_browser_checkpoint_identity == SOURCE

        drift = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        drift.trainer.resume_from_path = "/tmp/global_step_20"
        with pytest.raises(ValueError, match="ID187.*checkpoint"):
            _configure_joint_actor_extension(drift)

        drift = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        drift.trainer.validation_rollout_browser_expected_rows = 40
        with pytest.raises(ValueError, match="ID187.*browser"):
            _configure_joint_actor_extension(drift)

        drift = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        drift.trainer.joint_process_on_nodes = [4, 4]
        with pytest.raises(ValueError, match="heterogeneous Ray pool"):
            _configure_joint_actor_extension(drift)

        drift = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        drift.trainer.validation_rollout_browser_capture_mcts_process = False
        with pytest.raises(ValueError, match="ID187.*browser"):
            _configure_joint_actor_extension(drift)
