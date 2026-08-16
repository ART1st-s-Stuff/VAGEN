from __future__ import annotations

import json
import math
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from vagen.joint_policy.canary import summarize_canary_validation_rows
from vagen.joint_policy.integration_gate import (
    K4_ID183_CANARY_GATE_IMPLEMENTATION,
    JointIntegrationGate,
)
from vagen.main_ppo import _configure_joint_actor_extension


ROOT = Path(__file__).resolve().parents[1]


def _checkpoint_roots() -> tuple[str, str]:
    actor = (
        "/project/peilab/atst/nimloth/outputs/experiments/training/sft2/"
        "2026-08-15/176_id74_action_head_repair_balanced271x8_val40x8/"
        "checkpoint"
    )
    planning = (
        "/project/peilab/atst/nimloth/outputs/experiments/"
        "vagen_legacy_wm_k16_grid/2026-08-02/sft2/"
        "74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_"
        "ws16n3g844lw844_px100352/train_ws16/epoch_001"
    )
    return actor, planning


def _env() -> dict[str, str]:
    actor, planning = _checkpoint_roots()
    return {
        "ID183_TRAIN_CONFIG": "/tmp/train_navigation_joint_id183.yaml",
        "ID183_VAL_CONFIG": "/tmp/val_navigation_joint_id183.yaml",
        "ID183_ACTOR_MODEL": actor,
        "ID183_PLANNING_CHECKPOINT": planning,
        "ID183_RUN_OUT": "/tmp/183_canary",
        "ID183_RUN_NAME": (
            "183_canary_k4schemeb_jointupdate_dp8_tp8_u10_r5_test"
        ),
        "ID183_AGENT_CONFIG": "/tmp/agent.yaml",
    }


def test_id183_dataset_class_is_importable_without_vagen_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from verl.utils.import_utils import load_extern_type

    source = OmegaConf.load(ROOT / "vagen/configs/joint_id183_canary.yaml")
    monkeypatch.chdir(tmp_path)
    assert source.data.custom_cls.path == "pkg://vagen.gym_agent_dataset"
    dataset_type = load_extern_type(
        source.data.custom_cls.path,
        source.data.custom_cls.name,
    )
    assert dataset_type.__name__ == "AgenticDataset"


def test_gate_phases_are_exactly_five_then_fresh_ten() -> None:
    first = JointIntegrationGate(
        implementation=K4_ID183_CANARY_GATE_IMPLEMENTATION,
        experiment_id=183,
        phase="train_to_5",
    )
    second = JointIntegrationGate(
        implementation=K4_ID183_CANARY_GATE_IMPLEMENTATION,
        experiment_id=183,
        phase="resume_to_10",
    )
    assert first.expected_total_training_steps == 5
    assert first.expected_resume_mode == "disable"
    assert second.expected_total_training_steps == 10
    assert second.expected_resume_mode == "auto"
    for invalid in ("update_1", "restore_only", "resume_update_2"):
        with pytest.raises(ValueError, match="phase"):
            JointIntegrationGate(
                implementation=K4_ID183_CANARY_GATE_IMPLEMENTATION,
                experiment_id=183,
                phase=invalid,
            )


def _config_source():
    source = OmegaConf.load(ROOT / "vagen/configs/joint_id183_canary.yaml")
    return OmegaConf.merge(
        OmegaConf.create(
            {"actor_rollout_ref": {"actor": {"optim": {}}}}
        ),
        source,
    )


def test_full_id183_config_accepts_both_phases_and_rejects_drift() -> None:
    with patch.dict(os.environ, _env()):
        first = _config_source()
        training = _configure_joint_actor_extension(first)
        assert training.run_seed == 42179
        assert training.checkpoint_frequency == 5
        assert first.trainer.total_training_steps == 5
        assert first.trainer.total_epochs == 5
        assert first.trainer.val_before_train is True
        assert first.trainer.test_freq == -1
        assert first.trainer.save_freq == 5
        assert first.trainer.nnodes == 2
        assert first.trainer.n_gpus_per_node == 4
        assert first.ray_kwargs.ray_init.address == "auto"

        second = _config_source()
        second.joint_integration_gate.phase = "resume_to_10"
        second.trainer.total_training_steps = 10
        second.trainer.total_epochs = 10
        second.trainer.resume_mode = "auto"
        second.trainer.val_before_train = False
        second.trainer.test_freq = 10
        _configure_joint_actor_extension(second)

        drift = _config_source()
        drift.trainer.val_before_train = False
        with pytest.raises(ValueError, match="ID183.*validation"):
            _configure_joint_actor_extension(drift)

        drift = _config_source()
        drift.trainer.save_freq = 1
        with pytest.raises(ValueError, match="ID183.*checkpoint"):
            _configure_joint_actor_extension(drift)

        drift = _config_source()
        drift.joint_policy.beta = 85.0
        with pytest.raises(ValueError, match="ID183.*numerical"):
            _configure_joint_actor_extension(drift)

        drift = _config_source()
        drift.trainer.nnodes = 1
        drift.trainer.n_gpus_per_node = 8
        with pytest.raises(ValueError, match="multi-node 2x4"):
            _configure_joint_actor_extension(drift)


def test_target_tp8_uses_verl_multinode_tcp_control_path() -> None:
    spmd = (
        ROOT
        / "verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    ).read_text()
    replica = (
        ROOT
        / "verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py"
    ).read_text()
    assert 'local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])' in spmd
    assert 'socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"' in spmd
    assert 'address = f"tcp://{ip}:{port}"' in spmd
    assert "if self.config.data_parallel_size == 1:" in replica
    assert "gpus_per_node = self.world_size" in replica
    assert "workers = self.workers[node_rank * gpus_per_node" in replica
    assert "VERL_VLLM_ZMQ_ADDRESSES" in replica


def _rows(step: int) -> list[dict[str, object]]:
    rows = []
    for source_index, source in enumerate(
        (
            "navigation_base_val_id183",
            "navigation_common_sense_val_id183",
            "navigation_long_horizon_val_id183",
            "navigation_complex_instruction_val_id183",
            "navigation_visual_appearance_val_id183",
        )
    ):
        for seed in range(8):
            rows.append(
                {
                    "data_source": source,
                    "rollout_sample_id": f"sha256:{source_index:02x}{seed:02x}" + "0" * 60,
                    "score": float(seed == 0),
                    "traj_success": float(seed == 0),
                    "step": step,
                }
            )
    return rows


def test_validation_dump_persists_stable_sample_and_split_provenance() -> None:
    source = (ROOT / "vagen/ray_trainer.py").read_text()
    validation_start = source.index("def _validate(self):")
    validation_end = source.index("def init_workers", validation_start)
    validation = source[validation_start:validation_end]
    assert 'validation_dump_extras["data_source"]' in validation
    assert 'validation_dump_extras["rollout_sample_id"]' in validation
    assert "sample_data_sources.extend" in validation
    assert "sample_rollout_sample_ids.extend" in validation


def test_canary_validation_summary_requires_exact_five_by_eight() -> None:
    expected = (
        "navigation_base_val_id183",
        "navigation_common_sense_val_id183",
        "navigation_long_horizon_val_id183",
        "navigation_complex_instruction_val_id183",
        "navigation_visual_appearance_val_id183",
    )
    summary = summarize_canary_validation_rows(
        _rows(0),
        expected_data_sources=expected,
        expected_rows_per_source=8,
        expected_step=0,
    )
    assert summary["row_count"] == 40
    assert summary["success_count"] == 5
    assert summary["success_rate"] == 0.125
    assert set(summary["by_data_source"]) == set(expected)
    assert all(item["row_count"] == 8 for item in summary["by_data_source"].values())

    duplicate = _rows(0)
    duplicate[1]["rollout_sample_id"] = duplicate[0]["rollout_sample_id"]
    with pytest.raises(ValueError, match="unique"):
        summarize_canary_validation_rows(
            duplicate,
            expected_data_sources=expected,
            expected_rows_per_source=8,
            expected_step=0,
        )

    missing = _rows(0)[:-1]
    with pytest.raises(ValueError, match="exactly"):
        summarize_canary_validation_rows(
            missing,
            expected_data_sources=expected,
            expected_rows_per_source=8,
            expected_step=0,
        )

    invalid = _rows(0)
    invalid[0]["score"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        summarize_canary_validation_rows(
            invalid,
            expected_data_sources=expected,
            expected_rows_per_source=8,
            expected_step=0,
        )

    invalid = _rows(0)
    invalid[0]["traj_success"] = True
    with pytest.raises(ValueError, match="real number"):
        summarize_canary_validation_rows(
            invalid,
            expected_data_sources=expected,
            expected_rows_per_source=8,
            expected_step=0,
        )
    gym_loop = (ROOT / "vagen/agent_loop/gym_agent_loop_no_concat.py").read_text()
    assert '"traj_success": float(traj_success)' in gym_loop
