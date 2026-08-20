from __future__ import annotations

import math
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from omegaconf import OmegaConf

from vagen.joint_policy.integration_gate import (
    K4_ID185_FULL_EVAL_GATE_IMPLEMENTATION,
    JointIntegrationGate,
)
from vagen.main_ppo import _configure_joint_actor_extension
from vagen.ray_trainer import (
    _dump_single_rollout_visualization_audit,
    _visualization_turn_record,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "vagen/configs/joint_id185_visualize_one.yaml"


def _env() -> dict[str, str]:
    root = "/project/peilab/atst/nimloth/outputs/experiments"
    return {
        "ID185_VIS_TRAIN_CONFIG": "/tmp/train_navigation_joint_id185.yaml",
        "ID185_VIS_VAL_CONFIG": "/tmp/val_navigation_joint_id185.yaml",
        "ID185_VIS_ACTOR_MODEL": (
            f"{root}/training/sft2/2026-08-15/"
            "176_id74_action_head_repair_balanced271x8_val40x8/checkpoint"
        ),
        "ID185_VIS_PLANNING_CHECKPOINT": (
            f"{root}/vagen_legacy_wm_k16_grid/2026-08-02/sft2/"
            "74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_"
            "ws16n3g844lw844_px100352/train_ws16/epoch_001"
        ),
        "ID185_VIS_SOURCE_CHECKPOINT": (
            f"{root}/training/rl/2026-08-17/"
            "184_continue_k4schemeb_jointupdate_dp8_tp8_u20_from10_"
            "train3x60_b24_t20_s100_c1_a1_b85p78297006578457_t1_"
            "cot07p095_val5x8_retry1/checkpoints/global_step_20"
        ),
        "ID185_VIS_RUN_OUT": "/tmp/185_visualization",
        "ID185_VIS_RUN_NAME": (
            "185_visualize_k4schemeb_dp8_tp8_source20_base_failed"
        ),
        "ID185_VIS_AGENT_CONFIG": "/tmp/agent.yaml",
        "ID185_VIS_SEED": "2",
    }


def test_id185_visualization_config_is_one_read_only_rollout() -> None:
    gate = JointIntegrationGate(
        implementation=K4_ID185_FULL_EVAL_GATE_IMPLEMENTATION,
        experiment_id=185,
        phase="visualize_one",
    )
    assert gate.expected_total_training_steps == 20
    assert gate.expected_resume_mode == "resume_path"
    with patch.dict(os.environ, _env()):
        source = OmegaConf.load(CONFIG)
        config = OmegaConf.merge(
            OmegaConf.create({"actor_rollout_ref": {"actor": {"optim": {}}}}),
            source,
        )
        _configure_joint_actor_extension(config)
        assert config.trainer.val_only is True
        assert set(config.trainer.logger) == {"console"}
        assert config.trainer.validation_batch_journal_expected_rows == 1
        assert config.trainer.validation_visualization_data_source == (
            "navigation_base_test_id185"
        )
        assert int(config.trainer.validation_visualization_seed) == 2
        assert config.trainer.validation_visualization_audit_dir.endswith(
            "/visualization/rollout_audit"
        )
        assert config.trainer.validation_rollout_browser_expected_rows == 1
        assert config.trainer.validation_rollout_browser_policy_family == (
            "vagen_k4_joint"
        )
        assert config.trainer.validation_rollout_browser_dir.endswith(
            "/evaluation_browser"
        )


def test_visualization_audit_atomically_persists_true_image(tmp_path: Path) -> None:
    image_values = np.empty((1,), dtype=object)
    image_values[0] = [Image.new("RGB", (8, 8), color="red")]

    class FakeBatch:
        non_tensor_batch = {
            "group_idx": np.array(["group"], dtype=object),
            "traj_idx": np.array([0], dtype=object),
            "turn_idx": np.array([1], dtype=object),
            "rollout_sample_id": np.array(["sha256:sample"], dtype=object),
            "rollout_repeat_index": np.array([0], dtype=object),
            "reward_extra_info": np.array(
                [{"traj_success": 0.0}], dtype=object
            ),
            "image_data": image_values,
        }

        def __len__(self) -> int:
            return 1

    record = {
        "turn_index": 0,
        "turn_number": 1,
        "cot": "<think>real</think>",
        "action_ranking": [],
        "predicted_action_sequences": [],
    }
    destination = tmp_path / "audit"
    with patch(
        "vagen.ray_trainer._visualization_turn_record",
        return_value=record,
    ):
        payload = _dump_single_rollout_visualization_audit(
            FakeBatch(), destination
        )
    assert payload["rollout_sample_id"] == "sha256:sample"
    assert payload["success"] is False
    assert (destination / "step_00_observation.png").is_file()
    assert (destination / "rollout_audit.json").is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        _dump_single_rollout_visualization_audit(FakeBatch(), destination)


def test_visualization_turn_exposes_values_and_predicted_action_lists() -> None:
    names = [
        "MoveAhead",
        "RotateLeft",
        "RotateRight",
        "LookUp",
        "LookDown",
        "MoveBack",
        "MoveLeft",
        "MoveRight",
    ]
    prior_logits = [0.0] * 8
    root_values = [index / 100 for index in range(8)]
    behavior = {
        "snapshot_id": "snapshot",
        "action_space_names": names,
        "policy_config": {
            "implementation": "k4_mcts_guided_v1",
            "alpha": 1.0,
            "beta": 85.78297006578457,
            "prior_temperature": 1.0,
            "backprop_to_llm": True,
            "score_dtype": "float32",
            "planning_horizon": 4,
            "mcts_num_simulations": 100,
            "mcts_exploration_constant": 1.0,
        },
        "prior_logits": prior_logits,
        "prior_log_probs": [-math.log(8)] * 8,
        "direct_all_action_q": [index / 10 for index in range(8)],
        "planner_root_mean_values": root_values,
        "planner_root_visit_counts": [12, 12, 12, 12, 12, 12, 14, 14],
        "prior_action_id": 0,
        "guided_action_id": 7,
    }
    record = _visualization_turn_record(
        {
            "decision_ledger": {
                "behavior_record": behavior,
                "env_turn_reward": 0.01,
                "env_terminated": False,
                "rollout_truncated": False,
            },
            "frozen_k4_planning_scoring": {
                "snapshot_id": "snapshot",
                "snapshot_source_step": 796,
                "request_id": "request",
                "generation_id": "generation",
                "candidate_sequences": [[7, 6, 5, 4], [0, 1, 2, 3]],
                "candidate_mean_values": [0.9, 0.2],
                "candidate_visit_counts": [75, 25],
                "planner_latency_seconds": 0.5,
            },
            "policy_response_trace": {
                "raw_response": "<think>real generated reasoning</think><latent>"
            },
            "guided_action_draw": {},
            "guided_action_execution": {},
            "turn_idx": 1,
            "guided_turn_index": 0,
            "rollout_stop_reason": "continue",
        }
    )
    assert record["cot"] == "<think>real generated reasoning</think>"
    assert record["prior_action"] == "MoveAhead"
    assert record["executed_action"] == "MoveRight"
    assert record["action_ranking"][0]["action"] == "MoveRight"
    assert record["predicted_action_sequences"][0] == {
        "action_ids": [7, 6, 5, 4],
        "actions": ["MoveRight", "MoveLeft", "MoveBack", "LookDown"],
        "predicted_value": 0.9,
        "visits": 75,
    }
    assert record["current_state_value"] > 0.0
    assert record["executed_action_predicted_value"] == 0.07
