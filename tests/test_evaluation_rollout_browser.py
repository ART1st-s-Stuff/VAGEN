from __future__ import annotations

from unittest.mock import patch

import numpy as np
from PIL import Image

from nimloth.eval.rollout_browser import (
    finalize_evaluation_browser,
    write_evaluation_browser_batch,
)
from vagen.ray_trainer import _build_validation_rollout_browser_artifacts


def _object_array(values):
    array = np.empty((len(values),), dtype=object)
    for index, value in enumerate(values):
        array[index] = value
    return array


class FakeBatch:
    def __init__(self, rows):
        self.non_tensor_batch = {
            key: _object_array([row.get(key) for row in rows])
            for key in rows[0]
        }

    def __len__(self):
        return len(next(iter(self.non_tensor_batch.values())))


def _ledger(action=0):
    names = ["move_forward", "turn_left"]
    return {
        "schema": "vagen_decision_ledger_v1",
        "action_space": "navigation_v1",
        "action_space_names": names,
        "executed_action_ids": [action],
        "executed_action_names": [names[action]],
        "decision_sources": ["llm_text"],
        "decision_is_policy_sampled": [False],
        "env_turn_reward": 0.01,
        "env_terminated": False,
        "rollout_truncated": True,
        "format_valid": True,
    }


def _metadata(sample):
    return {
        (sample, 0): {
            "data_source": "navigation_base",
            "seed": 2,
            "split": "test",
            "reward": 0.2,
            "success": False,
        }
    }


def test_standard_vagen_rollout_declares_missing_planner_capabilities(
    tmp_path,
) -> None:
    sample = "sha256:sample"
    batch = FakeBatch(
        [
            {
                "group_idx": "group",
                "traj_idx": 0,
                "turn_idx": 1,
                "guided_turn_index": 0,
                "rollout_sample_id": sample,
                "rollout_repeat_index": 0,
                "rollout_stop_reason": "task_failure",
                "task_instruction": "navigate to the toaster",
                "raw_response": "<think>forward</think><action>",
                "decision_ledger": _ledger(),
                "image_data": [Image.new("RGB", (8, 8), "red")],
                "terminal_image_data": None,
                "terminal_state_trace": None,
            }
        ]
    )
    artifacts = _build_validation_rollout_browser_artifacts(
        batch,
        _metadata(sample),
        policy_family="vagen_greedy",
    )
    assert len(artifacts) == 1
    audit = artifacts[0].audit
    assert audit["task"] == "navigate to the toaster"
    assert audit["capabilities"]["planner"] is False
    assert audit["capabilities"]["action_distribution"] is False
    assert audit["turns"][0]["executed_action"]["name"] == "move_forward"
    root = tmp_path / "browser"
    write_evaluation_browser_batch(root, artifacts, batch_index=0)
    manifest = finalize_evaluation_browser(
        root,
        evaluation={
            "evaluation_id": "vagen-greedy",
            "policy_family": "vagen_greedy",
            "global_step": 0,
            "source_step": None,
            "checkpoint_identity": "sha256:model",
            "snapshot_identity": None,
        },
        expected_rollouts=1,
        expected_batches=1,
    )
    assert manifest["rollout_count"] == 1


def _policy_state_with_complete_mcts_trace():
    sequence_nodes = [()] + [tuple([0] * depth) for depth in range(1, 5)]
    nodes = [
        {
            "sequence": list(sequence),
            "depth": len(sequence),
            "predicted_state": (
                None if not sequence else np.full((8, 1024), len(sequence), dtype=np.float32).tolist()
            ),
            "visit_count": 100,
            "value_sum": 50.0,
            "mean_value": 0.5,
        }
        for sequence in sequence_nodes
    ]
    simulations = [
        {
            "simulation_index": index,
            "selection_steps": [
                {
                    "depth": depth,
                    "parent_sequence": [0] * depth,
                    "parent_visit_count": index,
                    "operation": "expand" if index == 0 else "select",
                    "action_id": 0,
                    "child_sequence": [0] * (depth + 1),
                    "uct_candidates": (
                        []
                        if index == 0
                        else [
                            {
                                "action_id": 0,
                                "child_sequence": [0] * (depth + 1),
                                "visit_count": index,
                                "mean_value": 0.5,
                                "exploration_bonus": 0.1,
                                "uct_score": 0.6,
                            }
                        ]
                    ),
                }
                for depth in range(4)
            ],
            "leaf": {
                "sequence": [0, 0, 0, 0],
                "decision_sequence": [0, 0, 0],
                "action_id": 0,
                "action_values": [0.5, 0.1],
                "value": 0.5,
            },
            "backups": [
                {
                    "sequence": [0] * depth,
                    "visit_count_before": index,
                    "value_sum_before": index * 0.5,
                    "visit_count_after": index + 1,
                    "value_sum_after": (index + 1) * 0.5,
                    "mean_value_after": 0.5,
                }
                for depth in range(5)
            ],
        }
        for index in range(100)
    ]
    return {
        "latent_hidden": np.zeros((16, 2048), dtype=np.float32).tolist(),
        "frozen_k4_planning": {
            "current_state": np.ones((8, 1024), dtype=np.float32).tolist(),
            "mcts_trace": {
                "schema": "nimloth_k4_mcts_process_v1",
                "num_simulations": 100,
                "horizon": 4,
                "exploration_constant": 1.0,
                "tree_nodes": nodes,
                "simulations": simulations,
            },
        },
    }


def test_k4_vagen_rollout_preserves_all_planner_candidates(tmp_path) -> None:
    sample = "sha256:k4"
    ledger = _ledger()
    ledger["schema"] = "vagen_decision_ledger_v3_k4_mcts_guided"
    ledger["snapshot_id"] = "sha256:snapshot"
    ledger["contract_id"] = "contract"
    ledger["behavior_record_id"] = "behavior"
    ledger["behavior_record"] = {}
    batch = FakeBatch(
        [
            {
                "group_idx": "group",
                "traj_idx": 0,
                "turn_idx": 1,
                "guided_turn_index": 0,
                "rollout_sample_id": sample,
                "rollout_repeat_index": 0,
                "rollout_stop_reason": "task_failure",
                "task_instruction": "navigate to the toaster",
                "raw_response": "<think>forward</think><action>",
                "decision_ledger": ledger,
                "frozen_k4_planning_scoring": {"present": True},
                "policy_state": _policy_state_with_complete_mcts_trace(),
                "image_data": [Image.new("RGB", (8, 8), "red")],
                "terminal_image_data": [Image.new("RGB", (8, 8), "blue")],
                "terminal_state_trace": {
                    "raw_response": "<think>terminal</think>",
                    "rollout_stop_reason": "task_failure",
                },
            }
        ]
    )
    record = {
        "turn_index": 0,
        "turn_number": 1,
        "cot": "<think>forward</think>",
        "raw_response": "<think>forward</think><action>",
        "prior_action": "turn_left",
        "executed_action": "move_forward",
        "current_state_value": 0.4,
        "executed_action_direct_q": 0.5,
        "executed_action_predicted_value": 0.6,
        "planning_horizon": 4,
        "mcts_num_simulations": 100,
        "mcts_exploration_constant": 1.0,
        "action_ranking": [
            {
                "action_id": 0,
                "action": "move_forward",
                "prior_probability": 0.4,
                "guided_probability": 0.8,
                "direct_q": 0.5,
                "predicted_root_value": 0.6,
                "root_visits": 60,
                "is_prior_action": False,
                "is_executed_action": True,
            },
            {
                "action_id": 1,
                "action": "turn_left",
                "prior_probability": 0.6,
                "guided_probability": 0.2,
                "direct_q": 0.1,
                "predicted_root_value": 0.2,
                "root_visits": 40,
                "is_prior_action": True,
                "is_executed_action": False,
            },
        ],
        "predicted_action_sequences": [
            {
                "action_ids": [index % 2] * 4,
                "actions": ["move_forward" if index % 2 == 0 else "turn_left"] * 4,
                "predicted_value": index / 100,
                "visits": 1,
            }
            for index in range(100)
        ],
        "env_turn_reward": 0.01,
        "env_terminated": False,
        "rollout_truncated": True,
        "rollout_stop_reason": "task_failure",
        "planner_latency_seconds": 0.1,
        "snapshot_id": "sha256:snapshot",
        "snapshot_source_step": 796,
        "request_id": "request",
        "generation_id": "generation",
    }
    with patch("vagen.ray_trainer._visualization_turn_record", return_value=record):
        artifact = _build_validation_rollout_browser_artifacts(
            batch,
            _metadata(sample),
            policy_family="vagen_k4_joint",
        )[0]
    planner = artifact.audit["turns"][0]["planner"]
    assert len(planner["candidates"]) == 100
    assert sum(row["visits"] for row in planner["candidates"]) == 100
    assert artifact.audit["capabilities"]["direct_q"] is True
    assert artifact.audit["capabilities"]["model_state"] is True
    assert artifact.audit["capabilities"]["mcts_process"] is True
    assert len(planner["mcts_process"]["simulations"]) == 100
    assert artifact.audit["turns"][0]["model_state"]["arrays"]["current_state"]["shape"] == [8, 1024]
    root = tmp_path / "k4-browser"
    write_evaluation_browser_batch(root, [artifact], batch_index=0)
    archives = list(root.glob("batches/batch_0000/rollouts/*/step_00_model_states.npz"))
    assert len(archives) == 1
    with np.load(archives[0], allow_pickle=False) as state_archive:
        assert state_archive["latent_hidden"].shape == (16, 2048)
        assert state_archive["current_state"].shape == (8, 1024)
        assert state_archive["mcts_node_states"].shape == (4, 8, 1024)
    manifest = finalize_evaluation_browser(
        root,
        evaluation={
            "evaluation_id": "vagen-k4",
            "policy_family": "vagen_k4_joint",
            "global_step": 20,
            "source_step": None,
            "checkpoint_identity": "sha256:model",
            "snapshot_identity": None,
        },
        expected_rollouts=1,
        expected_batches=1,
    )
    assert manifest["snapshot_identity"] == "sha256:snapshot"
    assert manifest["source_step"] == 796
