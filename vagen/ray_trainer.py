# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import hashlib
import json
import math
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from vagen.agent_loop.decision_ledger import (
    DECISION_LEDGER_SCHEMA,
    GUIDED_DECISION_LEDGER_SCHEMA,
    K4_GUIDED_DECISION_LEDGER_SCHEMA,
    parse_decision_ledger_enabled,
    summarize_decision_ledger_batch,
    validate_decision_ledger_reward_rows,
)
from vagen.joint_policy import (
    K4MCTSGuidedPolicyConfig,
    parse_joint_policy_section,
    parse_k4_world_model_training_section,
    validate_k4_joint_training_alignment,
)
from vagen.joint_policy.integration_gate import parse_joint_integration_gate
from vagen.joint_policy.training_contract import (
    JOINT_ADVANTAGE_ESTIMATOR,
    parse_joint_training_section,
)
from vagen.utils.image_dump_actor import ImageDumpActor
from vagen.utils.upload_hugging_face import HFUploadManager
from vagen.utils.image_validation_logger import ValidationGenerationsLogger
from vagen.utils.concat_val_multi_turn import concat_val_multi_turn
from vagen.utils.image_token_utils import replace_image_tokens_for_logging
import vagen.custom_advantage
from vagen.custom_metric.metric import METRIC_REGISTRY
from vagen.custom_filter.filter import FILTER_REGISTRY


def _assign_unique_validation_padding_identities(
    batch: DataProto,
    pad_size: int,
) -> None:
    """Give only framework-added validation padding rows dummy identities.

    ``pad_dataproto_to_divisor`` copies real rows onto the tail.  Guided
    rollout identity ownership must still reject duplicates among real rows,
    while no-concat validation must be able to generate and then discard the
    copied rows.  Synthetic UIDs are therefore assigned only to the known
    padding suffix and remain outside the pre-padding UID set.
    """

    if isinstance(pad_size, bool) or not isinstance(pad_size, int):
        raise TypeError("validation pad_size must be int")
    if pad_size < 0:
        raise ValueError("validation pad_size must be nonnegative")
    if pad_size == 0:
        return
    total_size = len(batch)
    if pad_size >= total_size:
        raise ValueError("validation padding must leave at least one real row")
    required = {
        "uid",
        "group_idx",
        "rollout_sample_id",
        "rollout_repeat_index",
    }
    missing = required - set(batch.non_tensor_batch)
    if missing:
        raise ValueError(
            "validation padding identity rewrite is missing fields: "
            f"{sorted(missing)}"
        )

    values_by_key: dict[str, np.ndarray] = {}
    for key in required:
        values = np.asarray(batch.non_tensor_batch[key], dtype=object)
        if len(values) != total_size:
            raise ValueError(
                f"validation padding field {key} has wrong batch size"
            )
        values_by_key[key] = values.copy()

    original_size = total_size - pad_size
    original_uids = set(values_by_key["uid"][:original_size].tolist())
    original_sample_ids = set(
        values_by_key["rollout_sample_id"][:original_size].tolist()
    )
    generated: set[str] = set()
    for offset, row in enumerate(range(original_size, total_size)):
        source_uid = values_by_key["uid"][row]
        source_sample_id = values_by_key["rollout_sample_id"][row]
        source_repeat = values_by_key["rollout_repeat_index"][row]
        identity_seed = (
            "vagen-validation-padding-v1:"
            f"{original_size}:{offset}:{source_uid!r}:"
            f"{source_sample_id!r}:{source_repeat!r}"
        )
        synthetic = (
            "__vagen_validation_padding__"
            f"{uuid.uuid5(uuid.NAMESPACE_URL, identity_seed)}"
        )
        if (
            synthetic in original_uids
            or synthetic in original_sample_ids
            or synthetic in generated
        ):
            raise ValueError("validation padding identity collision")
        generated.add(synthetic)
        values_by_key["uid"][row] = synthetic
        values_by_key["group_idx"][row] = synthetic
        values_by_key["rollout_sample_id"][row] = synthetic

    for key, values in values_by_key.items():
        batch.non_tensor_batch[key] = values


def _journal_json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"validation journal cannot serialize {type(value)!r}")


def _atomic_write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite validation journal: {path}")
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(
                f"validation journal appeared concurrently: {path}"
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_validation_batch_journal(
    *,
    journal_dir: str | os.PathLike[str],
    global_step: int,
    batch_index: int,
    inputs: list[str],
    outputs: list[str],
    ground_truths: list[Any],
    scores: list[float],
    uids: list[str],
    data_sources: list[str],
    rollout_sample_ids: list[str],
    rollout_repeat_indices: list[int],
    reward_extra_infos: dict[str, list[Any]],
) -> dict[str, Any]:
    """Atomically persist one completed validation batch and its marker."""

    if isinstance(batch_index, bool) or not isinstance(batch_index, int):
        raise TypeError("validation journal batch_index must be int")
    if batch_index < 0:
        raise ValueError("validation journal batch_index must be nonnegative")
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        raise TypeError("validation journal global_step must be int")
    row_count = len(inputs)
    fields = {
        "outputs": outputs,
        "ground_truths": ground_truths,
        "scores": scores,
        "uids": uids,
        "data_sources": data_sources,
        "rollout_sample_ids": rollout_sample_ids,
        "rollout_repeat_indices": rollout_repeat_indices,
    }
    for name, values in fields.items():
        if len(values) != row_count:
            raise ValueError(
                f"validation journal {name} length does not match inputs"
            )
    for name, values in reward_extra_infos.items():
        if len(values) != row_count:
            raise ValueError(
                f"validation journal reward extra {name} length mismatch"
            )

    rows: list[dict[str, Any]] = []
    identities: list[tuple[str, int]] = []
    for row in range(row_count):
        sample_id = rollout_sample_ids[row]
        repeat_index = rollout_repeat_indices[row]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(
                "validation journal sample identity must be non-empty str"
            )
        if isinstance(repeat_index, bool) or not isinstance(
            repeat_index,
            (int, np.integer),
        ):
            raise TypeError("validation journal repeat index must be int")
        repeat_index = int(repeat_index)
        if repeat_index < 0:
            raise ValueError(
                "validation journal repeat index must be nonnegative"
            )
        identities.append((sample_id, repeat_index))
        entry = {
            "input": inputs[row],
            "output": outputs[row],
            "gts": ground_truths[row],
            "score": scores[row],
            "step": global_step,
            "uid": str(uids[row]),
            "data_source": str(data_sources[row]),
            "rollout_sample_id": sample_id,
            "rollout_repeat_index": repeat_index,
        }
        for name, values in reward_extra_infos.items():
            entry[name] = values[row]
        rows.append(entry)
    if len(set(identities)) != len(identities):
        raise ValueError(
            "validation journal batch contains duplicate sample/repeat identities"
        )

    encoded_lines = [
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            default=_journal_json_default,
        )
        for row in rows
    ]
    rows_payload = ("\n".join(encoded_lines) + "\n").encode("utf-8")
    rows_sha256 = hashlib.sha256(rows_payload).hexdigest()
    identity_payload = json.dumps(
        identities,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    identity_sha256 = hashlib.sha256(identity_payload).hexdigest()
    root = Path(journal_dir)
    stem = f"batch_{batch_index:04d}"
    rows_path = root / f"{stem}.jsonl"
    marker_path = root / f"{stem}.complete.json"
    if rows_path.exists() or marker_path.exists():
        raise FileExistsError(
            f"validation journal batch {batch_index} already exists"
        )
    marker = {
        "schema": "vagen_validation_batch_journal_v1",
        "global_step": global_step,
        "batch_index": batch_index,
        "row_count": row_count,
        "rows_file": rows_path.name,
        "rows_sha256": f"sha256:{rows_sha256}",
        "identity_sha256": f"sha256:{identity_sha256}",
    }
    marker_payload = (
        json.dumps(marker, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write_new(rows_path, rows_payload)
    _atomic_write_new(marker_path, marker_payload)
    print(
        "VALIDATION_BATCH_JOURNAL_COMMIT "
        f"batch_index={batch_index} rows={row_count} "
        f"sha256={rows_sha256}"
    )
    return marker


def _finalize_validation_batch_journal(
    *,
    journal_dir: str | os.PathLike[str],
    global_step: int,
    expected_batch_count: int,
    expected_row_count: int,
) -> dict[str, Any]:
    """Verify all immutable batch files before publishing one complete marker."""

    if expected_batch_count <= 0 or expected_row_count <= 0:
        raise ValueError("validation journal expected counts must be positive")
    root = Path(journal_dir)
    markers = sorted(root.glob("batch_*.complete.json"))
    if len(markers) != expected_batch_count:
        raise ValueError(
            "validation journal batch count mismatch: "
            f"{len(markers)} != {expected_batch_count}"
        )
    all_identities: list[tuple[str, int]] = []
    data_source_counts: dict[str, int] = defaultdict(int)
    row_digests: list[str] = []
    total_rows = 0
    for expected_index, marker_path in enumerate(markers):
        marker = json.loads(marker_path.read_text())
        if marker != {
            **marker,
            "schema": "vagen_validation_batch_journal_v1",
            "global_step": global_step,
            "batch_index": expected_index,
        }:
            raise ValueError(
                f"validation journal marker mismatch: {marker_path}"
            )
        rows_path = root / marker["rows_file"]
        rows_payload = rows_path.read_bytes()
        actual_digest = "sha256:" + hashlib.sha256(rows_payload).hexdigest()
        if actual_digest != marker["rows_sha256"]:
            raise ValueError(
                f"validation journal row hash mismatch: {rows_path}"
            )
        rows = [
            json.loads(line)
            for line in rows_payload.decode("utf-8").splitlines()
            if line
        ]
        if len(rows) != marker["row_count"]:
            raise ValueError(
                f"validation journal row count mismatch: {rows_path}"
            )
        for row in rows:
            if row["step"] != global_step:
                raise ValueError("validation journal row step mismatch")
            identity = (
                row["rollout_sample_id"],
                int(row["rollout_repeat_index"]),
            )
            all_identities.append(identity)
            data_source_counts[str(row["data_source"])] += 1
        total_rows += len(rows)
        row_digests.append(marker["rows_sha256"])
    if total_rows != expected_row_count:
        raise ValueError(
            "validation journal total row count mismatch: "
            f"{total_rows} != {expected_row_count}"
        )
    if len(set(all_identities)) != len(all_identities):
        raise ValueError(
            "validation journal contains duplicate sample/repeat identities"
        )
    complete = {
        "schema": "vagen_validation_batch_journal_complete_v1",
        "global_step": global_step,
        "batch_count": expected_batch_count,
        "row_count": total_rows,
        "data_source_counts": dict(sorted(data_source_counts.items())),
        "batch_rows_sha256": row_digests,
    }
    complete_payload = (
        json.dumps(complete, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write_new(root / "complete.json", complete_payload)
    print(
        "VALIDATION_BATCH_JOURNAL_COMPLETE "
        f"batches={expected_batch_count} rows={total_rows}"
    )
    return complete


def _visualization_turn_record(raw: dict[str, Any]) -> dict[str, Any]:
    required = {
        "decision_ledger",
        "frozen_k4_planning_scoring",
        "policy_response_trace",
        "guided_action_draw",
        "guided_action_execution",
        "turn_idx",
        "guided_turn_index",
        "rollout_stop_reason",
    }
    missing = required - set(raw)
    if missing:
        raise ValueError(
            "visualization turn is missing fields: " f"{sorted(missing)}"
        )
    ledger = raw["decision_ledger"]
    behavior = ledger["behavior_record"]
    scoring = raw["frozen_k4_planning_scoring"]
    trace = raw["policy_response_trace"]
    if behavior["snapshot_id"] != scoring["snapshot_id"]:
        raise ValueError("visualization turn snapshot identity mismatch")
    names = list(behavior["action_space_names"])
    action_count = len(names)
    vectors = {
        "prior_log_probs": behavior["prior_log_probs"],
        "direct_all_action_q": behavior["direct_all_action_q"],
        "planner_root_mean_values": behavior["planner_root_mean_values"],
        "planner_root_visit_counts": behavior["planner_root_visit_counts"],
    }
    if any(len(values) != action_count for values in vectors.values()):
        raise ValueError("visualization action vectors do not align")
    from vagen.joint_policy.planning_contract import (
        K4MCTSGuidedPolicyConfig,
        k4_guided_log_probs_reference,
    )

    config = K4MCTSGuidedPolicyConfig.from_mapping(
        behavior["policy_config"]
    )
    _, guided_log_probs = k4_guided_log_probs_reference(
        behavior["prior_logits"],
        behavior["planner_root_mean_values"],
        config,
    )
    prior_probs = [math.exp(value) for value in behavior["prior_log_probs"]]
    guided_probs = [math.exp(value) for value in guided_log_probs]
    direct_q = [float(value) for value in behavior["direct_all_action_q"]]
    root_values = [
        float(value) for value in behavior["planner_root_mean_values"]
    ]
    visits = [int(value) for value in behavior["planner_root_visit_counts"]]
    current_state_value = sum(
        probability * value
        for probability, value in zip(guided_probs, direct_q, strict=True)
    )
    selected = int(behavior["guided_action_id"])
    action_rows = [
        {
            "action_id": action_id,
            "action": name,
            "prior_probability": prior_probs[action_id],
            "guided_probability": guided_probs[action_id],
            "direct_q": direct_q[action_id],
            "predicted_root_value": root_values[action_id],
            "root_visits": visits[action_id],
            "is_prior_action": action_id == int(behavior["prior_action_id"]),
            "is_executed_action": action_id == selected,
        }
        for action_id, name in enumerate(names)
    ]
    action_ranking = sorted(
        action_rows,
        key=lambda row: (
            -row["guided_probability"],
            -row["predicted_root_value"],
            row["action_id"],
        ),
    )
    candidates = []
    for sequence, value, count in zip(
        scoring["candidate_sequences"],
        scoring["candidate_mean_values"],
        scoring["candidate_visit_counts"],
        strict=True,
    ):
        candidates.append(
            {
                "action_ids": [int(action_id) for action_id in sequence],
                "actions": [names[int(action_id)] for action_id in sequence],
                "predicted_value": float(value),
                "visits": int(count),
            }
        )
    candidates.sort(
        key=lambda row: (-row["visits"], -row["predicted_value"], row["action_ids"])
    )
    raw_response = trace["raw_response"]
    if not isinstance(raw_response, str) or "</think>" not in raw_response:
        raise ValueError("visualization turn raw CoT is malformed")
    cot = raw_response.split("</think>", 1)[0] + "</think>"
    return {
        "turn_index": int(raw["guided_turn_index"]),
        "turn_number": int(raw["turn_idx"]),
        "cot": cot,
        "raw_response": raw_response,
        "prior_action": names[int(behavior["prior_action_id"])],
        "executed_action": names[selected],
        "current_state_value": current_state_value,
        "executed_action_direct_q": direct_q[selected],
        "executed_action_predicted_value": root_values[selected],
        "planning_horizon": config.planning_horizon,
        "mcts_num_simulations": config.mcts_num_simulations,
        "mcts_exploration_constant": config.mcts_exploration_constant,
        "action_ranking": action_ranking,
        "predicted_action_sequences": candidates,
        "env_turn_reward": float(ledger["env_turn_reward"]),
        "env_terminated": bool(ledger["env_terminated"]),
        "rollout_truncated": bool(ledger["rollout_truncated"]),
        "rollout_stop_reason": str(raw["rollout_stop_reason"]),
        "planner_latency_seconds": float(scoring["planner_latency_seconds"]),
        "snapshot_id": behavior["snapshot_id"],
        "snapshot_source_step": int(scoring["snapshot_source_step"]),
        "request_id": scoring["request_id"],
        "generation_id": scoring["generation_id"],
    }


def _dump_single_rollout_visualization_audit(
    batch: DataProto,
    audit_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Atomically persist true per-turn images and immutable planning evidence."""

    if len(batch) < 1:
        raise ValueError("visualization audit requires at least one turn")
    root = Path(audit_dir)
    if root.exists():
        raise FileExistsError(f"visualization audit output already exists: {root}")
    temporary = root.parent / f".{root.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True)
    try:
        non_tensor = batch.non_tensor_batch
        group_ids = {str(value) for value in non_tensor["group_idx"]}
        trajectory_ids = {int(value) for value in non_tensor["traj_idx"]}
        if len(group_ids) != 1 or trajectory_ids != {0}:
            raise ValueError("visualization audit requires exactly one trajectory")
        order = sorted(
            range(len(batch)),
            key=lambda index: int(non_tensor["turn_idx"][index]),
        )
        turns = []
        for position, index in enumerate(order):
            raw = {key: values[index] for key, values in non_tensor.items()}
            turn = _visualization_turn_record(raw)
            if turn["turn_index"] != position:
                raise ValueError("visualization audit turn indices are not contiguous")
            images = raw.get("image_data")
            if (
                not isinstance(images, (list, tuple, np.ndarray))
                or len(images) == 0
            ):
                raise ValueError("visualization turn is missing true observation image")
            observation = list(images)[-1]
            image_name = f"step_{position:02d}_observation.png"
            observation.save(temporary / image_name, format="PNG")
            turn["observation_image"] = image_name
            terminal_images = raw.get("terminal_image_data")
            terminal_trace = raw.get("terminal_state_trace")
            if terminal_images is not None or terminal_trace is not None:
                if position != len(order) - 1:
                    raise ValueError("terminal visualization evidence is not last")
                if (
                    not isinstance(terminal_images, (list, tuple, np.ndarray))
                    or len(terminal_images) == 0
                    or not isinstance(terminal_trace, dict)
                ):
                    raise ValueError("terminal visualization evidence is incomplete")
                terminal_name = "terminal_observation.png"
                list(terminal_images)[-1].save(
                    temporary / terminal_name,
                    format="PNG",
                )
                terminal_raw = terminal_trace["raw_response"]
                turn["terminal"] = {
                    "observation_image": terminal_name,
                    "cot": (
                        terminal_raw.split("</think>", 1)[0] + "</think>"
                        if "</think>" in terminal_raw
                        else terminal_raw
                    ),
                    "raw_response": terminal_raw,
                    "rollout_stop_reason": terminal_trace[
                        "rollout_stop_reason"
                    ],
                }
            turns.append(turn)
        payload = {
            "schema": "vagen_single_rollout_visualization_audit_v1",
            "rollout_sample_id": str(
                non_tensor["rollout_sample_id"][order[0]]
            ),
            "rollout_repeat_index": int(
                non_tensor["rollout_repeat_index"][order[0]]
            ),
            "turn_count": len(turns),
            "success": bool(
                non_tensor["reward_extra_info"][order[-1]]["traj_success"]
            ),
            "turns": turns,
        }
        audit_path = temporary / "rollout_audit.json"
        audit_path.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n"
        )
        with audit_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.rename(temporary, root)
    finally:
        if temporary.exists():
            import shutil

            shutil.rmtree(temporary)
    print(
        "SINGLE_ROLLOUT_VISUALIZATION_AUDIT_COMPLETE "
        f"sample_id={payload['rollout_sample_id']} turns={payload['turn_count']}"
    )
    return payload


def _actual_cot(raw_response: str) -> str | None:
    if not isinstance(raw_response, str) or not raw_response.startswith("<think>"):
        return None
    boundary = raw_response.find("</think>")
    return (
        raw_response[: boundary + len("</think>")]
        if boundary >= 0
        else None
    )


def _build_validation_rollout_browser_artifacts(
    batch: DataProto,
    trajectory_metadata: dict[tuple[str, int], dict[str, Any]],
    *,
    policy_family: str,
) -> list[Any]:
    """Adapt unpadded per-turn VAGEN evidence without model replay."""

    from nimloth.eval.rollout_browser.sft_adapter import RolloutBrowserArtifact
    from nimloth.eval.rollout_browser.schema import ROLLOUT_AUDIT_SCHEMA

    if len(batch) < 1:
        raise ValueError("evaluation rollout browser batch is empty")
    non_tensor = batch.non_tensor_batch
    required = {
        "group_idx",
        "traj_idx",
        "turn_idx",
        "rollout_sample_id",
        "rollout_repeat_index",
        "rollout_stop_reason",
        "task_instruction",
        "raw_response",
        "decision_ledger",
        "image_data",
    }
    missing = required - set(non_tensor)
    if missing:
        raise ValueError(
            "evaluation rollout browser is missing turn fields: "
            f"{sorted(missing)}"
        )
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index in range(len(batch)):
        key = (
            str(non_tensor["rollout_sample_id"][index]),
            int(non_tensor["rollout_repeat_index"][index]),
        )
        groups[key].append(index)
    if set(groups) != set(trajectory_metadata):
        raise ValueError("evaluation browser turn/trajectory identities do not align")
    artifacts = []
    for key in sorted(groups):
        indices = sorted(groups[key], key=lambda index: int(non_tensor["turn_idx"][index]))
        metadata = trajectory_metadata[key]
        tasks = {str(non_tensor["task_instruction"][index]) for index in indices}
        if len(tasks) != 1 or not next(iter(tasks)).strip() or tasks == {"None"}:
            raise ValueError("evaluation rollout task instruction is missing or inconsistent")
        task = next(iter(tasks))
        joint = all(
            non_tensor.get("frozen_k4_planning_scoring", [None] * len(batch))[index]
            is not None
            for index in indices
        )
        if any(
            non_tensor.get("frozen_k4_planning_scoring", [None] * len(batch))[index]
            is not None
            for index in indices
        ) != joint:
            raise ValueError("evaluation rollout has partial K4 planning evidence")
        raw_responses = [str(non_tensor["raw_response"][index]) for index in indices]
        cots = [_actual_cot(response) for response in raw_responses]
        has_cot = all(cot is not None for cot in cots)
        turns = []
        image_sources: dict[str, Any] = {}
        action_space_names: list[str] | None = None
        snapshot_ids: set[str] = set()
        source_steps: set[int] = set()
        has_terminal = False
        for turn_index, index in enumerate(indices):
            if int(non_tensor["turn_idx"][index]) != turn_index + 1:
                raise ValueError("evaluation rollout turn indices are not contiguous")
            raw = {name: values[index] for name, values in non_tensor.items()}
            ledger = raw["decision_ledger"]
            names = list(ledger["action_space_names"])
            if action_space_names is None:
                action_space_names = names
            elif action_space_names != names:
                raise ValueError("evaluation rollout action space changed between turns")
            executed_ids = list(ledger["executed_action_ids"])
            executed_names = list(ledger["executed_action_names"])
            if len(executed_ids) != len(executed_names) or len(executed_ids) > 1:
                raise ValueError("evaluation rollout requires at most one executed action")
            executed = (
                {"id": int(executed_ids[0]), "name": str(executed_names[0])}
                if executed_ids
                else None
            )
            images = raw["image_data"]
            if not isinstance(images, (list, tuple, np.ndarray)) or not len(images):
                raise ValueError("evaluation rollout turn is missing true image")
            image_name = f"step_{turn_index:02d}_observation.png"
            image_sources[image_name] = list(images)[-1]
            turn: dict[str, Any] = {
                "turn_index": turn_index,
                "observation": {
                    "text": "",
                    "image": image_name,
                    "sha256": "sha256:pending",
                },
                "raw_response": raw_responses[turn_index],
                "cot": cots[turn_index],
                "executed_action": executed,
                "environment": {
                    "reward": float(ledger["env_turn_reward"]),
                    "terminated": bool(ledger["env_terminated"]),
                    "truncated": bool(ledger["rollout_truncated"]),
                    "stop_reason": str(raw["rollout_stop_reason"]),
                },
            }
            if joint:
                record = _visualization_turn_record(raw)
                if record["raw_response"] != raw_responses[turn_index]:
                    raise ValueError(
                        "evaluation rollout raw response provenance mismatch"
                    )
                if executed is None or record["executed_action"] != executed["name"]:
                    raise ValueError(
                        "evaluation rollout executed action provenance mismatch"
                    )
                ordered = sorted(record["action_ranking"], key=lambda row: row["action_id"])
                turn["action_distribution"] = {
                    "kind": "guided_policy",
                    "log_probabilities": [
                        math.log(float(row["guided_probability"]))
                        if float(row["guided_probability"]) > 0.0
                        else None
                        for row in ordered
                    ],
                    "prior_probabilities": [
                        float(row["prior_probability"]) for row in ordered
                    ],
                }
                turn["direct_q"] = {
                    "values": [float(row["direct_q"]) for row in ordered],
                    "state_value": float(record["current_state_value"]),
                }
                candidates = [
                    {
                        "action_ids": list(row["action_ids"]),
                        "actions": list(row["actions"]),
                        "score": float(row["predicted_value"]),
                        "visits": int(row["visits"]),
                    }
                    for row in record["predicted_action_sequences"]
                ]
                turn["planner"] = {
                    "search_mode": "mcts",
                    "horizon": int(record["planning_horizon"]),
                    "num_simulations": int(record["mcts_num_simulations"]),
                    "exploration_constant": float(record["mcts_exploration_constant"]),
                    "root_scores": [
                        float(row["predicted_root_value"]) for row in ordered
                    ],
                    "root_visits": [int(row["root_visits"]) for row in ordered],
                    "candidates": candidates,
                }
                snapshot_ids.add(str(record["snapshot_id"]))
                source_steps.add(int(record["snapshot_source_step"]))
            terminal_images = raw.get("terminal_image_data")
            terminal_trace = raw.get("terminal_state_trace")
            if terminal_images is not None or terminal_trace is not None:
                if turn_index != len(indices) - 1:
                    raise ValueError("evaluation terminal evidence is not final")
                if (
                    not isinstance(terminal_images, (list, tuple, np.ndarray))
                    or not len(terminal_images)
                    or not isinstance(terminal_trace, dict)
                ):
                    raise ValueError("evaluation terminal evidence is incomplete")
                terminal_name = "terminal_observation.png"
                image_sources[terminal_name] = list(terminal_images)[-1]
                terminal_raw = str(terminal_trace["raw_response"])
                turn["terminal"] = {
                    "observation": {
                        "text": "",
                        "image": terminal_name,
                        "sha256": "sha256:pending",
                    },
                    "raw_response": terminal_raw,
                    "cot": _actual_cot(terminal_raw),
                    "stop_reason": str(terminal_trace["rollout_stop_reason"]),
                    "action_executed": False,
                }
                has_terminal = True
            turns.append(turn)
        final_ledger = non_tensor["decision_ledger"][indices[-1]]
        if joint and (len(snapshot_ids) != 1 or len(source_steps) != 1):
            raise ValueError("evaluation rollout snapshot identity changed between turns")
        audit = {
            "schema": ROLLOUT_AUDIT_SCHEMA,
            "identity": {
                "rollout_sample_id": key[0],
                "rollout_repeat_index": key[1],
                "record_id": str(non_tensor["group_idx"][indices[0]]),
            },
            "policy_family": policy_family,
            "action_space": {
                "id": str(final_ledger["action_space"]),
                "version": 1,
                "names": action_space_names,
            },
            "capabilities": {
                "task": True,
                "observations": True,
                "terminal_observation": has_terminal,
                "cot": has_cot,
                "token_trace": False,
                "action_distribution": joint,
                "direct_q": joint,
                "state_value": joint,
                "planner": joint,
                "mcts": joint,
            },
            "task": task,
            "data_source": metadata["data_source"],
            "seed": metadata["seed"],
            "split": metadata["split"],
            "success": bool(metadata["success"]),
            "reward": float(metadata["reward"]),
            "terminated": bool(final_ledger["env_terminated"]),
            "truncated": bool(final_ledger["rollout_truncated"]),
            "stop_reason": str(non_tensor["rollout_stop_reason"][indices[-1]]),
            "turn_count": len(turns),
            "provenance": {
                "snapshot_id": next(iter(snapshot_ids)) if snapshot_ids else None,
                "source_step": next(iter(source_steps)) if source_steps else None,
            },
            "turns": turns,
        }
        artifacts.append(RolloutBrowserArtifact(audit=audit, image_sources=image_sources))
    return artifacts


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld
    token_level_kl_penalty = -beta * kld
    data.batch["token_level_kl_penalty"] = token_level_kl_penalty

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_custom_metrics(data: DataProto, prefix: str = "custom_metrics") -> dict:
    """Compute all custom metrics registered in METRIC_REGISTRY.

    Args:
        data (DataProto): The data containing batch information.
        prefix (str): Prefix for metric names in the returned dictionary.

    Returns:
        dict: A dictionary containing all computed custom metrics with appropriate prefixes.
    """
    custom_metrics = {}

    for metric_name, metric_fn in METRIC_REGISTRY.items():
        try:
            metric_value = metric_fn(data)
            custom_metrics[f"{prefix}/{metric_name}"] = metric_value
        except Exception as e:
            print(f"Warning: Failed to compute custom metric '{metric_name}': {e}")

    return custom_metrics

def _default_eps(
    x: torch.Tensor,
    small_eps: float = 1e-2,
    large_eps: float = 1e-6,
) -> float:
    """
    Choose a comparison tolerance (eps) based on tensor dtype.
    """
    if x.dtype in (torch.float16, torch.bfloat16):
        return small_eps
    return large_eps


def compute_value_mask(
    data: DataProto,
    ignore_value: float = -100.0,
    eps: float | None = None,
) -> torch.Tensor:
    """
    Compute value-function loss mask from token-level returns.

    Value loss is only computed at positions where `returns` is valid.
    Invalid / ignored positions are marked by a float sentinel
    `ignore_value` (default: -100.0), similar in spirit to
    CrossEntropy's `ignore_index`.

    If you do NOT want a certain token position to participate in
    value-function training, simply write:

        returns[..., pos] = ignore_value

    This mask will then automatically exclude that position from
    value loss computation.
    """
    returns = data.batch["returns"]

    if eps is None:
        eps = _default_eps(returns)

    # Identify ignored positions via approximate comparison
    is_ignored = (returns - ignore_value).abs() < eps

    # Mask dtype is aligned with response_mask for numerical stability
    return (~is_ignored).to(dtype=data.batch["response_mask"].dtype)


    

def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "data": data,
            "config": config,
            "gamma": gamma,
            "lam": lam,
            "num_repeat": num_repeat,
            "norm_adv_by_std_in_grpo": norm_adv_by_std_in_grpo,
        }
        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        self._image_dump_actors = {}
        self._pending_dump_futures = []
        self._log_image_cfg = self.config.trainer.get("log_image", {})
        self._log_image_enable = self._log_image_cfg.get("enable", False)
        self._max_pending_dumps = self._log_image_cfg.get("max_pending", 2)

        ledger_enabled = parse_decision_ledger_enabled(
            self.config.get("decision_ledger")
        )
        self.joint_policy_config = parse_joint_policy_section(
            self.config.get("joint_policy", {"enabled": False})
        )
        self.joint_training_config = parse_joint_training_section(
            self.config.get("joint_training", {"enabled": False})
        )
        self.k4_world_model_training_config = (
            parse_k4_world_model_training_section(
                self.config.get(
                    "k4_world_model_training",
                    {"enabled": False},
                )
            )
        )
        self.joint_integration_gate = parse_joint_integration_gate(
            self.config.get("joint_integration_gate", {"enabled": False})
        )
        if (self.joint_policy_config is None) != (
            self.joint_training_config is None
        ):
            raise ValueError(
                "joint_policy and joint_training must be enabled together"
            )
        is_k4 = isinstance(
            self.joint_policy_config,
            K4MCTSGuidedPolicyConfig,
        )
        if is_k4 != (self.k4_world_model_training_config is not None):
            raise ValueError(
                "K4 joint policy and world-model training must be enabled together"
            )
        if (
            self.k4_world_model_training_config is not None
            and self.joint_training_config is None
        ):
            raise ValueError("K4 world-model training requires joint training")
        if self.k4_world_model_training_config is not None:
            validate_k4_joint_training_alignment(
                self.joint_training_config,
                self.k4_world_model_training_config,
            )
        if (
            self.joint_integration_gate is not None
            and self.joint_training_config is None
        ):
            raise ValueError("joint integration gate requires joint training")
        if self.joint_policy_config is not None and not ledger_enabled:
            raise ValueError(
                "joint_policy.enabled requires decision_ledger.enabled=true"
            )
        if ledger_enabled and (
            self.config.actor_rollout_ref.rollout.mode != "async"
            or self.config.trainer.get("concat_multi_turn", True)
        ):
            raise ValueError(
                "decision_ledger.enabled requires async rollout with "
                "trainer.concat_multi_turn=false"
            )
        if self.joint_training_config is not None:
            if self.config.algorithm.adv_estimator != JOINT_ADVANTAGE_ESTIMATOR:
                raise ValueError(
                    "joint training requires algorithm.adv_estimator="
                    f"{JOINT_ADVANTAGE_ESTIMATOR}"
                )
            if self.use_critic:
                raise ValueError(
                    "joint training must disable the stock scalar critic role"
                )
            if (
                self.joint_training_config.token_kl_coefficient > 0.0
                and not self.use_reference_policy
            ):
                raise ValueError(
                    "joint token KL requires a frozen reference policy worker"
                )
            if self.joint_integration_gate is None:
                raise NotImplementedError(
                    "joint update and atomic resume paths exist but have not passed "
                    "the required Torch/Ray/distributed integration gates; "
                    "refusing production training"
                )

        # HuggingFace Hub upload
        self._hf_upload_manager = HFUploadManager(config)

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, images, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

        # Save images to subfolders
        if images and self._log_image_enable:
            actor = self._image_dump_actors.get(dump_path)
            if actor is None:
                actor = ImageDumpActor.remote(base_dir=dump_path)
                self._image_dump_actors[dump_path] = actor

            compress_level = self._log_image_cfg.get("png_compress_level", 0)
            fut = actor.dump_images.remote(
                step=self.global_steps,
                images=images,
                compress_level=compress_level,
            )
            self._pending_dump_futures.append(fut)

            if self._max_pending_dumps > 0 and len(self._pending_dump_futures) > self._max_pending_dumps:
                done, rest = ray.wait(self._pending_dump_futures, num_returns=1)
                ray.get(done)
                self._pending_dump_futures = rest

    def _flush_image_dumps(self):
        if not self._pending_dump_futures:
            return
        ray.get(self._pending_dump_futures)
        self._pending_dump_futures = []

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            
            inputs = batch.batch["prompts"]
            outputs = batch.batch["responses"]
            
            # remove pad tokens for logging (keeps other special tokens like <|endoftext|>
            # visible so we can spot degenerate model outputs)
            pad_token_id = self.tokenizer.pad_token_id
            skip_pad_tokens = self.config.trainer.get("skip_pad_tokens", True)
            if skip_pad_tokens:
                inputs = self.tokenizer.batch_decode(
                    [s[-l:] if l else [] for s, l in zip(inputs.tolist(),  (inputs  != pad_token_id).sum(1).tolist())],
                    skip_special_tokens=False)
                outputs = self.tokenizer.batch_decode(
                    [s[:l]  if l else [] for s, l in zip(outputs.tolist(), (outputs != pad_token_id).sum(1).tolist())],
                    skip_special_tokens=False)
            else:
                inputs = self.tokenizer.batch_decode(inputs.tolist(), skip_special_tokens=False)
                outputs = self.tokenizer.batch_decode(outputs.tolist(), skip_special_tokens=False)

            if self.config.trainer.get("replace_image_tokens_for_logging", False):
                inputs = replace_image_tokens_for_logging(inputs, processor=self.processor, tokenizer=self.tokenizer)
                outputs = replace_image_tokens_for_logging(outputs, processor=self.processor, tokenizer=self.tokenizer)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]
            # Extract images from non_tensor_batch (extra_fields are stored there)
            sample_images=[]
            if "image_data" in batch.non_tensor_batch:
                batch_images = batch.non_tensor_batch["image_data"]
                sample_images.extend(batch_images.tolist() if hasattr(batch_images, 'tolist') else batch_images)
            else:
                sample_images.extend([None] * len(outputs))
            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                images=sample_images,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores, images=None):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score, image) and sort by input text
        if images is None or len(images) == 0:
            images = [None] * len(inputs)
        else:
            non_none_count = sum(1 for img in images if img is not None)
            print(f"Logging {non_none_count}/{len(images)} validation samples with images to wandb")

        samples = list(zip(inputs, outputs, scores, images, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _assign_group_and_traj_idx(self, gen_batch: DataProto, num_traj_per_sample: int) -> None:
        """Assign group_idx and traj_idx for no-concat mode.

        Args:
            gen_batch: The generated batch after repeat operation
            num_traj_per_sample: Number of trajectories per sample (repeat_times)
        """
        # Assign group_idx from uid
        gen_batch.non_tensor_batch["group_idx"] = gen_batch.non_tensor_batch["uid"]

        # Assign traj_idx based on repeat pattern
        # Since repeat with interleave=True creates [A, A, A, B, B, B, C, C, C] pattern,
        # traj_idx should be [0, 1, 2, 0, 1, 2, 0, 1, 2]
        batch_size = len(gen_batch.non_tensor_batch["uid"])
        traj_idx = np.tile(np.arange(num_traj_per_sample), batch_size // num_traj_per_sample)
        gen_batch.non_tensor_batch["traj_idx"] = traj_idx

        # The UUID-backed group_idx remains an internal GAE grouping key.  A
        # guided draw instead uses the dataset's restart-stable sample identity
        # plus this explicit repeat index.
        if "rollout_sample_id" in gen_batch.non_tensor_batch:
            sample_ids = gen_batch.non_tensor_batch["rollout_sample_id"]
            if len(sample_ids) != batch_size:
                raise ValueError("rollout_sample_id batch size mismatch")
            if any(not isinstance(value, str) or not value for value in sample_ids):
                raise ValueError("rollout_sample_id must contain non-empty strings")
            gen_batch.non_tensor_batch["rollout_sample_id"] = sample_ids
            gen_batch.non_tensor_batch["rollout_repeat_index"] = traj_idx.copy()


    def _post_process_no_concat_batch(self, batch: DataProto, gen_batch_output: DataProto) -> DataProto:
        """Re-align and union batch with gen_batch_output in no-concat mode.

        In no-concat mode, each trajectory has multiple prompt-response pairs with varying lengths.
        Each original batch item may correspond to a different number of gen_batch_output items
        depending on how many turns were generated for that trajectory.

        The key insight: we build a selection index list that maps each gen_batch_output item
        to its corresponding original batch item, then use select_idxs to replicate and reorder
        the batch to match gen_batch_output's uid sequence.

        Args:
            batch: Original batch with reward model keys (uid, data_source, etc.)
            gen_batch_output: Generated output with sequences and uid (variable items per original uid)

        Returns:
            DataProto: Aligned and unified batch ready for downstream processing

        Example:
            Original batch: [item_0 (uid=C), item_1 (uid=B), item_2 (uid=A)]
            gen_batch_output uids: [A, A, A, B, B, C, C, C, C]
            -> selection_indices: [2, 2, 2, 1, 1, 0, 0, 0, 0]
            -> select_idxs([2,2,2,1,1,0,0,0,0]): [A, A, A, B, B, C, C, C, C]
            -> Perfectly aligned with gen_batch_output!
        """
        # Step 1: Verify uid exists in both batches
        assert "uid" in batch.non_tensor_batch, "batch must contain 'uid' in non_tensor_batch for alignment"
        gen_batch_output.non_tensor_batch["uid"]=gen_batch_output.non_tensor_batch["group_idx"]
        assert "uid" in gen_batch_output.non_tensor_batch, (
            "gen_batch_output must contain 'uid' in non_tensor_batch for alignment"
        )

        # Step 2: Build uid to index mapping for original batch
        batch_uid_to_idx = {str(uid): idx for idx, uid in enumerate(batch.non_tensor_batch["uid"])}

        # Step 3: Build selection indices by mapping each gen_batch_output uid to its batch index
        # This automatically handles:
        # - Variable repetition (each uid can appear different number of times)
        # - Arbitrary ordering (gen_batch_output uids can be in any order)
        selection_indices = []

        for gen_uid in gen_batch_output.non_tensor_batch["uid"]:
            gen_uid_str = str(gen_uid)
            if gen_uid_str not in batch_uid_to_idx:
                raise ValueError(
                    f"uid '{gen_uid_str}' from gen_batch_output not found in batch. "
                    f"Available uids: {list(batch_uid_to_idx.keys())[:5]}... "
                    f"This suggests a data alignment issue in agent loop."
                )
            batch_idx = batch_uid_to_idx[gen_uid_str]
            selection_indices.append(batch_idx)

        # Step 4: Use select_idxs to replicate and reorder batch to match gen_batch_output
        # This single operation handles both repetition and reordering
        batch = batch.select_idxs(selection_indices)

        # Step 5: Verify the size matches
        assert len(batch) == len(gen_batch_output), (
            f"After alignment, batch size ({len(batch)}) should match gen_batch_output size ({len(gen_batch_output)}). "
            f"selection_indices length: {len(selection_indices)}"
        )

        # Step 6: Union the aligned batches
        batch = batch.union(gen_batch_output)

        return batch

    def _validate_decision_ledger_batch(self, batch: DataProto, metrics: dict) -> None:
        """Validate no-concat execution facts before policy replay."""

        if not parse_decision_ledger_enabled(self.config.get("decision_ledger")):
            return
        ledgers = batch.non_tensor_batch.get("decision_ledger")
        if ledgers is None:
            raise ValueError("no-concat rollout is missing the required decision_ledger")
        metrics.update(
            summarize_decision_ledger_batch(
                ledgers,
                expected_batch_size=len(batch),
                allowed_schemas=(
                    {
                        DECISION_LEDGER_SCHEMA,
                        GUIDED_DECISION_LEDGER_SCHEMA,
                        K4_GUIDED_DECISION_LEDGER_SCHEMA,
                    }
                    if self.joint_training_config is not None
                    else {DECISION_LEDGER_SCHEMA}
                ),
            )
        )
        if "rm_scores" not in batch.batch:
            raise ValueError(
                "decision-ledger rollout is missing token-level rm_scores"
            )
        validate_decision_ledger_reward_rows(
            ledgers,
            reward_rows=batch.batch["rm_scores"].detach().cpu().tolist(),
            response_masks=batch.batch["response_mask"].detach().cpu().tolist(),
        )

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        custom_metrics_accumulator: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        sample_data_sources = []
        sample_rollout_sample_ids = []
        sample_images = []

        pad_token_id = self.tokenizer.pad_token_id
        skip_pad_tokens = self.config.trainer.get("skip_pad_tokens", True)
        validation_journal_dir = self.config.trainer.get(
            "validation_batch_journal_dir",
            None,
        )
        validation_journal_expected_rows = self.config.trainer.get(
            "validation_batch_journal_expected_rows",
            None,
        )
        visualization_sample_id = self.config.trainer.get(
            "validation_visualization_rollout_sample_id",
            None,
        )
        visualization_data_source = self.config.trainer.get(
            "validation_visualization_data_source",
            None,
        )
        visualization_seed = self.config.trainer.get(
            "validation_visualization_seed",
            None,
        )
        visualization_audit_dir = self.config.trainer.get(
            "validation_visualization_audit_dir",
            None,
        )
        rollout_browser_root = self.config.trainer.get(
            "validation_rollout_browser_dir",
            None,
        )
        rollout_browser_dir = (
            str(Path(str(rollout_browser_root)) / f"global_step_{self.global_steps}")
            if rollout_browser_root
            else None
        )
        rollout_browser_expected_rows = self.config.trainer.get(
            "validation_rollout_browser_expected_rows",
            None,
        )
        rollout_browser_policy_family = self.config.trainer.get(
            "validation_rollout_browser_policy_family",
            None,
        )
        rollout_browser_evaluation_id = self.config.trainer.get(
            "validation_rollout_browser_evaluation_id",
            None,
        )
        rollout_browser_checkpoint_identity = self.config.trainer.get(
            "validation_rollout_browser_checkpoint_identity",
            None,
        )
        rollout_browser_snapshot_identity = self.config.trainer.get(
            "validation_rollout_browser_snapshot_identity",
            None,
        )
        rollout_browser_source_step = self.config.trainer.get(
            "validation_rollout_browser_source_step",
            None,
        )
        has_sample_selector = bool(visualization_sample_id)
        has_source_seed_selector = (
            bool(visualization_data_source) and visualization_seed is not None
        )
        if (
            has_sample_selector == has_source_seed_selector
            or not visualization_audit_dir
        ) and any(
            (
                has_sample_selector,
                bool(visualization_data_source),
                visualization_seed is not None,
                bool(visualization_audit_dir),
            )
        ):
            raise ValueError(
                "validation visualization requires exactly one paired selector"
            )
        visualization_match_count = 0
        visualization_audit_count = 0
        rollout_browser_batch_count = 0
        if rollout_browser_dir:
            required_browser_values = {
                "expected_rows": rollout_browser_expected_rows,
                "policy_family": rollout_browser_policy_family,
                "evaluation_id": rollout_browser_evaluation_id,
                "checkpoint_identity": rollout_browser_checkpoint_identity,
            }
            missing_browser_values = [
                name for name, value in required_browser_values.items() if value is None
            ]
            if missing_browser_values:
                raise ValueError(
                    "validation rollout browser is missing config: "
                    f"{sorted(missing_browser_values)}"
                )
            if self.concat_multi_turn:
                raise ValueError(
                    "validation rollout browser requires no-concat validation"
                )
        if validation_journal_dir and self.concat_multi_turn:
            raise ValueError(
                "validation batch journaling requires no-concat validation"
            )
        completed_validation_batches = 0

        for validation_batch_index, test_data in enumerate(self.val_dataloader):
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )
            batch_seeds = (
                [int(value) for value in test_batch.non_tensor_batch["seed"]]
                if "seed" in test_batch.non_tensor_batch
                else [None] * len(test_batch)
            )
            if visualization_audit_dir:
                required_selector_fields = {"rollout_sample_id"}
                if has_source_seed_selector:
                    required_selector_fields |= {"data_source", "seed"}
                missing_selector_fields = required_selector_fields - set(
                    test_batch.non_tensor_batch
                )
                if missing_selector_fields:
                    raise ValueError(
                        "validation visualization is missing selector fields: "
                        f"{sorted(missing_selector_fields)}"
                    )
                selected = []
                for index, sample_id in enumerate(
                    test_batch.non_tensor_batch["rollout_sample_id"]
                ):
                    if has_sample_selector:
                        matches = str(sample_id) == str(
                            visualization_sample_id
                        )
                    else:
                        matches = (
                            str(test_batch.non_tensor_batch["data_source"][index])
                            == str(visualization_data_source)
                            and int(test_batch.non_tensor_batch["seed"][index])
                            == int(visualization_seed)
                        )
                    if matches:
                        selected.append(index)
                if not selected:
                    continue
                test_batch = test_batch.select_idxs(selected)
                visualization_match_count += len(test_batch)
            strict_canary_provenance = (
                self.joint_integration_gate is not None
                and self.joint_integration_gate.experiment_id in {183, 184, 185, 186, 187, 188}
            )
            if "data_source" not in test_batch.non_tensor_batch:
                if strict_canary_provenance:
                    raise ValueError(
                        f"ID{self.joint_integration_gate.experiment_id} "
                        "validation batch is missing data_source provenance"
                    )
                batch_data_sources = ["unknown"] * len(test_batch)
            else:
                batch_data_sources = test_batch.non_tensor_batch["data_source"]
            if "rollout_sample_id" not in test_batch.non_tensor_batch:
                if strict_canary_provenance:
                    raise ValueError(
                        f"ID{self.joint_integration_gate.experiment_id} "
                        "validation batch is missing stable "
                        "rollout_sample_id provenance"
                    )
                batch_sample_ids = test_batch.non_tensor_batch["uid"]
            else:
                batch_sample_ids = test_batch.non_tensor_batch[
                    "rollout_sample_id"
                ]
            sample_data_sources.extend(str(value) for value in batch_data_sources)
            sample_rollout_sample_ids.extend(
                str(value) for value in batch_sample_ids
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)

            if not self.concat_multi_turn:
                # we need to create group_idx, traj_idx for each traj in no-concat mode
                num_traj_per_sample = self.config.actor_rollout_ref.rollout.val_kwargs.n
                self._assign_group_and_traj_idx(test_gen_batch, num_traj_per_sample)
                batch_repeat_indices = [
                    int(value)
                    for value in test_gen_batch.non_tensor_batch[
                        "rollout_repeat_index"
                    ]
                ]
            else:
                batch_repeat_indices = [0] * len(test_gen_batch)

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )

            # In no-concat mode, save original uids before padding for filtering later
            if not self.concat_multi_turn:
                original_uids = set(test_gen_batch.non_tensor_batch["uid"])

            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.concat_multi_turn and pad_size:
                _assign_unique_validation_padding_identities(
                    test_gen_batch_padded,
                    pad_size,
                )
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            rollout_browser_turn_batch = None
            if self.concat_multi_turn:
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            else:
                # In no-concat mode, filter by uid since each input generates variable number of outputs
                # We need to keep only outputs whose uid is in the original (pre-padding) uid set
                valid_indices = [
                    i for i, uid in enumerate(test_output_gen_batch_padded.non_tensor_batch["group_idx"]) # uid in test_gen become group index in test_output_gen
                    if uid in original_uids
                ]
                test_output_gen_batch = test_output_gen_batch_padded.select_idxs(valid_indices)
                if rollout_browser_dir:
                    rollout_browser_turn_batch = test_output_gen_batch
                if visualization_audit_dir:
                    _dump_single_rollout_visualization_audit(
                        test_output_gen_batch,
                        visualization_audit_dir,
                    )
                    visualization_audit_count += 1
                # Concatenate multi-turn trajectories into single entries
                test_output_gen_batch = concat_val_multi_turn(test_output_gen_batch, test_gen_batch,self.tokenizer)
                # after this, we can assume no-concat mode and concat_multi_turn can be handled equally


            print("validation generation end")
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True
            # Store generated outputs
            
            inputs = test_batch.batch["prompts"]
            outputs = test_batch.batch["responses"]
            if skip_pad_tokens:
                inputs = self.tokenizer.batch_decode(
                    [s[-l:] if l else [] for s, l in zip(inputs.tolist(),  (inputs  != pad_token_id).sum(1).tolist())],
                    skip_special_tokens=False)
                outputs = self.tokenizer.batch_decode(
                    [s[:l]  if l else [] for s, l in zip(outputs.tolist(), (outputs != pad_token_id).sum(1).tolist())],
                    skip_special_tokens=False)
            else:
                inputs = self.tokenizer.batch_decode(inputs.tolist(), skip_special_tokens=False)
                outputs = self.tokenizer.batch_decode(outputs.tolist(), skip_special_tokens=False)
           
            sample_inputs.extend(inputs)
            sample_outputs.extend(outputs)

            # Extract images from non_tensor_batch (extra_fields are stored there)
            if "image_data" in test_batch.non_tensor_batch:
                batch_images = test_batch.non_tensor_batch["image_data"]
                sample_images.extend(batch_images.tolist() if hasattr(batch_images, 'tolist') else batch_images)
            else:
                sample_images.extend([None] * len(outputs))

            
            
            
            
            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            batch_reward_extra_infos: dict[str, list[Any]] = {
                "reward": list(scores),
            }
            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    batch_values = list(lst)
                    if len(batch_values) != len(scores):
                        raise ValueError(
                            "validation reward extra batch length mismatch: "
                            f"{key}"
                        )
                    batch_reward_extra_infos[key] = batch_values
                    reward_extra_infos_dict[key].extend(batch_values)

            if rollout_browser_dir:
                if rollout_browser_turn_batch is None:
                    raise RuntimeError(
                        "validation rollout browser lost pre-concat turn batch"
                    )
                trajectory_metadata = {}
                success_values = batch_reward_extra_infos.get("traj_success")
                if success_values is None:
                    raise ValueError(
                        "validation rollout browser requires traj_success reward metadata"
                    )
                for row_index, sample_id in enumerate(batch_sample_ids):
                    identity = (str(sample_id), int(batch_repeat_indices[row_index]))
                    if identity in trajectory_metadata:
                        raise ValueError(
                            "validation rollout browser metadata contains duplicate identity"
                        )
                    seed = batch_seeds[row_index]
                    if seed is None:
                        raise ValueError(
                            "validation rollout browser requires environment seed"
                        )
                    trajectory_metadata[identity] = {
                        "data_source": str(batch_data_sources[row_index]),
                        "seed": int(seed),
                        "split": "validation",
                        "reward": float(scores[row_index]),
                        "success": bool(success_values[row_index]),
                    }
                artifacts = _build_validation_rollout_browser_artifacts(
                    rollout_browser_turn_batch,
                    trajectory_metadata,
                    policy_family=str(rollout_browser_policy_family),
                )
                from nimloth.eval.rollout_browser import (
                    write_evaluation_browser_batch,
                )

                write_evaluation_browser_batch(
                    Path(str(rollout_browser_dir)),
                    artifacts,
                    batch_index=rollout_browser_batch_count,
                )
                rollout_browser_batch_count += 1

            # Add token_level_scores to batch for custom metrics computation
            test_batch.batch["token_level_scores"] = reward_tensor

            # Compute custom metrics for validation
            custom_val_metrics = compute_custom_metrics(test_batch, prefix="custom_metrics")
            for metric_name, metric_value in custom_val_metrics.items():
                custom_metrics_accumulator[metric_name].append(metric_value)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

            if validation_journal_dir:
                _write_validation_batch_journal(
                    journal_dir=validation_journal_dir,
                    global_step=int(self.global_steps),
                    batch_index=completed_validation_batches,
                    inputs=list(inputs),
                    outputs=list(outputs),
                    ground_truths=list(ground_truths),
                    scores=list(scores),
                    uids=[
                        str(value)
                        for value in test_batch.non_tensor_batch["uid"]
                    ],
                    data_sources=[str(value) for value in batch_data_sources],
                    rollout_sample_ids=[
                        str(value) for value in batch_sample_ids
                    ],
                    rollout_repeat_indices=batch_repeat_indices,
                    reward_extra_infos=batch_reward_extra_infos,
                )
                completed_validation_batches += 1

        if visualization_audit_dir and (
            visualization_match_count != 1 or visualization_audit_count != 1
        ):
            raise ValueError(
                "validation visualization did not produce exactly one rollout"
            )
        if rollout_browser_dir:
            from nimloth.eval.rollout_browser import finalize_evaluation_browser

            finalize_evaluation_browser(
                Path(str(rollout_browser_dir)),
                evaluation={
                    "evaluation_id": str(rollout_browser_evaluation_id),
                    "policy_family": str(rollout_browser_policy_family),
                    "global_step": int(self.global_steps),
                    "source_step": (
                        int(rollout_browser_source_step)
                        if rollout_browser_source_step is not None
                        else None
                    ),
                    "checkpoint_identity": (
                        f"{rollout_browser_checkpoint_identity}"
                        f"@global_step_{self.global_steps}"
                    ),
                    "snapshot_identity": rollout_browser_snapshot_identity,
                },
                expected_rollouts=int(rollout_browser_expected_rows),
                expected_batches=rollout_browser_batch_count,
            )
        if validation_journal_dir:
            if validation_journal_expected_rows is None:
                raise ValueError(
                    "validation journal expected row count is required"
                )
            _finalize_validation_batch_journal(
                journal_dir=validation_journal_dir,
                global_step=int(self.global_steps),
                expected_batch_count=completed_validation_batches,
                expected_row_count=int(validation_journal_expected_rows),
            )

        if self.config.trainer.get("replace_image_tokens_for_logging", False):
            sample_inputs = replace_image_tokens_for_logging(sample_inputs, processor=self.processor, tokenizer=self.tokenizer)
            sample_outputs = replace_image_tokens_for_logging(sample_outputs, processor=self.processor, tokenizer=self.tokenizer)
            
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, images=sample_images)
        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            validation_dump_extras = dict(reward_extra_infos_dict)
            validation_dump_extras["data_source"] = sample_data_sources
            validation_dump_extras["rollout_sample_id"] = (
                sample_rollout_sample_ids
            )
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                images=sample_images,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=validation_dump_extras,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else: 
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        # Add aggregated custom metrics to metric_dict
        for metric_name, values in custom_metrics_accumulator.items():
            if len(values) > 0:
                # Use mean aggregation for custom metrics
                metric_dict[f"custom_metrics/val/{metric_name.split('/')[-1]}"] = np.mean(values)

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        self.concat_multi_turn = True # whether to concat history in async rollout --> if not, one traj has multiple prompt-response pairs
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            if self.config.trainer.get("concat_multi_turn", True):
                from verl.experimental.agent_loop import AgentLoopManager
            else:
                from .agent_loop.agent_loop_no_concat import AgentLoopManager
                self.concat_multi_turn = False
            manager_kwargs: dict[str, Any] = {}
            if self.joint_training_config is not None:
                from vagen.joint_policy.bootstrap import (
                    build_initial_joint_snapshot_state,
                )

                manager_kwargs = {
                    "initial_frozen_q_snapshot_state": (
                        build_initial_joint_snapshot_state(
                            tokenizer=self.tokenizer,
                            policy_config=self.joint_policy_config,
                            training_config=self.joint_training_config,
                            k4_world_model_config=(
                                self.k4_world_model_training_config
                            ),
                        )
                    ),
                    "guided_draw_run_seed": self.joint_training_config.run_seed,
                }
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rm_wg=self.rm_wg,
                **manager_kwargs,
            )

    def _publish_joint_snapshot_after_update(self, batch: DataProto) -> dict[str, Any]:
        """Publish only after every replicated actor/critic rank completed."""

        if self.joint_training_config is None:
            raise RuntimeError("joint snapshot publication requires joint training")
        expected_source = int(batch.meta_info["joint_snapshot_source_step"])
        expected_version = int(batch.meta_info["joint_activation_version"])
        expected_snapshot = str(batch.meta_info["joint_snapshot_id"])
        exports = self.actor_rollout_wg.export_joint_critic_snapshot(
            {
                "source_step": expected_source + 1,
                "contract_id": str(batch.meta_info["joint_contract_id"]),
                "score_dtype": self.joint_policy_config.score_dtype,
            }
        )
        from vagen.joint_policy.update_transaction import (
            publish_replicated_joint_snapshot,
        )

        activated = publish_replicated_joint_snapshot(
            manager=self.async_rollout_manager,
            rank_exports=exports,
            expected_world_size=self.actor_rollout_wg.world_size,
            expected_active_snapshot_id=expected_snapshot,
            expected_active_source_step=expected_source,
            expected_activation_version=expected_version,
        )
        return activated

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        if self.joint_training_config is not None:
            dataloader_temp_path = f"{dataloader_local_path}.tmp.{os.getpid()}"
            try:
                with open(dataloader_temp_path, "wb") as handle:
                    torch.save(dataloader_state_dict, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(dataloader_temp_path, dataloader_local_path)
            finally:
                if os.path.exists(dataloader_temp_path):
                    os.unlink(dataloader_temp_path)
        else:
            torch.save(dataloader_state_dict, dataloader_local_path)

        if self.joint_training_config is not None:
            from vagen.joint_policy.checkpoint import (
                assemble_joint_checkpoint,
                save_atomic_joint_checkpoint,
                sha256_file,
            )

            owner_state = self.async_rollout_manager.frozen_q_checkpoint_state()
            active = owner_state["active_snapshot_state"]
            active_source_step = active.get(
                "source_step",
                active.get("snapshot_source_step"),
            )
            rank_exports = self.actor_rollout_wg.export_joint_checkpoint(
                {
                    "source_step": active_source_step,
                    "contract_id": active["contract_id"],
                    "score_dtype": active["score_dtype"],
                }
            )
            from vagen.joint_policy.training_contract import (
                joint_training_contract_id,
            )

            joint_payload = assemble_joint_checkpoint(
                global_step=self.global_steps,
                run_seed=self.joint_training_config.run_seed,
                rank_exports=rank_exports,
                owner_checkpoint_state=owner_state,
                expected_world_size=self.actor_rollout_wg.world_size,
                dataloader_sha256=sha256_file(dataloader_local_path),
                training_contract_id=joint_training_contract_id(
                    self.joint_training_config,
                    self.joint_policy_config,
                    self.k4_world_model_training_config,
                ),
            )
            save_atomic_joint_checkpoint(
                local_global_step_folder,
                joint_payload,
            )

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            if self.joint_training_config is not None:
                from vagen.joint_policy.checkpoint import (
                    find_latest_complete_joint_checkpoint,
                )

                global_step_folder = find_latest_complete_joint_checkpoint(
                    checkpoint_folder
                )
            else:
                global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
                if self.joint_training_config is not None:
                    from vagen.joint_policy.checkpoint import (
                        JOINT_COMPLETION_FILENAME,
                    )

                    marker = os.path.join(
                        global_step_folder,
                        JOINT_COMPLETION_FILENAME,
                    )
                    if not os.path.isfile(marker):
                        raise ValueError(
                            "joint resume_path is not a complete global-update checkpoint"
                        )
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )
        if self.joint_training_config is not None:
            from vagen.joint_policy.checkpoint import (
                load_complete_joint_checkpoint,
            )

            joint = load_complete_joint_checkpoint(global_step_folder)
            from vagen.joint_policy.training_contract import (
                joint_training_contract_id,
            )

            expected_training_contract_id = joint_training_contract_id(
                self.joint_training_config,
                self.joint_policy_config,
                self.k4_world_model_training_config,
            )
            if (
                self.joint_integration_gate is not None
                and self.joint_integration_gate.experiment_id in {184, 185, 186, 187}
            ):
                from dataclasses import replace
                from pathlib import Path

                active_transport = joint["frozen_q_owner"][
                    "active_snapshot_state"
                ]["transport_path"]
                source_snapshot_root = str(Path(active_transport).parents[1])
                source_world_model = replace(
                    self.k4_world_model_training_config,
                    snapshot_transport_root=source_snapshot_root,
                )
                expected_source_training_contract_id = (
                    joint_training_contract_id(
                        self.joint_training_config,
                        self.joint_policy_config,
                        source_world_model,
                    )
                )
                if (
                    joint["training_contract_id"]
                    != expected_source_training_contract_id
                ):
                    raise ValueError(
                        f"ID{self.joint_integration_gate.experiment_id} "
                        "source training contract mismatch"
                    )
                if self.joint_integration_gate.experiment_id == 184:
                    migration_marker = (
                        "ID184_TRAINING_CONTRACT_PATH_MIGRATION_OK"
                    )
                elif self.joint_integration_gate.experiment_id == 185:
                    migration_marker = (
                        "ID185_TRAINING_CONTRACT_PATH_MIGRATION_OK"
                    )
                elif self.joint_integration_gate.experiment_id == 186:
                    migration_marker = (
                        "ID186_TRAINING_CONTRACT_PATH_MIGRATION_OK"
                    )
                else:
                    migration_marker = (
                        "ID187_TRAINING_CONTRACT_PATH_MIGRATION_OK"
                    )
                print(
                    f"{migration_marker} source={source_snapshot_root} "
                    "destination="
                    f"{self.k4_world_model_training_config.snapshot_transport_root}"
                )
            elif (
                joint["training_contract_id"]
                != expected_training_contract_id
            ):
                raise ValueError("joint checkpoint training contract mismatch")
            if (
                joint["global_step"] != self.global_steps
                or joint["run_seed"] != self.joint_training_config.run_seed
                or joint["world_size"] != self.actor_rollout_wg.world_size
            ):
                raise ValueError("joint checkpoint run identity mismatch")
            restored_owner = self.async_rollout_manager.restore_frozen_q_checkpoint_state(
                joint["frozen_q_owner"]
            )
            restored_ranks = self.actor_rollout_wg.load_joint_checkpoint(
                joint["actor_critic"]
            )
            expected_ranks = list(range(self.actor_rollout_wg.world_size))
            if sorted(row["rank"] for row in restored_ranks) != expected_ranks:
                raise ValueError("joint checkpoint did not restore every actor rank")
            optimizer_field = (
                "planning_optimizer_fingerprint"
                if isinstance(
                    self.joint_policy_config,
                    K4MCTSGuidedPolicyConfig,
                )
                else "critic_optimizer_fingerprint"
            )
            for row in restored_ranks:
                if (
                    row["source_step"] != restored_owner["active_source_step"]
                    or row["snapshot_id"] != restored_owner["active_snapshot_id"]
                    or row["completed_updates"]
                    != restored_owner["activation_version"]
                    or row["optimizer_fingerprint"]
                    != joint["actor_critic"][optimizer_field]
                ):
                    raise ValueError("joint actor and frozen Q restore state mismatch")

        # A changed dataset cannot consume the source sampler cursor. ID184 and
        # ID186 phase1 are explicit exceptions: model/optimizer/RNG/joint state
        # remain exact while the new deterministic full-split sampler restarts.
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        dataloader_policy = str(
            self.config.trainer.get(
                "joint_dataloader_resume_policy",
                "exact",
            )
        )
        if dataloader_policy not in {"exact", "reset"}:
            raise ValueError("unsupported joint dataloader resume policy")
        if dataloader_policy == "reset":
            id184_reset = (
                self.joint_integration_gate is not None
                and self.joint_integration_gate.experiment_id == 184
                and self.global_steps == 10
            )
            id186_reset = (
                self.joint_integration_gate is not None
                and self.joint_integration_gate.experiment_id == 186
                and self.joint_integration_gate.phase == "resume_20_to_30"
                and self.global_steps == 20
            )
            if (
                self.joint_training_config is None
                or not (id184_reset or id186_reset)
                or not os.path.isfile(dataloader_local_path)
            ):
                raise ValueError(
                    "joint dataloader reset is restricted to an approved "
                    "complete continuation checkpoint"
                )
            if id184_reset:
                print("ID184_DATALOADER_RESET_OK global_step=10")
            else:
                print("ID186_DATALOADER_RESET_OK global_step=20")
        elif os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(
                dataloader_local_path,
                weights_only=False,
            )
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            if self.joint_training_config is not None:
                raise ValueError(
                    f"joint checkpoint is missing dataloader state: {dataloader_local_path}"
                )
            print(
                f"Warning: No dataloader state found at {dataloader_local_path}, "
                "will start from scratch"
            )

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        if self.joint_integration_gate is not None and (
            self.joint_integration_gate.experiment_id in {183, 184, 185, 186, 187, 188}
        ):
            if self.joint_integration_gate.experiment_id == 183:
                expected_loaded_step = (
                    0
                    if self.joint_integration_gate.phase == "train_to_5"
                    else 5
                )
            elif self.joint_integration_gate.experiment_id == 184:
                expected_loaded_step = 10
            elif self.joint_integration_gate.experiment_id in {185, 187}:
                expected_loaded_step = 20
            elif self.joint_integration_gate.experiment_id == 188:
                expected_loaded_step = 0
            elif self.joint_integration_gate.phase == "resume_20_to_30":
                expected_loaded_step = 20
            else:
                expected_loaded_step = 30
            if self.global_steps != expected_loaded_step:
                raise ValueError(
                    f"ID{self.joint_integration_gate.experiment_id} loaded "
                    "an unexpected checkpoint boundary"
                )
            if self.joint_integration_gate.phase == "resume_to_10":
                print("ID183_K4_CANARY_RESUME_OK global_step=5")
            elif self.joint_integration_gate.phase == "resume_10_to_20":
                print("ID184_K4_CONTINUE_RESUME_OK global_step=10")
            elif self.joint_integration_gate.phase == "full_eval_test300":
                print("ID185_K4_FULL_EVAL_RESTORE_OK global_step=20")
            elif self.joint_integration_gate.phase == "visualize_one":
                print("ID185_K4_VISUALIZATION_RESTORE_OK global_step=20")
            elif self.joint_integration_gate.phase == "source20_visualize_one":
                print("ID187_K4_SOURCE20_RESTORE_OK global_step=20")
            elif self.joint_integration_gate.phase == "step0_visualize_one":
                print("ID188_K4_STEP0_BOOTSTRAP_OK global_step=0")
            elif self.joint_integration_gate.phase in {
                "resume_20_to_30",
                "resume_30_to_40",
            }:
                print(
                    "ID186_K4_CONTINUE_RESUME_OK "
                    f"global_step={self.global_steps}"
                )

        if (
            self.joint_integration_gate is not None
            and self.joint_integration_gate.phase == "restore_only"
        ):
            if self.global_steps != self.total_training_steps:
                raise ValueError(
                    "restore-only gate did not load its complete target step"
                )
            print(
                f"ID{self.joint_integration_gate.experiment_id}_"
                "K4_FRESH_RESTORE_ONLY_ALL_OK "
                f"global_step={self.global_steps}"
            )
            self._flush_image_dumps()
            self._hf_upload_manager.flush()
            return

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._flush_image_dumps()
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                if not self.concat_multi_turn:
                    # we need to create group_idx, traj_idx for each traj in no-concat mode
                    num_traj_per_sample = self.config.actor_rollout_ref.rollout.n
                    self._assign_group_and_traj_idx(gen_batch_output, num_traj_per_sample)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if not self.concat_multi_turn:
                            raise NotImplementedError("REMAX advantage estimation is not supported in no-concat mode yet.")
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    if self.concat_multi_turn:
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)
                    else:
                        # In no-concat mode, each trajectory has multiple prompt-response pairs.
                        # We need to re-generate batch to align with gen_batch_output.
                        batch = self._post_process_no_concat_batch(batch, gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    if not self.concat_multi_turn:
                        self._validate_decision_ledger_batch(batch, metrics)
                    if self.joint_training_config is not None:
                        from vagen.joint_policy.training_batch import (
                            prepare_joint_training_batch,
                        )

                        prepare_joint_training_batch(
                            batch,
                            config=self.joint_training_config,
                        )
                        batch.meta_info["temperature"] = float(
                            self.config.actor_rollout_ref.rollout.temperature
                        )

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        if not self.concat_multi_turn: # pad to divisor of dp_size
                            divisor_size = self.actor_rollout_wg.world_size
                            batch_size = len(batch.batch["attention_mask"])
                            batch, pad_size = pad_dataproto_to_divisor(batch, divisor_size)
                            if self.joint_training_config is not None:
                                from vagen.joint_policy.training_batch import (
                                    mark_joint_padding_invalid,
                                )

                                mark_joint_padding_invalid(batch, pad_size)
                            print(f"Pad {pad_size} samples to make batch size {batch_size} divisible by {divisor_size} dp_workers")
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if self.joint_training_config is not None:
                        # Guided behavior log-prob is persisted in the rollout
                        # ledger; stock token old-log-prob replay is unrelated.
                        pass
                    elif bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    if self.joint_training_config is None:
                        assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                            if self.joint_training_config is not None:
                                batch.batch["joint_reference_token_log_probs"] = (
                                    batch.batch["ref_log_prob"]
                                )

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        if not self.concat_multi_turn and parse_decision_ledger_enabled(
                            self.config.get("decision_ledger")
                        ):
                            validate_decision_ledger_reward_rows(
                                batch.non_tensor_batch["decision_ledger"],
                                reward_rows=reward_tensor.detach().cpu().tolist(),
                                response_masks=batch.batch["response_mask"].detach().cpu().tolist(),
                            )
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            self.joint_training_config is None
                            and rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        if self.joint_training_config is None:
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                    if (
                        self.joint_training_config is None
                        and self.config.algorithm.adv_estimator in ["no_concat_gae_last", "no_concat_gae_first"]
                    ):
                        batch.batch["value_mask"] = compute_value_mask(batch)

                    # compute custom metrics
                    with marked_timer("custom_metrics", timing_raw, color="magenta"):
                        if self.joint_training_config is None:
                            custom_train_metrics = compute_custom_metrics(batch, prefix="custom_metrics/train")
                            metrics.update(custom_train_metrics)

                    
                    
                    # filter the training batch for effective update (Refer to STARPO-S and DAPO)
                    if self.config.filter.get("enable", False):
                        batch,metrics = FILTER_REGISTRY.get(self.config.filter.name)(batch, metrics,**self.config.filter.filter_kwargs)
                        if self.config.trainer.balance_batch:
                            # re-balance after filtering
                            divisor_size = self.actor_rollout_wg.world_size
                            batch_size = len(batch.batch["attention_mask"])
                            batch, pad_size = pad_dataproto_to_divisor(batch, divisor_size)
                            print(f"After filtering: Pad {pad_size} samples to make batch size {batch_size} divisible by {divisor_size} dp_workers")
                            self._balance_batch(batch, metrics=metrics, logging_prefix="filtered_global_seqlen")
                    
                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                        if self.joint_training_config is not None:
                            activated = self._publish_joint_snapshot_after_update(batch)
                            metrics.update(
                                {
                                    "joint/active_source_step": float(
                                        activated["active_source_step"]
                                    ),
                                    "joint/activation_version": float(
                                        activated["activation_version"]
                                    ),
                                }
                            )

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                should_save_ckpt = self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                )
                should_upload_hf = self._hf_upload_manager.should_upload(self.global_steps)

                if should_save_ckpt or should_upload_hf:
                    # Flush pending HF uploads before saving to avoid conflicts
                    # with checkpoint deletion (max_actor_ckpt_to_keep)
                    self._hf_upload_manager.flush()
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                if should_upload_hf:
                    self._hf_upload_manager.maybe_upload(self.global_steps)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                if self.joint_training_config is not None:
                    from vagen.joint_policy.training_batch import joint_data_metrics

                    metrics.update(joint_data_metrics(batch))
                else:
                    metrics.update(
                        compute_data_metrics(
                            batch=batch,
                            use_critic=self.use_critic,
                        )
                    )
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    self._flush_image_dumps()
                    self._hf_upload_manager.flush()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
