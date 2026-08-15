"""Strict training contract and frozen-value targets for guided joint PPO."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

from .contract import (
    FrozenQGuidedPolicyConfig,
    GuidedPolicyBehaviorRecord,
    guided_log_probs_reference,
)
from .planning_contract import (
    K4MCTSGuidedBehaviorRecord,
    K4MCTSGuidedPolicyConfig,
    k4_guided_log_probs_reference,
)

_IMPLEMENTATION = "replicated_joint_update_v1"
_STOP_REASONS = frozenset(
    {
        "continue",
        "success",
        "task_failure",
        "environment_failure",
        "infrastructure_truncation",
    }
)
_CONFIG_FIELDS = frozenset(
    {
        "implementation",
        "run_seed",
        "gamma",
        "gae_lambda",
        "ppo_clip_ratio",
        "normalize_advantages",
        "token_kl_coefficient",
        "token_kl_type",
        "guided_entropy_coefficient",
        "checkpoint_frequency",
        "actor_optimizer",
        "critic_checkpoint",
        "initial_snapshot_source_step",
        "critic_qwen_hidden_dim",
        "critic_grid_tokens",
        "critic_state_dim",
        "critic_action_count",
        "critic_huber_delta",
        "critic_grad_clip",
        "critic_optimizer",
    }
)
_ROW_FIELDS = frozenset(
    {
        "group_idx",
        "traj_idx",
        "guided_turn_index",
        "rollout_stop_reason",
        "decision_ledger",
    }
)
_LEDGER_FIELDS = frozenset(
    {
        "schema",
        "action_space",
        "action_space_names",
        "executed_action_ids",
        "executed_action_names",
        "decision_sources",
        "decision_is_policy_sampled",
        "env_turn_reward",
        "env_terminated",
        "rollout_truncated",
        "format_valid",
        "snapshot_id",
        "contract_id",
        "behavior_record_id",
        "behavior_record",
    }
)


@dataclass(frozen=True)
class JointCriticOptimizerConfig:
    """Explicit optimizer values for the replicated current critic."""

    name: str
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "JointCriticOptimizerConfig":
        if not isinstance(raw, Mapping):
            raise ValueError("joint critic optimizer must be a mapping")
        fields = frozenset(cls.__dataclass_fields__)
        missing = fields - set(raw)
        if missing:
            raise ValueError(
                f"joint critic optimizer is missing fields: {sorted(missing)}"
            )
        unexpected = set(raw) - fields
        if unexpected:
            raise ValueError(
                "joint critic optimizer has unexpected fields: "
                f"{sorted(unexpected)}"
            )
        if raw["name"] != "adamw":
            raise ValueError("joint critic optimizer supports only explicit adamw")
        betas_raw = _plain_sequence(raw["betas"], "critic optimizer betas")
        if len(betas_raw) != 2:
            raise ValueError("joint critic optimizer betas must contain two values")
        betas = tuple(_finite_float(value, "critic optimizer beta") for value in betas_raw)
        if any(value < 0.0 or value >= 1.0 for value in betas):
            raise ValueError("joint critic optimizer betas must be in [0, 1)")
        return cls(
            name="adamw",
            lr=_positive_float(raw["lr"], "critic optimizer lr"),
            betas=(betas[0], betas[1]),
            eps=_positive_float(raw["eps"], "critic optimizer eps"),
            weight_decay=_nonnegative_float(
                raw["weight_decay"],
                "critic optimizer weight_decay",
            ),
        )


@dataclass(frozen=True)
class JointActorOptimizerConfig:
    """Explicit FSDP actor AdamW, clipping, and scheduler values."""

    name: str
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    grad_clip: float
    lr_scheduler_type: str
    lr_warmup_steps: int
    lr_warmup_steps_ratio: float
    min_lr_ratio: float | None
    num_cycles: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "JointActorOptimizerConfig":
        if not isinstance(raw, Mapping):
            raise ValueError("joint actor optimizer must be a mapping")
        fields = frozenset(cls.__dataclass_fields__)
        if set(raw) != fields:
            raise ValueError(
                "joint actor optimizer fields are invalid: "
                f"missing={sorted(fields - set(raw))}, "
                f"unexpected={sorted(set(raw) - fields)}"
            )
        if raw["name"] != "adamw":
            raise ValueError("joint actor optimizer supports only explicit adamw")
        betas_raw = _plain_sequence(raw["betas"], "actor optimizer betas")
        if len(betas_raw) != 2:
            raise ValueError("joint actor optimizer betas must contain two values")
        betas = tuple(_finite_float(value, "actor optimizer beta") for value in betas_raw)
        if any(value < 0.0 or value >= 1.0 for value in betas):
            raise ValueError("joint actor optimizer betas must be in [0, 1)")
        scheduler = raw["lr_scheduler_type"]
        if scheduler not in {"constant", "cosine"}:
            raise ValueError(
                "joint actor lr_scheduler_type must be constant or cosine"
            )
        warmup_ratio = _nonnegative_float(
            raw["lr_warmup_steps_ratio"],
            "actor lr_warmup_steps_ratio",
        )
        if warmup_ratio > 1.0:
            raise ValueError("joint actor lr_warmup_steps_ratio must be <= 1")
        warmup_steps = _nonnegative_int(
            raw["lr_warmup_steps"],
            "actor lr_warmup_steps",
        )
        if warmup_steps > 0 and warmup_ratio > 0.0:
            raise ValueError(
                "joint actor warmup steps and ratio cannot both be positive"
            )
        min_lr_raw = raw["min_lr_ratio"]
        min_lr_ratio = None
        if min_lr_raw is not None:
            min_lr_ratio = _nonnegative_float(
                min_lr_raw,
                "actor min_lr_ratio",
            )
            if min_lr_ratio > 1.0:
                raise ValueError("joint actor min_lr_ratio must be <= 1")
        return cls(
            name="adamw",
            lr=_positive_float(raw["lr"], "actor optimizer lr"),
            betas=(betas[0], betas[1]),
            eps=_positive_float(raw["eps"], "actor optimizer eps"),
            weight_decay=_nonnegative_float(
                raw["weight_decay"],
                "actor optimizer weight_decay",
            ),
            grad_clip=_positive_float(raw["grad_clip"], "actor grad_clip"),
            lr_scheduler_type=scheduler,
            lr_warmup_steps=warmup_steps,
            lr_warmup_steps_ratio=warmup_ratio,
            min_lr_ratio=min_lr_ratio,
            num_cycles=_positive_float(raw["num_cycles"], "actor num_cycles"),
        )


@dataclass(frozen=True)
class JointTrainingConfig:
    """No-default training semantics required before joint PPO may run."""

    implementation: str
    run_seed: int
    gamma: float
    gae_lambda: float
    ppo_clip_ratio: float
    normalize_advantages: bool
    token_kl_coefficient: float
    token_kl_type: str
    guided_entropy_coefficient: float
    checkpoint_frequency: int
    actor_optimizer: JointActorOptimizerConfig
    critic_checkpoint: str
    initial_snapshot_source_step: int
    critic_qwen_hidden_dim: int
    critic_grid_tokens: int
    critic_state_dim: int
    critic_action_count: int
    critic_huber_delta: float
    critic_grad_clip: float
    critic_optimizer: JointCriticOptimizerConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "JointTrainingConfig":
        if not isinstance(raw, Mapping):
            raise ValueError("joint training config must be a mapping")
        missing = _CONFIG_FIELDS - set(raw)
        if missing:
            raise ValueError(f"joint training config is missing fields: {sorted(missing)}")
        unexpected = set(raw) - _CONFIG_FIELDS
        if unexpected:
            raise ValueError(
                f"joint training config has unexpected fields: {sorted(unexpected)}"
            )
        if raw["implementation"] != _IMPLEMENTATION:
            raise ValueError(
                f"unsupported joint training implementation: {raw['implementation']!r}"
            )
        gamma = _finite_float(raw["gamma"], "gamma")
        if gamma <= 0.0 or gamma > 1.0:
            raise ValueError("joint training gamma must be in (0, 1]")
        gae_lambda = _finite_float(raw["gae_lambda"], "gae_lambda")
        if gae_lambda < 0.0 or gae_lambda > 1.0:
            raise ValueError("joint training gae_lambda must be in [0, 1]")
        clip_ratio = _finite_float(raw["ppo_clip_ratio"], "ppo_clip_ratio")
        if clip_ratio <= 0.0 or clip_ratio >= 1.0:
            raise ValueError("joint training ppo_clip_ratio must be in (0, 1)")
        if raw["normalize_advantages"] is not True:
            raise ValueError(
                "joint training normalize_advantages must be explicit true"
            )
        checkpoint = raw["critic_checkpoint"]
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("joint training critic_checkpoint must be non-empty str")
        return cls(
            implementation=_IMPLEMENTATION,
            run_seed=_nonnegative_int(raw["run_seed"], "run_seed"),
            gamma=gamma,
            gae_lambda=gae_lambda,
            ppo_clip_ratio=clip_ratio,
            normalize_advantages=True,
            token_kl_coefficient=_nonnegative_float(
                raw["token_kl_coefficient"],
                "token_kl_coefficient",
            ),
            token_kl_type=_token_kl_type(raw["token_kl_type"]),
            guided_entropy_coefficient=_nonnegative_float(
                raw["guided_entropy_coefficient"],
                "guided_entropy_coefficient",
            ),
            checkpoint_frequency=_positive_int(
                raw["checkpoint_frequency"],
                "checkpoint_frequency",
            ),
            actor_optimizer=JointActorOptimizerConfig.from_mapping(
                raw["actor_optimizer"]
            ),
            critic_checkpoint=checkpoint,
            initial_snapshot_source_step=_nonnegative_int(
                raw["initial_snapshot_source_step"],
                "initial_snapshot_source_step",
            ),
            critic_qwen_hidden_dim=_positive_int(
                raw["critic_qwen_hidden_dim"],
                "critic_qwen_hidden_dim",
            ),
            critic_grid_tokens=_positive_int(
                raw["critic_grid_tokens"],
                "critic_grid_tokens",
            ),
            critic_state_dim=_positive_int(
                raw["critic_state_dim"],
                "critic_state_dim",
            ),
            critic_action_count=_positive_int(
                raw["critic_action_count"],
                "critic_action_count",
            ),
            critic_huber_delta=_positive_float(
                raw["critic_huber_delta"],
                "critic_huber_delta",
            ),
            critic_grad_clip=_positive_float(
                raw["critic_grad_clip"],
                "critic_grad_clip",
            ),
            critic_optimizer=JointCriticOptimizerConfig.from_mapping(
                raw["critic_optimizer"]
            ),
        )


@dataclass(frozen=True)
class JointTrainingTargets:
    """Driver-compiled actor and selected-action critic targets in input order."""

    discounted_returns: tuple[float, ...]
    frozen_state_values: tuple[float, ...]
    raw_advantages: tuple[float, ...]
    advantages: tuple[float, ...]
    executed_action_ids: tuple[int, ...]
    snapshot_id: str
    contract_id: str


def joint_training_contract_id(
    training: JointTrainingConfig,
    policy: Any,
) -> str:
    """Bind every explicit optimizer/loss value to the policy contract."""

    if not isinstance(training, JointTrainingConfig):
        raise TypeError("joint training contract ID requires JointTrainingConfig")
    if not isinstance(
        policy,
        (FrozenQGuidedPolicyConfig, K4MCTSGuidedPolicyConfig),
    ):
        raise TypeError(
            "joint training contract ID requires a supported guided policy config"
        )
    payload = {
        "training": asdict(training),
        "policy": asdict(policy),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def parse_joint_training_section(raw: Mapping[str, Any]) -> JointTrainingConfig | None:
    """Parse the training opt-in without inventing disabled or enabled defaults."""

    if not isinstance(raw, Mapping):
        raise ValueError("joint_training section must be a mapping")
    if "enabled" not in raw:
        raise ValueError("joint_training section is missing fields: ['enabled']")
    if not isinstance(raw["enabled"], bool):
        raise ValueError("joint_training.enabled must be explicit bool")
    if not raw["enabled"]:
        unexpected = set(raw) - (_CONFIG_FIELDS | {"enabled"})
        populated = {
            field for field in _CONFIG_FIELDS if field in raw and raw[field] is not None
        }
        if unexpected or populated:
            raise ValueError(
                "disabled joint_training section has unexpected populated fields: "
                f"{sorted(unexpected | populated)}"
            )
        return None
    missing = _CONFIG_FIELDS - set(raw)
    if missing:
        raise ValueError(
            f"joint_training section is missing fields: {sorted(missing)}"
        )
    unexpected = set(raw) - (_CONFIG_FIELDS | {"enabled"})
    if unexpected:
        raise ValueError(
            f"joint_training section has unexpected fields: {sorted(unexpected)}"
        )
    return JointTrainingConfig.from_mapping(
        {field: raw[field] for field in _CONFIG_FIELDS}
    )


def compile_outcome_returns_and_frozen_v_gae(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: JointTrainingConfig,
    selected_action_q_baseline: bool = False,
) -> JointTrainingTargets:
    """Compile discounted environment rewards and behavior-frozen GAE.

    Reward shaping is an environment/config policy and may assign finite rewards
    to any real turn. Infrastructure truncation is invalid data and never becomes
    a policy outcome.
    """

    if selected_action_q_baseline:
        raise TypeError(
            "selected-action Q baseline is action-dependent; use frozen state value"
        )
    if selected_action_q_baseline is not False:
        raise TypeError("selected_action_q_baseline must be explicit bool")
    if not isinstance(config, JointTrainingConfig):
        raise TypeError("joint target compiler requires JointTrainingConfig")
    records = list(_plain_sequence(rows, "joint training rows"))
    if not records:
        raise ValueError("joint training rows must be non-empty")

    prepared: list[dict[str, Any]] = []
    snapshot_id = None
    contract_id = None
    for input_index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise ValueError(f"joint training row {input_index} must be a mapping")
        missing = _ROW_FIELDS - set(raw)
        if missing:
            raise ValueError(
                f"joint training row is missing fields: {sorted(missing)}"
            )
        unexpected = set(raw) - _ROW_FIELDS
        if unexpected:
            raise ValueError(
                f"joint training row has unexpected fields: {sorted(unexpected)}"
            )
        group_idx = raw["group_idx"]
        if isinstance(group_idx, bool) or not isinstance(group_idx, (str, int)):
            raise ValueError("joint training group_idx must be str or int")
        if isinstance(group_idx, str) and not group_idx:
            raise ValueError("joint training group_idx must be non-empty")
        traj_idx = _nonnegative_int(raw["traj_idx"], "traj_idx")
        turn_index = _nonnegative_int(
            raw["guided_turn_index"],
            "guided_turn_index",
        )
        stop_reason = raw["rollout_stop_reason"]
        if stop_reason not in _STOP_REASONS:
            raise ValueError(
                f"unsupported rollout stop reason: {stop_reason!r}"
            )
        if stop_reason == "infrastructure_truncation":
            raise ValueError(
                "infrastructure truncation is invalid training data and must fail closed"
            )
        ledger = _validated_guided_ledger(raw["decision_ledger"])
        behavior = _guided_behavior_record(ledger)
        if len(behavior.action_space_names) != config.critic_action_count:
            raise ValueError(
                "joint training action count does not match critic contract"
            )
        if snapshot_id is None:
            snapshot_id = behavior.snapshot_id
            contract_id = behavior.contract_id
        elif snapshot_id != behavior.snapshot_id or contract_id != behavior.contract_id:
            raise ValueError(
                "joint training batch must use one frozen snapshot and contract"
            )
        if isinstance(behavior, K4MCTSGuidedBehaviorRecord):
            guided_log_probs = k4_guided_log_probs_reference(
                behavior.prior_logits,
                behavior.planner_root_mean_values,
                behavior.policy_config,
            )[1]
            direct_all_action_q = behavior.direct_all_action_q
        else:
            guided_log_probs = guided_log_probs_reference(
                behavior.prior_logits,
                behavior.frozen_all_action_q,
                behavior.policy_config,
            )[1]
            direct_all_action_q = behavior.frozen_all_action_q
        state_value = sum(
            math.exp(log_prob) * q_value
            for log_prob, q_value in zip(
                guided_log_probs,
                direct_all_action_q,
                strict=True,
            )
        )
        if not math.isfinite(state_value):
            raise ValueError("joint training frozen state value must be finite")
        prepared.append(
            {
                "input_index": input_index,
                "trajectory": (str(group_idx), traj_idx),
                "turn_index": turn_index,
                "stop_reason": stop_reason,
                "reward": float(ledger["env_turn_reward"]),
                "terminated": ledger["env_terminated"],
                "truncated": ledger["rollout_truncated"],
                "state_value": state_value,
                "action_id": behavior.guided_action_id,
            }
        )

    trajectories: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in prepared:
        trajectories.setdefault(record["trajectory"], []).append(record)

    returns = [0.0] * len(prepared)
    raw_advantages = [0.0] * len(prepared)
    for identity, trajectory in trajectories.items():
        trajectory.sort(key=lambda item: item["turn_index"])
        expected_turns = list(range(len(trajectory)))
        actual_turns = [item["turn_index"] for item in trajectory]
        if actual_turns != expected_turns:
            raise ValueError(
                "joint training trajectory turn indices must be contiguous from zero: "
                f"trajectory={identity}, actual={actual_turns}"
            )
        _validate_trajectory_sequence(trajectory)

        running_return = 0.0
        running_advantage = 0.0
        for reverse_index in range(len(trajectory) - 1, -1, -1):
            item = trajectory[reverse_index]
            final = reverse_index == len(trajectory) - 1
            next_value = 0.0 if final else trajectory[reverse_index + 1]["state_value"]
            next_advantage = 0.0 if final else running_advantage
            running_return = item["reward"] + (
                0.0 if final else config.gamma * running_return
            )
            delta = (
                item["reward"]
                + config.gamma * next_value
                - item["state_value"]
            )
            running_advantage = delta + (
                0.0
                if final
                else config.gamma * config.gae_lambda * next_advantage
            )
            input_index = item["input_index"]
            returns[input_index] = _finite_float(
                running_return,
                "discounted return",
            )
            raw_advantages[input_index] = _finite_float(
                running_advantage,
                "frozen-V GAE",
            )

    advantages = _normalize(raw_advantages)
    return JointTrainingTargets(
        discounted_returns=tuple(returns),
        frozen_state_values=tuple(item["state_value"] for item in prepared),
        raw_advantages=tuple(raw_advantages),
        advantages=tuple(advantages),
        executed_action_ids=tuple(item["action_id"] for item in prepared),
        snapshot_id=str(snapshot_id),
        contract_id=str(contract_id),
    )


def _validated_guided_ledger(raw: Any) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("joint training decision ledger must be a mapping")
    missing = _LEDGER_FIELDS - set(raw)
    if missing:
        raise ValueError(f"guided decision ledger is missing fields: {sorted(missing)}")
    unexpected = set(raw) - _LEDGER_FIELDS
    if unexpected:
        raise ValueError(
            f"guided decision ledger has unexpected fields: {sorted(unexpected)}"
        )
    if raw["schema"] not in {
        "vagen_decision_ledger_v2_frozen_q_guided",
        "vagen_decision_ledger_v3_k4_mcts_guided",
    }:
        raise ValueError("joint training requires a guided decision ledger schema")
    behavior = _guided_behavior_record(raw)
    if raw["behavior_record_id"] != behavior.record_id():
        raise ValueError("guided decision ledger behavior identity mismatch")
    if raw["snapshot_id"] != behavior.snapshot_id:
        raise ValueError("guided decision ledger snapshot identity mismatch")
    if raw["contract_id"] != behavior.contract_id:
        raise ValueError("guided decision ledger contract identity mismatch")
    if raw["executed_action_ids"] != [behavior.guided_action_id]:
        raise ValueError("guided decision ledger executed action mismatch")
    if raw["executed_action_names"] != [
        behavior.action_space_names[behavior.guided_action_id]
    ]:
        raise ValueError("guided decision ledger executed action name mismatch")
    expected_source = (
        "k4_mcts_guided"
        if isinstance(behavior, K4MCTSGuidedBehaviorRecord)
        else "frozen_q_guided"
    )
    if raw["decision_sources"] != [expected_source]:
        raise ValueError("guided decision ledger source mismatch")
    if raw["decision_is_policy_sampled"] != [True]:
        raise ValueError("guided decision ledger policy ownership mismatch")
    for field in ("env_terminated", "rollout_truncated", "format_valid"):
        if not isinstance(raw[field], bool):
            raise ValueError(f"guided decision ledger {field} must be bool")
    if raw["env_terminated"] and raw["rollout_truncated"]:
        raise ValueError("guided turn cannot be terminated and truncated")
    reward = raw["env_turn_reward"]
    if isinstance(reward, bool) or not isinstance(reward, Real) or not math.isfinite(float(reward)):
        raise ValueError("guided decision ledger reward must be finite")
    if raw["format_valid"] is not True:
        raise ValueError("joint training rejects format-invalid executed actions")
    return raw


def _guided_behavior_record(
    ledger: Mapping[str, Any],
) -> GuidedPolicyBehaviorRecord | K4MCTSGuidedBehaviorRecord:
    if ledger.get("schema") == "vagen_decision_ledger_v3_k4_mcts_guided":
        return K4MCTSGuidedBehaviorRecord.from_mapping(ledger["behavior_record"])
    return GuidedPolicyBehaviorRecord.from_mapping(ledger["behavior_record"])


def _validate_trajectory_sequence(trajectory: list[dict[str, Any]]) -> None:
    for item in trajectory[:-1]:
        if item["stop_reason"] != "continue":
            raise ValueError("only the final trajectory turn may carry an outcome")
        if item["terminated"] or item["truncated"]:
            raise ValueError("non-final trajectory turn cannot be terminal or truncated")
    final = trajectory[-1]
    reason = final["stop_reason"]
    if reason == "continue":
        raise ValueError("joint training trajectory is missing a final outcome")
    if reason == "success":
        if not final["terminated"] or final["truncated"]:
            raise ValueError("success must be an environment terminal outcome")
    elif reason == "environment_failure":
        if not final["terminated"] or final["truncated"]:
            raise ValueError("environment failure must be terminal")
    elif reason == "task_failure":
        if final["terminated"] or not final["truncated"]:
            raise ValueError(
                "task failure must preserve environment non-terminal/truncated facts"
            )
    else:
        raise ValueError(f"invalid final rollout outcome: {reason!r}")


def _normalize(values: Sequence[float]) -> list[float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if not math.isfinite(variance):
        raise ValueError("joint training advantage variance must be finite")
    if variance <= 1e-16:
        return [0.0 for _ in values]
    scale = math.sqrt(variance)
    normalized = [(value - mean) / scale for value in values]
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("normalized joint training advantages must be finite")
    return normalized


def _plain_sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a plain sequence")
    return value


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"joint training {field} must be a finite real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"joint training {field} must be finite")
    return 0.0 if result == 0.0 else result


def _positive_float(value: Any, field: str) -> float:
    result = _finite_float(value, field)
    if result <= 0.0:
        raise ValueError(f"joint training {field} must be positive")
    return result


def _nonnegative_float(value: Any, field: str) -> float:
    result = _finite_float(value, field)
    if result < 0.0:
        raise ValueError(f"joint training {field} must be non-negative")
    return result


def _token_kl_type(value: Any) -> str:
    if value != "low_var_kl":
        raise ValueError("joint training token_kl_type supports only low_var_kl")
    return "low_var_kl"


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"joint training {field} must be a non-negative int")
    return value


def _positive_int(value: Any, field: str) -> int:
    result = _nonnegative_int(value, field)
    if result < 1:
        raise ValueError(f"joint training {field} must be positive")
    return result


__all__ = [
    "JointCriticOptimizerConfig",
    "JointTrainingConfig",
    "JointTrainingTargets",
    "compile_outcome_returns_and_frozen_v_gae",
    "parse_joint_training_section",
]
