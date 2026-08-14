"""Contracts for VAGEN joint policies."""

from .contract import (
    GUIDED_BEHAVIOR_SCHEMA,
    FrozenQGuidedPolicyConfig,
    GuidedPolicyBehaviorRecord,
    guided_log_probs_reference,
    parse_joint_policy_section,
)
from .critic_loss import SelectedActionHuberLoss, selected_action_huber_loss
from .execution import (
    GUIDED_ACTION_EXECUTION_SCHEMA,
    GuidedActionExecutionRequest,
    validate_guided_action_execution_result,
)
from .replay import replay_guided_behavior_log_probs
from .sampling import (
    GUIDED_ACTION_DRAW_KEY_SCHEMA,
    GUIDED_ACTION_DRAW_SCHEMA,
    GuidedActionDrawCoordinator,
    GuidedActionDrawKey,
    GuidedPolicyActionDrawRecord,
    sample_frozen_q_guided_action,
)
from .terminal_state import TERMINAL_STATE_TRACE_SCHEMA, TerminalStateTrace
from .torch_policy import frozen_q_guided_log_probs
from .training_contract import (
    JointActorOptimizerConfig,
    JointCriticOptimizerConfig,
    JointTrainingConfig,
    JointTrainingTargets,
    compile_outcome_returns_and_frozen_v_gae,
    joint_training_contract_id,
    parse_joint_training_section,
)

__all__ = [
    "GUIDED_ACTION_DRAW_KEY_SCHEMA",
    "GUIDED_ACTION_DRAW_SCHEMA",
    "GUIDED_ACTION_EXECUTION_SCHEMA",
    "GUIDED_BEHAVIOR_SCHEMA",
    "TERMINAL_STATE_TRACE_SCHEMA",
    "SelectedActionHuberLoss",
    "FrozenQGuidedPolicyConfig",
    "GuidedActionDrawCoordinator",
    "GuidedActionDrawKey",
    "GuidedActionExecutionRequest",
    "GuidedPolicyActionDrawRecord",
    "GuidedPolicyBehaviorRecord",
    "JointActorOptimizerConfig",
    "JointCriticOptimizerConfig",
    "JointTrainingConfig",
    "JointTrainingTargets",
    "TerminalStateTrace",
    "compile_outcome_returns_and_frozen_v_gae",
    "guided_log_probs_reference",
    "joint_training_contract_id",
    "parse_joint_policy_section",
    "parse_joint_training_section",
    "frozen_q_guided_log_probs",
    "replay_guided_behavior_log_probs",
    "sample_frozen_q_guided_action",
    "selected_action_huber_loss",
    "validate_guided_action_execution_result",
]
