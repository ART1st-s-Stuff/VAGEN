from __future__ import annotations

import math

import pytest

from vagen.k4_beta_calibration import calibrate_beta_from_action_spreads
from vagen.joint_policy.planning_contract import (
    K4MCTSGuidedBehaviorRecord,
    K4MCTSGuidedPolicyConfig,
    k4_guided_log_probs_reference,
    parse_k4_mcts_joint_policy_section,
)
from vagen.agent_loop.decision_ledger import (
    K4_GUIDED_DECISION_LEDGER_SCHEMA,
    build_guided_decision_ledger,
    validate_decision_ledger,
)
from vagen.joint_policy.planning_execution import (
    K4MCTSGuidedActionExecutionRequest,
    parse_guided_action_execution_request,
)
from vagen.joint_policy.planning_sampling import sample_k4_mcts_guided_action
from vagen.joint_policy.sampling import GuidedActionDrawKey


def _config(*, beta: float = 1.0) -> K4MCTSGuidedPolicyConfig:
    return K4MCTSGuidedPolicyConfig.from_mapping(
        {
            "implementation": "k4_mcts_guided_v1",
            "alpha": 1.0,
            "beta": beta,
            "prior_temperature": 1.0,
            "backprop_to_llm": True,
            "score_dtype": "float32",
            "planning_horizon": 4,
            "mcts_num_simulations": 100,
            "mcts_exploration_constant": 1.0,
        }
    )


def _key(config: K4MCTSGuidedPolicyConfig) -> GuidedActionDrawKey:
    contract_id = config.contract_id("navigation_v1", ("left", "right"), (11, 12))
    return GuidedActionDrawKey.build(
        run_seed=42002,
        policy_step=3,
        rollout_sample_id="base_train:7",
        rollout_repeat_index=0,
        turn_index=2,
        is_validation=False,
        snapshot_id="sha256:" + "a" * 64,
        contract_id=contract_id,
    )


def test_k4_distribution_uses_planner_means_not_direct_q() -> None:
    config = _config()
    prior, guided = k4_guided_log_probs_reference(
        prior_logits=(0.0, 0.0),
        planner_root_mean_values=(-2.0, 2.0),
        config=config,
    )

    assert prior == pytest.approx((-math.log(2), -math.log(2)))
    assert guided[1] > guided[0]

    draw = sample_k4_mcts_guided_action(
        action_space="navigation_v1",
        action_space_names=("left", "right"),
        action_token_ids=(11, 12),
        prior_logits=(0.0, 0.0),
        direct_all_action_q=(100.0, -100.0),
        planner_root_mean_values=(-2.0, 2.0),
        planner_root_visit_counts=(50, 50),
        draw_key=_key(config),
        config=config,
    )
    assert draw.direct_all_action_q == (100.0, -100.0)
    assert draw.planner_root_mean_values == (-2.0, 2.0)
    assert draw.guided_log_probs == tuple(guided)


def test_k4_behavior_persists_direct_q_and_planner_evidence_separately() -> None:
    config = _config(beta=0.0)
    draw = sample_k4_mcts_guided_action(
        action_space="navigation_v1",
        action_space_names=("left", "right"),
        action_token_ids=(11, 12),
        prior_logits=(1.0, 0.0),
        direct_all_action_q=(0.2, 0.7),
        planner_root_mean_values=(0.9, -0.3),
        planner_root_visit_counts=(61, 39),
        draw_key=_key(config),
        config=config,
    )
    behavior = K4MCTSGuidedBehaviorRecord.build(
        action_space=draw.action_space,
        action_space_names=draw.action_space_names,
        action_token_ids=draw.action_token_ids,
        snapshot_id=draw.draw_key.snapshot_id,
        prior_token_id=11,
        prior_action_id=0,
        prior_response_idx=7,
        behavior_llm_prior_logprob=draw.prior_log_probs[0],
        prior_logits=draw.prior_logits,
        direct_all_action_q=draw.direct_all_action_q,
        planner_root_mean_values=draw.planner_root_mean_values,
        planner_root_visit_counts=draw.planner_root_visit_counts,
        guided_action_id=draw.guided_action_id,
        behavior_guided_logprob=draw.behavior_guided_logprob,
        config=config,
    )

    restored = K4MCTSGuidedBehaviorRecord.from_mapping(behavior.to_mapping())
    assert restored == behavior
    assert restored.direct_all_action_q == (0.2, 0.7)
    assert restored.planner_root_mean_values == (0.9, -0.3)
    assert sum(restored.planner_root_visit_counts) == 100


def test_beta_calibration_persists_zero_prior_spread_for_review() -> None:
    result = calibrate_beta_from_action_spreads(
        [0.0, 0.0, 0.25],
        [0.5, 1.0, 1.5],
        minimum_median_planner_spread=1e-8,
    )

    assert result["calibration_accepted"] is False
    assert result["review_reason"] == "llm_median_action_spread_is_zero"
    assert result["calibrated_beta_requires_human_approval"] == 0.0
    assert result["prior_action_spreads"]["zero_count"] == 2
    assert result["mcts_action_spreads"]["median"] == 1.0


def test_beta_calibration_rejects_tiny_mcts_spread_without_division() -> None:
    result = calibrate_beta_from_action_spreads(
        [1.0, 2.0],
        [0.0, 1e-10],
        minimum_median_planner_spread=1e-8,
    )

    assert result["calibration_accepted"] is False
    assert result["review_reason"] == "mcts_median_action_spread_too_small"
    assert result["calibrated_beta_requires_human_approval"] is None


def test_beta_calibration_accepts_positive_ratio() -> None:
    result = calibrate_beta_from_action_spreads(
        [1.0, 3.0],
        [0.5, 1.5],
        minimum_median_planner_spread=1e-8,
    )

    assert result["calibration_accepted"] is True
    assert result["review_reason"] is None
    assert result["calibrated_beta_requires_human_approval"] == 2.0


def test_k4_execution_and_decision_ledger_preserve_planning_behavior() -> None:
    config = _config(beta=0.0)
    draw = sample_k4_mcts_guided_action(
        action_space="navigation_v1",
        action_space_names=("left", "right"),
        action_token_ids=(11, 12),
        prior_logits=(1.0, 0.0),
        direct_all_action_q=(0.2, 0.7),
        planner_root_mean_values=(0.9, -0.3),
        planner_root_visit_counts=(61, 39),
        draw_key=_key(config),
        config=config,
    )
    behavior = K4MCTSGuidedBehaviorRecord.build(
        action_space=draw.action_space,
        action_space_names=draw.action_space_names,
        action_token_ids=draw.action_token_ids,
        snapshot_id=draw.draw_key.snapshot_id,
        prior_token_id=11,
        prior_action_id=0,
        prior_response_idx=3,
        behavior_llm_prior_logprob=-0.5,
        prior_logits=draw.prior_logits,
        direct_all_action_q=draw.direct_all_action_q,
        planner_root_mean_values=draw.planner_root_mean_values,
        planner_root_visit_counts=draw.planner_root_visit_counts,
        guided_action_id=draw.guided_action_id,
        behavior_guided_logprob=draw.behavior_guided_logprob,
        config=config,
    )
    execution = K4MCTSGuidedActionExecutionRequest.from_behavior(
        behavior,
        raw_response="<think>real</think><action>",
        response_trace_id="sha256:" + "b" * 64,
        action_draw_record_id=draw.record_id(),
    )
    restored = parse_guided_action_execution_request(execution.to_mapping())
    assert restored == execution
    ledger = build_guided_decision_ledger(
        behavior=behavior,
        env_turn_reward=0.01,
        env_terminated=False,
        rollout_truncated=False,
        format_valid=True,
    )
    validate_decision_ledger(ledger)
    assert ledger["schema"] == K4_GUIDED_DECISION_LEDGER_SCHEMA
    assert ledger["decision_sources"] == ["k4_mcts_guided"]


def test_k4_contract_requires_exact_search_settings_and_complete_root_visits() -> None:
    with pytest.raises(ValueError, match="planning_horizon"):
        K4MCTSGuidedPolicyConfig.from_mapping(
            {
                **_config().to_mapping(),
                "planning_horizon": 3,
            }
        )
    config = _config()
    with pytest.raises(ValueError, match="sum to mcts_num_simulations"):
        sample_k4_mcts_guided_action(
            action_space="navigation_v1",
            action_space_names=("left", "right"),
            action_token_ids=(11, 12),
            prior_logits=(0.0, 0.0),
            direct_all_action_q=(0.0, 0.0),
            planner_root_mean_values=(0.0, 0.0),
            planner_root_visit_counts=(49, 50),
            draw_key=_key(config),
            config=config,
        )


def test_k4_top_level_parser_has_no_defaults() -> None:
    raw = {"enabled": True, **_config().to_mapping()}
    assert parse_k4_mcts_joint_policy_section(raw) == _config()
    with pytest.raises(ValueError, match="missing fields"):
        parse_k4_mcts_joint_policy_section(
            {key: value for key, value in raw.items() if key != "mcts_num_simulations"}
        )
