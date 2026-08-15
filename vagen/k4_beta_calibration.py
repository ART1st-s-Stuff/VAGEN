"""Optimizer-free balanced K4 MCTS calibration for Scheme-B beta."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from vagen.agent_loop.decision_ledger import (
    K4_GUIDED_DECISION_LEDGER_SCHEMA,
    summarize_decision_ledger_batch,
    validate_decision_ledger_reward_rows,
)
from vagen.standalone_one_turn_smoke import (
    atomic_write_json,
    build_config,
    build_ray_runtime_env,
)

_TRAIN_SPLITS = ("base_train", "common_sense_train", "long_horizon_train")


def build_initial_transport(
    args: argparse.Namespace,
    config: Any,
    tokenizer: Any,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Create the self-verifying ID74 full-planner transport before Ray rollout."""

    import torch

    from nimloth.latent import LatentActionTokens, special_token_ids
    from nimloth.training.rl.joint_planner import (
        FrozenMCTSPlanningConfig,
        create_frozen_planning_snapshot,
        load_joint_world_model_critic,
        save_frozen_planning_snapshot_file,
    )
    from vagen.envs.navigation.utils.nimloth_format import ACTION_NAMES
    from vagen.joint_policy import K4MCTSGuidedPolicyConfig
    from vagen.joint_policy.planning_owner import (
        FROZEN_K4_PLANNER_TRANSPORT_SCHEMA,
    )

    raw_policy = OmegaConf.to_container(config.joint_policy, resolve=True)
    policy = K4MCTSGuidedPolicyConfig.from_mapping(
        {key: value for key, value in raw_policy.items() if key != "enabled"}
    )
    tokens = LatentActionTokens()
    token_ids = special_token_ids(
        tokenizer,
        latent_token_count=args.latent_token_count,
    )
    action_token_ids = tuple(token_ids[token] for token in tokens.action_tokens)
    contract_id = policy.contract_id(
        "navigation_v1",
        ACTION_NAMES,
        action_token_ids,
    )
    model = load_joint_world_model_critic(
        checkpoint_root=args.critic_checkpoint,
        expected_qwen_hidden_dim=args.critic_qwen_hidden_dim,
        expected_grid_tokens=args.latent_token_count,
        expected_state_dim=args.critic_state_dim,
        expected_action_count=len(ACTION_NAMES),
        expected_prediction_horizon=args.planning_horizon,
        device=torch.device("cpu"),
        trainable=False,
    )
    snapshot = create_frozen_planning_snapshot(
        model,
        source_step=args.joint_snapshot_source_step,
        contract_id=contract_id,
        score_dtype=args.joint_score_dtype,
        planning_config=FrozenMCTSPlanningConfig(
            horizon=args.planning_horizon,
            num_simulations=args.mcts_num_simulations,
            exploration_constant=args.mcts_exploration_constant,
        ),
    )
    transport_path = save_frozen_planning_snapshot_file(
        snapshot,
        output_dir / "frozen_k4_planner.pt",
    )
    return {
        "schema": FROZEN_K4_PLANNER_TRANSPORT_SCHEMA,
        "transport_path": str(transport_path),
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_source_step": snapshot.source_step,
        "contract_id": snapshot.contract_id,
        "score_dtype": snapshot.score_dtype,
        "planning_horizon": snapshot.planning_config.horizon,
        "mcts_num_simulations": snapshot.planning_config.num_simulations,
        "mcts_exploration_constant": (
            snapshot.planning_config.exploration_constant
        ),
    }


def build_input(args: argparse.Namespace) -> Any:
    """Build 24 unique balanced train-split trajectories and no validation rows."""

    from verl.protocol import DataProto

    rows = [
        (split, seed)
        for split in _TRAIN_SPLITS
        for seed in range(args.seed_start, args.seed_start + args.seeds_per_split)
    ]
    if len(rows) != args.trajectory_count:
        raise ValueError(
            f"balanced calibration row count {len(rows)} != {args.trajectory_count}"
        )
    configs = [
        {
            "base_urls": args.env_url,
            "timeout": args.env_timeout,
            "retries": args.env_retries,
            "eval_set": split,
            "prompt_format": "nimloth",
            "latent_token_count": args.latent_token_count,
            "max_actions_per_step": 1,
            "action_sep": "|",
            "example_count": 0,
            "format_reward": args.format_reward,
            "per_turn_format_reward": args.per_turn_format_reward,
            "success_reward": args.success_reward,
        }
        for split, _seed in rows
    ]
    non_tensors = {
        "agent_name": np.array(["gym_agent"] * len(rows), dtype=object),
        "env_name": np.array(["RemoteEnv"] * len(rows), dtype=object),
        "seed": np.array([seed for _split, seed in rows], dtype=object),
        "max_turns": np.array([args.max_turns] * len(rows), dtype=object),
        "response_length_per_turn": np.array(
            [args.response_length] * len(rows),
            dtype=object,
        ),
        "group_idx": np.array([split for split, _seed in rows], dtype=object),
        "traj_idx": np.array(
            [seed - args.seed_start for _split, seed in rows],
            dtype=np.int64,
        ),
        "rollout_sample_id": np.array(
            [f"navigation:{split}:{seed}" for split, seed in rows],
            dtype=object,
        ),
        "rollout_repeat_index": np.zeros(len(rows), dtype=np.int64),
        "data_source": np.array(["navigation"] * len(rows), dtype=object),
        "config": np.array(configs, dtype=object),
    }
    return DataProto.from_dict(
        non_tensors=non_tensors,
        meta_info={"validate": False, "global_steps": 0},
    )


def validate_and_summarize(
    result: Any,
    args: argparse.Namespace,
    *,
    transport: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate every turn and calculate one fixed 1:1 median-spread beta."""

    from nimloth.training.rl.joint_planning_scoring import (
        FrozenK4PlanningScoringRecord,
    )
    from vagen.joint_policy import K4MCTSGuidedBehaviorRecord
    from vagen.joint_policy.terminal_state import TerminalStateTrace

    row_count = len(result)
    if row_count < args.trajectory_count:
        raise RuntimeError("K4 calibration returned fewer rows than trajectories")
    ledgers = result.non_tensor_batch["decision_ledger"].tolist()
    metrics = summarize_decision_ledger_batch(
        ledgers,
        expected_batch_size=row_count,
        allowed_schemas={K4_GUIDED_DECISION_LEDGER_SCHEMA},
    )
    validate_decision_ledger_reward_rows(
        ledgers,
        reward_rows=result.batch["rm_scores"].detach().cpu().tolist(),
        response_masks=result.batch["response_mask"].detach().cpu().tolist(),
    )
    required_non_tensor = {
        "group_idx",
        "traj_idx",
        "guided_turn_index",
        "rollout_stop_reason",
        "policy_state",
        "policy_response_trace",
        "joint_policy_batch_pin",
        "frozen_k4_planning_scoring",
        "guided_action_draw",
        "guided_action_execution",
        "terminal_state_trace",
    }
    missing = required_non_tensor - set(result.non_tensor_batch)
    if missing:
        raise RuntimeError(f"K4 calibration output is missing fields: {sorted(missing)}")

    prior_spreads: list[float] = []
    planner_spreads: list[float] = []
    planner_latencies: list[float] = []
    compact_rows: list[dict[str, Any]] = []
    final_by_trajectory: dict[tuple[str, int], str] = {}
    turns_by_split = {split: 0 for split in _TRAIN_SPLITS}
    successes_by_split = {split: 0 for split in _TRAIN_SPLITS}
    for index, ledger in enumerate(ledgers):
        split = str(result.non_tensor_batch["group_idx"][index])
        traj_idx = int(result.non_tensor_batch["traj_idx"][index])
        if split not in turns_by_split:
            raise RuntimeError(f"K4 calibration returned non-train split {split!r}")
        turns_by_split[split] += 1
        behavior = K4MCTSGuidedBehaviorRecord.from_mapping(
            ledger["behavior_record"]
        )
        if behavior.policy_config.beta != 0.0:
            raise RuntimeError("beta calibration behavior must use beta=0")
        scoring = FrozenK4PlanningScoringRecord.from_mapping(
            result.non_tensor_batch["frozen_k4_planning_scoring"][index]
        )
        if (
            scoring.snapshot_id != transport["snapshot_id"]
            or scoring.contract_id != transport["contract_id"]
            or scoring.direct_all_action_q != behavior.direct_all_action_q
            or scoring.planner_root_mean_values
            != behavior.planner_root_mean_values
            or scoring.planner_root_visit_counts
            != behavior.planner_root_visit_counts
        ):
            raise RuntimeError("K4 calibration scoring and behavior evidence mismatch")
        policy_state = result.non_tensor_batch["policy_state"][index]
        if (
            policy_state.get("schema") != "nimloth_policy_state_k4_mcts_v1"
            or policy_state.get("request_id") != scoring.request_id
            or policy_state.get("generation_id") != scoring.generation_id
        ):
            raise RuntimeError("K4 calibration policy-state identity mismatch")
        prior_spread = statistics.pstdev(behavior.prior_logits)
        planner_spread = statistics.pstdev(behavior.planner_root_mean_values)
        if not math.isfinite(prior_spread) or not math.isfinite(planner_spread):
            raise RuntimeError("K4 calibration action spread is non-finite")
        prior_spreads.append(prior_spread)
        planner_spreads.append(planner_spread)
        planner_latencies.append(scoring.planner_latency_seconds)
        stop_reason = str(result.non_tensor_batch["rollout_stop_reason"][index])
        if stop_reason == "infrastructure_truncation":
            raise RuntimeError("K4 calibration contains infrastructure truncation")
        identity = (split, traj_idx)
        terminal_raw = result.non_tensor_batch["terminal_state_trace"][index]
        if stop_reason == "continue":
            if terminal_raw is not None:
                raise RuntimeError("K4 nonterminal row contains terminal trace")
        else:
            if identity in final_by_trajectory:
                raise RuntimeError("K4 calibration trajectory has multiple final rows")
            terminal = TerminalStateTrace.from_mapping(terminal_raw)
            if terminal.rollout_stop_reason != stop_reason:
                raise RuntimeError("K4 terminal trace stop reason mismatch")
            final_by_trajectory[identity] = stop_reason
            successes_by_split[split] += int(stop_reason == "success")
        state_hash = hashlib.sha256(
            json.dumps(
                policy_state,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        compact_rows.append(
            {
                "split": split,
                "trajectory_index": traj_idx,
                "guided_turn_index": int(
                    result.non_tensor_batch["guided_turn_index"][index]
                ),
                "rollout_stop_reason": stop_reason,
                "env_turn_reward": float(ledger["env_turn_reward"]),
                "policy_state_sha256": f"sha256:{state_hash}",
                "scoring_record": scoring.to_mapping(),
                "behavior_record": behavior.to_mapping(),
                "response_trace": result.non_tensor_batch[
                    "policy_response_trace"
                ][index],
                "guided_action_draw": result.non_tensor_batch[
                    "guided_action_draw"
                ][index],
                "guided_action_execution": result.non_tensor_batch[
                    "guided_action_execution"
                ][index],
            }
        )
    if len(final_by_trajectory) != args.trajectory_count:
        raise RuntimeError(
            "K4 calibration did not complete every trajectory: "
            f"{len(final_by_trajectory)} != {args.trajectory_count}"
        )
    expected_per_split = args.seeds_per_split
    for split in _TRAIN_SPLITS:
        completed = sum(key[0] == split for key in final_by_trajectory)
        if completed != expected_per_split:
            raise RuntimeError(
                f"K4 calibration split {split} completed {completed}, expected {expected_per_split}"
            )
    median_prior = statistics.median(prior_spreads)
    median_planner = statistics.median(planner_spreads)
    if median_planner <= args.minimum_median_planner_spread:
        raise RuntimeError(
            "K4 planner median action spread is too small to calibrate beta: "
            f"{median_planner} <= {args.minimum_median_planner_spread}"
        )
    calibrated_beta = median_prior / median_planner
    if not math.isfinite(calibrated_beta) or calibrated_beta <= 0.0:
        raise RuntimeError("calibrated K4 beta is not finite and positive")
    summary = {
        "schema": "vagen_k4_beta_calibration_summary_v1",
        "status": "passed",
        "optimizer": None,
        "checkpoint_output": None,
        "beta_applied_during_calibration": 0.0,
        "calibrated_beta_requires_human_approval": calibrated_beta,
        "calibration_rule": "median_population_std_prior_logits / median_population_std_mcts_root_mean",
        "trajectory_count": args.trajectory_count,
        "executed_turn_count": row_count,
        "train_splits": list(_TRAIN_SPLITS),
        "seeds_per_split": args.seeds_per_split,
        "turns_by_split": turns_by_split,
        "successes_by_split": successes_by_split,
        "median_prior_action_spread": median_prior,
        "median_mcts_action_spread": median_planner,
        "minimum_median_planner_spread": args.minimum_median_planner_spread,
        "planner_latency_seconds": {
            "mean": statistics.fmean(planner_latencies),
            "median": statistics.median(planner_latencies),
            "max": max(planner_latencies),
        },
        "planning_snapshot": transport,
        "decision_ledger_metrics": metrics,
    }
    return summary, compact_rows


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                json.dump(row, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> dict[str, Any]:
    import ray
    import vagen.rollout.nimloth_vllm  # noqa: F401 -- register custom replica
    from verl.utils import hf_tokenizer
    from vagen.agent_loop.agent_loop_no_concat import AgentLoopManager

    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite K4 calibration output: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True)
    os.environ.setdefault("VLLM_USE_V1", "1")
    ray_kwargs: dict[str, Any] = {
        "address": "local",
        "runtime_env": build_ray_runtime_env(),
        "include_dashboard": False,
    }
    if ray_temp_dir := os.environ.get("RAY_TMPDIR"):
        ray_kwargs["_temp_dir"] = ray_temp_dir
    try:
        config = build_config(args)
        tokenizer = hf_tokenizer(str(args.model), trust_remote_code=True)
        transport = build_initial_transport(
            args,
            config,
            tokenizer,
            output_dir=args.output_dir,
        )
        atomic_write_json(args.output_dir / "planning_transport.json", transport)
        ray.init(**ray_kwargs)
        manager = AgentLoopManager(
            config,
            worker_group=None,
            initial_frozen_q_snapshot_state=transport,
            guided_draw_run_seed=args.joint_run_seed,
        )
        result = manager.generate_sequences(build_input(args))
        summary, rows = validate_and_summarize(
            result,
            args,
            transport=transport,
        )
        summary.update(
            {
                "model": str(args.model.resolve()),
                "critic_checkpoint": str(args.critic_checkpoint.resolve()),
                "env_url": args.env_url,
                "run_name": args.run_name,
                "max_turns": args.max_turns,
                "response_length": args.response_length,
                "cot_temperature": args.temperature,
                "cot_top_p": args.top_p,
                "per_turn_format_reward": args.per_turn_format_reward,
                "format_reward": args.format_reward,
                "success_reward": args.success_reward,
                "run_seed": args.joint_run_seed,
            }
        )
        _atomic_write_jsonl(args.output_dir / "turn_records.jsonl", rows)
        atomic_write_json(args.output_dir / "summary.json", summary)
        return summary
    except BaseException as exc:
        atomic_write_json(
            args.output_dir / "failure.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "optimizer": None,
                "checkpoint_output": None,
            },
        )
        raise
    finally:
        if ray.is_initialized():
            ray.shutdown()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model",
        "critic-checkpoint",
        "agent-loop-config",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--latent-token-count", type=int, required=True)
    parser.add_argument("--critic-qwen-hidden-dim", type=int, required=True)
    parser.add_argument("--critic-state-dim", type=int, required=True)
    parser.add_argument("--joint-snapshot-source-step", type=int, required=True)
    parser.add_argument("--joint-run-seed", type=int, required=True)
    parser.add_argument("--joint-alpha", type=float, required=True)
    parser.add_argument("--joint-beta", type=float, required=True)
    parser.add_argument("--joint-prior-temperature", type=float, required=True)
    parser.add_argument("--joint-score-dtype", required=True, choices=tuple(("float32", "bfloat16", "float64")))
    parser.add_argument("--planning-horizon", type=int, required=True)
    parser.add_argument("--mcts-num-simulations", type=int, required=True)
    parser.add_argument("--mcts-exploration-constant", type=float, required=True)
    parser.add_argument("--trajectory-count", type=int, required=True)
    parser.add_argument("--seeds-per-split", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--response-length", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--per-turn-format-reward", type=float, required=True)
    parser.add_argument("--format-reward", type=float, required=True)
    parser.add_argument("--success-reward", type=float, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--agent-loop-num-workers", type=int, required=True)
    parser.add_argument("--env-timeout", type=float, required=True)
    parser.add_argument("--env-retries", type=int, required=True)
    parser.add_argument("--minimum-median-planner-spread", type=float, required=True)
    args = parser.parse_args(argv)
    args.k4_guided = True
    args.guided = False
    if args.planning_horizon != 4:
        parser.error("--planning-horizon must be exactly 4")
    if args.tensor_parallel_size != 8:
        parser.error("K4 calibration requires --tensor-parallel-size=8")
    if args.trajectory_count != 24 or args.seeds_per_split != 8:
        parser.error("K4 calibration requires 24 trajectories and 8 seeds per split")
    if args.joint_alpha != 1.0 or args.joint_beta != 0.0 or args.joint_prior_temperature != 1.0:
        parser.error("K4 calibration requires alpha=1, beta=0, prior temperature=1")
    for path, label in (
        (args.model, "--model"),
        (args.critic_checkpoint, "--critic-checkpoint"),
    ):
        if not path.is_dir():
            parser.error(f"{label} must be a directory")
    if not args.agent_loop_config.is_file():
        parser.error("--agent-loop-config must exist")
    for field in (
        "latent_token_count",
        "critic_qwen_hidden_dim",
        "critic_state_dim",
        "mcts_num_simulations",
        "max_turns",
        "prompt_length",
        "response_length",
        "max_num_seqs",
        "agent_loop_num_workers",
    ):
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.mcts_num_simulations < 8:
        parser.error("MCTS simulations must visit all 8 root actions")
    if not 0.0 <= args.temperature or not 0.0 < args.top_p <= 1.0:
        parser.error("invalid CoT sampling temperature/top-p")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        parser.error("invalid GPU memory utilization")
    if args.env_retries < 0 or args.joint_run_seed < 0 or args.seed_start < 0:
        parser.error("seeds and retry count must be non-negative")
    finite_fields = (
        "mcts_exploration_constant",
        "per_turn_format_reward",
        "format_reward",
        "success_reward",
        "env_timeout",
        "minimum_median_planner_spread",
    )
    if any(not math.isfinite(float(getattr(args, field))) for field in finite_fields):
        parser.error("all calibration numeric fields must be finite")
    if args.mcts_exploration_constant < 0.0 or args.minimum_median_planner_spread <= 0.0:
        parser.error("MCTS exploration must be non-negative and spread threshold positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
