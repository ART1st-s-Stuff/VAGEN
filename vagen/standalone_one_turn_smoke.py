"""Optimizer-free one-turn Navigation smoke for the Nimloth K-slot protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from vagen.agent_loop.decision_ledger import (
    DECISION_LEDGER_SCHEMA,
    GUIDED_DECISION_LEDGER_SCHEMA,
    summarize_decision_ledger_batch,
    validate_decision_ledger_reward_rows,
)


def build_config(args: argparse.Namespace) -> Any:
    """Construct only model/rollout/environment config; no optimizer exists."""

    joint_policy: dict[str, Any]
    if getattr(args, "guided", False):
        joint_policy = {
            "enabled": True,
            "implementation": "frozen_q_guided_v1",
            "alpha": args.joint_alpha,
            "beta": args.joint_beta,
            "prior_temperature": args.joint_prior_temperature,
            "backprop_to_llm": True,
            "score_dtype": args.joint_score_dtype,
        }
    else:
        joint_policy = {"enabled": False}

    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "_target_": "verl.workers.config.HFModelConfig",
                    "path": str(args.model),
                    "trust_remote_code": True,
                    "use_shm": False,
                    "enable_gradient_checkpointing": False,
                    "external_lib": "vagen.rollout.nimloth_vllm",
                },
                "rollout": {
                    "_target_": "verl.workers.config.RolloutConfig",
                    "name": "nimloth_vllm",
                    "mode": "async",
                    "prompt_length": args.prompt_length,
                    "response_length": args.response_length,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": -1,
                    "do_sample": args.temperature > 0.0,
                    "calculate_log_probs": True,
                    "n": 1,
                    "dtype": "bfloat16",
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "enforce_eager": True,
                    "free_cache_engine": False,
                    "load_format": "auto",
                    "tensor_model_parallel_size": args.tensor_parallel_size,
                    "data_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "expert_parallel_size": 1,
                    "max_num_batched_tokens": args.prompt_length
                    + args.response_length,
                    "max_num_seqs": 1,
                    "enable_chunked_prefill": True,
                    "enable_prefix_caching": False,
                    "engine_kwargs": {
                        "vllm": {"mm_encoder_tp_mode": "data"}
                    },
                    "agent": {
                        "_target_": "verl.workers.config.AgentLoopConfig",
                        "num_workers": 1,
                        "default_agent_loop": "gym_agent",
                        "agent_loop_config_path": str(args.agent_loop_config),
                        "custom_async_server": {
                            "_target_": "verl.workers.config.CustomAsyncServerConfig",
                            "path": None,
                            "name": None,
                        },
                    },
                    "multi_turn": {
                        "_target_": "verl.workers.config.MultiTurnConfig",
                        "enable": True,
                    },
                    "val_kwargs": {
                        "_target_": "verl.workers.config.SamplingConfig",
                        "top_k": -1,
                        "top_p": args.top_p,
                        "temperature": args.temperature,
                        "do_sample": args.temperature > 0.0,
                        "n": 1,
                    },
                    "trace": {
                        "_target_": "verl.workers.config.TraceConfig",
                        "backend": None,
                        "token2text": False,
                    },
                    "prometheus": {
                        "_target_": "verl.workers.config.PrometheusConfig",
                        "enable": False,
                    },
                },
            },
            "trainer": {
                "n_gpus_per_node": args.tensor_parallel_size,
                "nnodes": 1,
                "project_name": "nimloth-rl",
                "experiment_name": args.run_name,
                "concat_multi_turn": False,
            },
            "reward_model": {
                "enable": False,
                "enable_resource_pool": False,
                "reward_manager": "naive",
            },
            "data": {"apply_chat_template_kwargs": {}},
            "env_registry": {
                "RemoteEnv": "vagen.envs_remote.GymImageEnvClient"
            },
            "decision_ledger": {"enabled": True},
            "joint_policy": joint_policy,
        }
    )


def build_input(args: argparse.Namespace) -> Any:
    from verl.protocol import DataProto

    non_tensors = {
        "agent_name": np.array(["gym_agent"], dtype=object),
        "env_name": np.array(["RemoteEnv"], dtype=object),
        "seed": np.array([int(args.seed)], dtype=object),
        "max_turns": np.array([1], dtype=object),
        "response_length_per_turn": np.array(
            [int(args.response_length)], dtype=object
        ),
        "group_idx": np.array([args.run_name], dtype=object),
        "traj_idx": np.array([0]),
        "rollout_sample_id": np.array(
            [f"standalone:navigation:{args.eval_set}:{int(args.seed)}"],
            dtype=object,
        ),
        "rollout_repeat_index": np.array([0]),
        "data_source": np.array(["navigation"], dtype=object),
        "config": np.array(
            [
                {
                    "base_urls": args.env_url,
                    "timeout": args.env_timeout,
                    "retries": 0,
                    "eval_set": args.eval_set,
                    "prompt_format": "nimloth",
                    "latent_token_count": args.latent_token_count,
                    "max_actions_per_step": 1,
                    "action_sep": "|",
                    "example_count": 0,
                    "format_reward": 0.0,
                    "per_turn_format_reward": 0.0,
                    "success_reward": 10.0,
                    "success_threshold": 1.5,
                    "step_length": 0.5,
                }
            ],
            dtype=object,
        ),
    }
    return DataProto.from_dict(
        non_tensors=non_tensors,
        meta_info={"validate": True, "global_steps": 0},
    )


def validate_result(result: Any, tokenizer: Any) -> dict[str, Any]:
    if len(result) != 1:
        raise ValueError(f"one-turn smoke expected one row, got {len(result)}")
    ledgers = result.non_tensor_batch["decision_ledger"].tolist()
    metrics = summarize_decision_ledger_batch(
        ledgers,
        expected_batch_size=1,
        allowed_schemas={
            DECISION_LEDGER_SCHEMA,
            GUIDED_DECISION_LEDGER_SCHEMA,
        },
    )
    response_masks = result.batch["response_mask"].detach().cpu().tolist()
    reward_rows = result.batch["rm_scores"].detach().cpu().tolist()
    validate_decision_ledger_reward_rows(
        ledgers,
        reward_rows=reward_rows,
        response_masks=response_masks,
    )
    ledger = ledgers[0]
    expected_action_names = [
        "move_forward",
        "move_backward",
        "move_right",
        "move_left",
        "turn_right",
        "turn_left",
        "look_up",
        "look_down",
    ]
    if (
        ledger["action_space"] != "navigation_v1"
        or ledger["action_space_names"] != expected_action_names
    ):
        raise RuntimeError("ledger does not match the Navigation action contract")
    is_guided = ledger["schema"] == GUIDED_DECISION_LEDGER_SCHEMA
    if is_guided:
        if (
            ledger["decision_sources"] != ["frozen_q_guided"]
            or ledger["decision_is_policy_sampled"] != [True]
        ):
            raise RuntimeError("guided ledger has invalid action ownership")
        behavior = ledger["behavior_record"]
        prior_action_id = int(behavior["prior_action_id"])
        if ledger["executed_action_ids"] != [int(behavior["guided_action_id"])]:
            raise RuntimeError("guided ledger did not bind the executed action")
    else:
        if (
            ledger["decision_sources"] != ["llm_text"]
            or ledger["decision_is_policy_sampled"] != [False]
        ):
            raise RuntimeError("ledger does not match the Navigation M1 action contract")
        prior_action_id = int(ledger["executed_action_ids"][0])
    if not ledger["format_valid"] or len(ledger["executed_action_ids"]) != 1:
        raise RuntimeError(
            "K-slot response was not accepted as exactly one environment action"
        )
    if int(result.non_tensor_batch["turn_idx"][0]) != 1:
        raise RuntimeError("one-turn smoke produced an unexpected turn index")
    policy_state = result.non_tensor_batch["policy_state"][0]
    if (
        not isinstance(policy_state, dict)
        or policy_state.get("schema") != "nimloth_policy_state_v2"
        or not isinstance(policy_state.get("request_id"), str)
        or not policy_state["request_id"]
        or not isinstance(policy_state.get("generation_id"), str)
        or not policy_state["generation_id"]
        or policy_state["generation_id"] == policy_state["request_id"]
        or len(policy_state.get("latent_hidden", [])) != 16
        or any(len(row) != 2048 for row in policy_state["latent_hidden"])
        or len(policy_state.get("action_logits", [])) != 8
        or any(
            not math.isfinite(float(value))
            for row in policy_state["latent_hidden"]
            for value in row
        )
        or any(
            not math.isfinite(float(value))
            for value in policy_state["action_logits"]
        )
    ):
        raise RuntimeError("one-turn smoke received an invalid policy-state capture")

    if "rollout_log_probs" not in result.batch:
        raise RuntimeError("Nimloth smoke requires rollout token log-probabilities")
    response_ids = result.batch["responses"][0]
    attention = result.batch["attention_mask"][0]
    prompt_width = result.batch["prompts"].shape[1]
    response_length = int(attention[prompt_width:].sum().item())
    continuation_ids = response_ids[:response_length].tolist()
    continuation_text = tokenizer.decode(
        continuation_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    response_mask = response_masks[0][:response_length]
    rollout_log_probs = (
        result.batch["rollout_log_probs"][0][:response_length]
        .detach()
        .cpu()
        .tolist()
    )
    if len(rollout_log_probs) != response_length or any(
        not math.isfinite(float(value))
        for value, sampled in zip(rollout_log_probs, response_mask, strict=True)
        if sampled
    ):
        raise RuntimeError("rollout token log-probabilities are missing or non-finite")
    environment_response = "<think>" + continuation_text
    action_token = f"<|action_({prior_action_id})|>"
    if action_token not in environment_response:
        raise RuntimeError("prior action does not match generated action token")
    if is_guided:
        for field in (
            "guided_turn_index",
            "joint_policy_batch_pin",
            "frozen_q_scoring",
            "policy_response_trace",
            "guided_action_draw",
            "guided_action_execution",
        ):
            if field not in result.non_tensor_batch:
                raise RuntimeError(f"guided smoke is missing provenance field {field}")
        if int(result.non_tensor_batch["guided_turn_index"][0]) != (
            int(result.non_tensor_batch["turn_idx"][0]) - 1
        ):
            raise RuntimeError("guided and legacy turn indices do not align")
    return {
        "status": "passed",
        "optimizer": None,
        "checkpoint_output": None,
        "generated_continuation": continuation_text,
        "environment_response": environment_response,
        "decision_ledger": ledger,
        "env_turn_reward": float(ledger["env_turn_reward"]),
        "reward_anchor_index": max(
            index for index, value in enumerate(response_masks[0]) if value
        ),
        "response_token_count": response_length,
        "policy_state": policy_state,
        "metrics": metrics,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def build_ray_runtime_env() -> dict[str, Any]:
    """Pin the environment required by standalone rollout actors."""

    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    runtime_env = get_ppo_ray_runtime_env()
    actor_env = runtime_env.setdefault("env_vars", {})
    actor_env["VLLM_USE_V1"] = "1"
    for name in (
        "PATH",
        "PYTHONPATH",
        "HOME",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "TMPDIR",
        "VLLM_USE_FLASHINFER_SAMPLER",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "VLLM_ALLREDUCE_USE_SYMM_MEM",
        "NIMLOTH_LATENT_TOKEN_COUNT",
    ):
        if value := os.environ.get(name):
            actor_env[name] = value
    return runtime_env


def build_initial_frozen_q_snapshot(
    args: argparse.Namespace,
    config: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    """Load the explicitly supplied critic sidecars for a guided smoke."""

    import torch

    from nimloth.latent import LatentActionTokens, special_token_ids
    from nimloth.training.rl.joint_critic import (
        create_frozen_critic_snapshot,
        export_frozen_critic_snapshot,
        load_joint_action_value_critic,
    )
    from vagen.envs.navigation.utils.nimloth_format import ACTION_NAMES
    from vagen.joint_policy import parse_joint_policy_section

    raw_policy = OmegaConf.to_container(config.joint_policy, resolve=True)
    policy = parse_joint_policy_section(raw_policy)
    if policy is None:
        raise RuntimeError("guided snapshot bootstrap requires enabled joint policy")
    tokens = LatentActionTokens()
    token_ids = special_token_ids(
        tokenizer,
        latent_token_count=args.latent_token_count,
    )
    action_token_ids = tuple(token_ids[token] for token in tokens.action_tokens)
    critic = load_joint_action_value_critic(
        checkpoint_root=args.critic_checkpoint,
        expected_qwen_hidden_dim=args.critic_qwen_hidden_dim,
        expected_grid_tokens=args.latent_token_count,
        expected_state_dim=args.critic_state_dim,
        expected_action_count=len(ACTION_NAMES),
        device=torch.device("cpu"),
        trainable=False,
    )
    snapshot = create_frozen_critic_snapshot(
        critic,
        source_step=args.joint_snapshot_source_step,
        contract_id=policy.contract_id(
            "navigation_v1",
            ACTION_NAMES,
            action_token_ids,
        ),
        score_dtype=policy.score_dtype,
    )
    return export_frozen_critic_snapshot(snapshot).to_mapping()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import ray
    import vagen.rollout.nimloth_vllm  # noqa: F401 -- registers replica
    from verl.utils import hf_tokenizer
    from vagen.agent_loop.agent_loop_no_concat import AgentLoopManager

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite smoke output: {args.output}")
    os.environ.setdefault("VLLM_USE_V1", "1")
    ray_init_kwargs: dict[str, Any] = {
        "address": "local",
        "runtime_env": build_ray_runtime_env(),
        "include_dashboard": False,
    }
    if ray_temp_dir := os.environ.get("RAY_TMPDIR"):
        ray_init_kwargs["_temp_dir"] = ray_temp_dir
    manager = None
    try:
        ray.init(**ray_init_kwargs)
        config = build_config(args)
        tokenizer = hf_tokenizer(str(args.model), trust_remote_code=True)
        manager_kwargs: dict[str, Any] = {}
        if getattr(args, "guided", False):
            manager_kwargs = {
                "initial_frozen_q_snapshot_state": (
                    build_initial_frozen_q_snapshot(args, config, tokenizer)
                ),
                "guided_draw_run_seed": args.joint_run_seed,
            }
        manager = AgentLoopManager(
            config,
            worker_group=None,
            **manager_kwargs,
        )
        result = manager.generate_sequences(build_input(args))
        payload = validate_result(result, tokenizer)
        payload["model"] = str(args.model)
        payload["env_url"] = args.env_url
        payload["eval_set"] = args.eval_set
        payload["seed"] = args.seed
        payload["latent_token_count"] = args.latent_token_count
        payload["guided"] = bool(getattr(args, "guided", False))
        atomic_write_json(args.output, payload)
        return payload
    finally:
        if ray.is_initialized():
            ray.shutdown()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--agent-loop-config", type=Path, required=True)
    parser.add_argument("--eval-set", default="base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-token-count", type=int, default=16)
    parser.add_argument("--prompt-length", type=int, default=9000)
    parser.add_argument("--response-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--env-timeout", type=float, default=500.0)
    parser.add_argument("--guided", action="store_true")
    parser.add_argument("--critic-checkpoint", type=Path)
    parser.add_argument("--critic-qwen-hidden-dim", type=int)
    parser.add_argument("--critic-state-dim", type=int)
    parser.add_argument("--joint-alpha", type=float)
    parser.add_argument("--joint-beta", type=float)
    parser.add_argument("--joint-prior-temperature", type=float)
    parser.add_argument(
        "--joint-score-dtype",
        choices=("float32", "bfloat16", "float64"),
    )
    parser.add_argument("--joint-run-seed", type=int)
    parser.add_argument("--joint-snapshot-source-step", type=int)
    args = parser.parse_args(argv)
    if args.latent_token_count < 1:
        parser.error("--latent-token-count must be positive")
    if args.response_length < 1 or args.prompt_length < 1:
        parser.error("token lengths must be positive")
    if not 0.0 <= args.temperature:
        parser.error("--temperature must be non-negative")
    if not 0.0 < args.top_p <= 1.0:
        parser.error("--top-p must be in (0, 1]")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1)")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be positive")
    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        parser.error("--model must be a complete local HF checkpoint")
    if not args.agent_loop_config.is_file():
        parser.error("--agent-loop-config must exist")
    if args.guided:
        required = {
            "--critic-checkpoint": args.critic_checkpoint,
            "--critic-qwen-hidden-dim": args.critic_qwen_hidden_dim,
            "--critic-state-dim": args.critic_state_dim,
            "--joint-alpha": args.joint_alpha,
            "--joint-beta": args.joint_beta,
            "--joint-prior-temperature": args.joint_prior_temperature,
            "--joint-score-dtype": args.joint_score_dtype,
            "--joint-run-seed": args.joint_run_seed,
            "--joint-snapshot-source-step": args.joint_snapshot_source_step,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(
                "--guided requires explicit values for " + ", ".join(missing)
            )
        if not args.critic_checkpoint.is_dir():
            parser.error("--critic-checkpoint must be a checkpoint directory")
        if args.critic_qwen_hidden_dim < 1 or args.critic_state_dim < 1:
            parser.error("critic dimensions must be positive")
        if not math.isfinite(args.joint_alpha) or args.joint_alpha <= 0.0:
            parser.error("--joint-alpha must be finite and positive")
        if not math.isfinite(args.joint_beta) or args.joint_beta < 0.0:
            parser.error("--joint-beta must be finite and non-negative")
        if (
            not math.isfinite(args.joint_prior_temperature)
            or args.joint_prior_temperature <= 0.0
        ):
            parser.error("--joint-prior-temperature must be finite and positive")
        if args.joint_run_seed < 0 or args.joint_snapshot_source_step < 0:
            parser.error("joint run seed and snapshot source step must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
