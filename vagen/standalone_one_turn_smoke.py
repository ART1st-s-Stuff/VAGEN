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
    summarize_decision_ledger_batch,
    validate_decision_ledger_reward_rows,
)


def build_config(args: argparse.Namespace) -> Any:
    """Construct only model/rollout/environment config; no optimizer exists."""

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
                    "tensor_model_parallel_size": 1,
                    "data_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "expert_parallel_size": 1,
                    "max_num_batched_tokens": args.prompt_length
                    + args.response_length,
                    "max_num_seqs": 1,
                    "enable_chunked_prefill": True,
                    "enable_prefix_caching": False,
                    "engine_kwargs": {"vllm": {}},
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
                "n_gpus_per_node": 1,
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
            "joint_policy": {"enabled": False},
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
        allowed_schemas={DECISION_LEDGER_SCHEMA},
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
        or ledger["decision_sources"] != ["llm_text"]
        or ledger["decision_is_policy_sampled"] != [False]
    ):
        raise RuntimeError("ledger does not match the Navigation M1 action contract")
    if not ledger["format_valid"] or len(ledger["executed_action_ids"]) != 1:
        raise RuntimeError(
            "K-slot response was not accepted as exactly one environment action"
        )
    if int(result.non_tensor_batch["turn_idx"][0]) != 1:
        raise RuntimeError("one-turn smoke produced an unexpected turn index")

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
    action_token = f"<|action_({ledger['executed_action_ids'][0]})|>"
    if action_token not in environment_response:
        raise RuntimeError("executed action does not match generated action token")
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    import ray
    import vagen.rollout.nimloth_vllm  # noqa: F401 -- registers replica
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    from verl.utils import hf_tokenizer
    from vagen.agent_loop.agent_loop_no_concat import AgentLoopManager

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite smoke output: {args.output}")
    os.environ.setdefault("VLLM_USE_V1", "1")
    runtime_env = get_ppo_ray_runtime_env()
    runtime_env.setdefault("env_vars", {})["VLLM_USE_V1"] = "1"
    manager = None
    try:
        ray.init(runtime_env=runtime_env)
        config = build_config(args)
        manager = AgentLoopManager(config, worker_group=None)
        result = manager.generate_sequences(build_input(args))
        tokenizer = hf_tokenizer(str(args.model), trust_remote_code=True)
        payload = validate_result(result, tokenizer)
        payload["model"] = str(args.model)
        payload["env_url"] = args.env_url
        payload["eval_set"] = args.eval_set
        payload["seed"] = args.seed
        payload["latent_token_count"] = args.latent_token_count
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
    parser.add_argument("--env-timeout", type=float, default=500.0)
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
    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        parser.error("--model must be a complete local HF checkpoint")
    if not args.agent_loop_config.is_file():
        parser.error("--agent-loop-config must exist")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
