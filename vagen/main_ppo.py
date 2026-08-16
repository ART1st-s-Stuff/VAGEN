# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other mpain.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from .ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_type


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            # Add runtime environment variables for transfer queue
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)  # please make sure main_task is not scheduled on head

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


def _validate_joint_integration_gate_runtime(
    config,
    *,
    training,
    policy,
    k4_world_model,
    gate,
) -> None:
    """Keep each human-approved escape hatch narrower than production."""

    from vagen.joint_policy.integration_gate import (
        K4_ID179_INTEGRATION_GATE_IMPLEMENTATION,
        K4_ID180_INTEGRATION_GATE_IMPLEMENTATION,
        K4_ID181_INTEGRATION_GATE_IMPLEMENTATION,
        K4_ID182_INTEGRATION_GATE_IMPLEMENTATION,
        K4_ID183_CANARY_GATE_IMPLEMENTATION,
    )

    if gate.implementation in {
        K4_ID179_INTEGRATION_GATE_IMPLEMENTATION,
        K4_ID180_INTEGRATION_GATE_IMPLEMENTATION,
        K4_ID181_INTEGRATION_GATE_IMPLEMENTATION,
        K4_ID182_INTEGRATION_GATE_IMPLEMENTATION,
        K4_ID183_CANARY_GATE_IMPLEMENTATION,
    }:
        _validate_k4_integration_gate_runtime(
            config,
            training=training,
            policy=policy,
            world_model=k4_world_model,
            gate=gate,
        )
        return
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    trainer = config.trainer
    model = config.actor_rollout_ref.model
    logger_names = list(trainer.logger)
    expected_model = (
        "/project/peilab/atst/nimloth/outputs/experiments/"
        "vagen_legacy_wm_k16_grid/2026-08-02/sft2/"
        "74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_"
        "ws16n3g844lw844_px100352/train_ws16/epoch_001"
    )
    actor_optim = training.actor_optimizer
    critic_optim = training.critic_optimizer
    if (
        policy.alpha != 1.0
        or policy.beta != 1.0
        or policy.prior_temperature != 1.0
        or policy.score_dtype != "float32"
        or training.run_seed != 42001
        or training.gamma != 0.99
        or training.gae_lambda != 0.95
        or training.ppo_clip_ratio != 0.2
        or training.token_kl_coefficient != 0.01
        or training.guided_entropy_coefficient != 0.01
        or training.checkpoint_frequency != 1
        or training.initial_snapshot_source_step != 776
        or training.critic_qwen_hidden_dim != 2048
        or training.critic_grid_tokens != 16
        or training.critic_state_dim != 1024
        or training.critic_action_count != 8
        or training.critic_huber_delta != 1.0
        or training.critic_grad_clip != 1.0
        or actor_optim.lr != 1.0e-7
        or actor_optim.betas != (0.9, 0.95)
        or actor_optim.eps != 1.0e-8
        or actor_optim.weight_decay != 0.01
        or actor_optim.grad_clip != 1.0
        or actor_optim.lr_scheduler_type != "constant"
        or actor_optim.lr_warmup_steps != 0
        or actor_optim.lr_warmup_steps_ratio != 0.0
        or actor_optim.min_lr_ratio is not None
        or actor_optim.num_cycles != 0.5
        or critic_optim.lr != 1.0e-4
        or critic_optim.betas != (0.9, 0.95)
        or critic_optim.eps != 1.0e-8
        or critic_optim.weight_decay != 0.01
    ):
        raise ValueError("ID171 integration gate numerical contract mismatch")
    if int(trainer.total_training_steps) != gate.expected_total_training_steps:
        raise ValueError("ID171 integration gate total_training_steps mismatch")
    if int(trainer.total_epochs) != gate.expected_total_training_steps:
        raise ValueError("ID171 integration gate total_epochs mismatch")
    if trainer.resume_mode != gate.expected_resume_mode:
        raise ValueError("ID171 integration gate resume_mode mismatch")
    if trainer.project_name != "vagen" or not str(
        trainer.experiment_name
    ).startswith("171_smoke_vagenlite_jointupdate_dp8_tp8_"):
        raise ValueError("ID171 integration gate W&B identity mismatch")
    if set(logger_names) != {"console", "wandb"}:
        raise ValueError("ID171 integration gate requires console and W&B logging")
    if trainer.val_before_train or int(trainer.test_freq) != -1:
        raise ValueError("ID171 integration gate forbids validation rollout")
    if not str(trainer.default_local_dir).endswith("/checkpoints"):
        raise ValueError("ID171 integration gate checkpoint directory mismatch")
    if int(config.data.train_batch_size) != 8 or int(rollout.n) != 1:
        raise ValueError("ID171 integration gate requires 8 trajectories and rollout n=1")
    if not str(config.data.train_files).endswith(
        "train_navigation_joint_id171.yaml"
    ):
        raise ValueError("ID171 integration gate train split config mismatch")
    if not bool(actor.freeze_vision_tower):
        raise ValueError("ID171 integration gate requires frozen vision tower")
    if str(model.path) != expected_model or training.critic_checkpoint != expected_model:
        raise ValueError("ID171 integration gate checkpoint initialization mismatch")
    if int(actor.ppo_mini_batch_size) != 8 or int(
        actor.ppo_micro_batch_size_per_gpu
    ) != 1:
        raise ValueError("ID171 integration gate PPO batch layout mismatch")
    if not bool(rollout.enforce_eager):
        raise ValueError("ID171 integration gate requires eager vLLM")
    engine_kwargs = rollout.get("engine_kwargs", {})
    if engine_kwargs.get("vllm", {}).get("mm_encoder_tp_mode") != "data":
        raise ValueError("ID171 integration gate requires mm_encoder_tp_mode=data")
    from vagen.agent_loop.decision_ledger import parse_decision_ledger_enabled

    if not parse_decision_ledger_enabled(config.get("decision_ledger")):
        raise ValueError("ID171 integration gate requires decision ledger")
    if trainer.get("concat_multi_turn", True):
        raise ValueError("ID171 integration gate requires no-concat training")


def _validate_k4_integration_gate_runtime(
    config,
    *,
    training,
    policy,
    world_model,
    gate,
) -> None:
    from pathlib import Path
    from vagen.joint_policy import K4MCTSGuidedPolicyConfig

    expected_actor = (
        "/project/peilab/atst/nimloth/outputs/experiments/training/sft2/"
        "2026-08-15/176_id74_action_head_repair_balanced271x8_val40x8/"
        "checkpoint"
    )
    expected_planning = (
        "/project/peilab/atst/nimloth/outputs/experiments/"
        "vagen_legacy_wm_k16_grid/2026-08-02/sft2/"
        "74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_"
        "ws16n3g844lw844_px100352/train_ws16/epoch_001"
    )
    experiment_id = gate.experiment_id
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    trainer = config.trainer
    actor_optim = training.actor_optimizer
    critic_optim = training.critic_optimizer
    planning_optim = world_model.optimizer if world_model is not None else None
    from vagen.joint_policy.integration_gate import (
        K4_ID183_CANARY_GATE_IMPLEMENTATION,
    )

    is_canary = gate.implementation == K4_ID183_CANARY_GATE_IMPLEMENTATION
    expected_checkpoint_frequency = 5 if is_canary else 1
    if not isinstance(policy, K4MCTSGuidedPolicyConfig) or world_model is None:
        raise ValueError(
            f"ID{experiment_id} gate requires K4 policy and world-model training"
        )
    if (
        policy.alpha != 1.0
        or policy.beta != 85.78297006578457
        or policy.prior_temperature != 1.0
        or policy.score_dtype != "float32"
        or policy.planning_horizon != 4
        or policy.mcts_num_simulations != 100
        or policy.mcts_exploration_constant != 1.0
        or training.run_seed != 42179
        or training.gamma != 1.0
        or training.gae_lambda != 0.95
        or training.ppo_clip_ratio != 0.2
        or training.token_kl_coefficient != 0.01
        or training.guided_entropy_coefficient != 0.01
        or training.checkpoint_frequency != expected_checkpoint_frequency
        or training.initial_snapshot_source_step != 776
        or training.critic_qwen_hidden_dim != 2048
        or training.critic_grid_tokens != 16
        or training.critic_state_dim != 1024
        or training.critic_action_count != 8
        or training.critic_huber_delta != 1.0
        or training.critic_grad_clip != 1.0
        or actor_optim.lr != 1.0e-7
        or actor_optim.betas != (0.9, 0.95)
        or actor_optim.eps != 1.0e-8
        or actor_optim.weight_decay != 0.01
        or actor_optim.grad_clip != 1.0
        or actor_optim.lr_scheduler_type != "constant"
        or actor_optim.lr_warmup_steps != 0
        or actor_optim.lr_warmup_steps_ratio != 0.0
        or actor_optim.min_lr_ratio is not None
        or actor_optim.num_cycles != 0.5
        or critic_optim.lr != 1.0e-4
        or critic_optim.betas != (0.9, 0.95)
        or critic_optim.eps != 1.0e-8
        or critic_optim.weight_decay != 0.01
        or planning_optim.projector_lr != 1.0e-4
        or planning_optim.predictor_lr != 1.0e-4
        or planning_optim.value_head_lr != 1.0e-4
        or world_model.state_mse_weight != 1.0
        or world_model.dino_grid_weight != 0.5
        or world_model.sigreg_weight != 0.1
        or world_model.selected_action_huber_delta != 1.0
        or world_model.grad_clip != 1.0
    ):
        raise ValueError(
            f"ID{experiment_id} integration gate numerical contract mismatch"
        )
    if (
        str(config.actor_rollout_ref.model.path) != expected_actor
        or training.critic_checkpoint != expected_planning
        or world_model.planning_checkpoint != expected_planning
    ):
        raise ValueError(
            f"ID{experiment_id} integration gate checkpoint roots mismatch"
        )
    expected_snapshots = str(
        Path(str(trainer.default_local_dir)).parent / "planning_snapshots"
    )
    if world_model.snapshot_transport_root != expected_snapshots:
        raise ValueError(
            f"ID{experiment_id} integration gate snapshot root mismatch"
        )
    expected_epochs = gate.expected_total_training_steps if is_canary else 1
    if (
        int(trainer.total_training_steps) != gate.expected_total_training_steps
        or int(trainer.total_epochs) != expected_epochs
        or trainer.resume_mode != gate.expected_resume_mode
    ):
        raise ValueError(
            f"ID{experiment_id} integration gate phase runtime mismatch"
        )
    expected_run_prefix = (
        "183_canary_k4schemeb_jointupdate_dp8_tp8_"
        if is_canary
        else f"{experiment_id}_gate_k4schemeb_jointupdate_dp8_tp8_"
    )
    if trainer.project_name != "vagen" or not str(
        trainer.experiment_name
    ).startswith(expected_run_prefix):
        raise ValueError(
            f"ID{experiment_id} integration gate W&B identity mismatch"
        )
    if set(trainer.logger) != {"console", "wandb"}:
        raise ValueError(
            f"ID{experiment_id} integration gate requires console and W&B"
        )
    if is_canary:
        is_first_phase = gate.phase == "train_to_5"
        if (
            bool(trainer.val_before_train) != is_first_phase
            or int(trainer.test_freq) != (-1 if is_first_phase else 10)
            or bool(trainer.get("val_only", False))
        ):
            raise ValueError("ID183 canary validation phase mismatch")
        if (
            int(trainer.save_freq) != 5
            or int(trainer.max_actor_ckpt_to_keep) != 2
        ):
            raise ValueError("ID183 canary checkpoint schedule mismatch")
    elif trainer.val_before_train or int(trainer.test_freq) != -1:
        raise ValueError(
            f"ID{experiment_id} integration gate forbids validation rollout"
        )
    expected_train_file = f"train_navigation_joint_id{experiment_id}.yaml"
    if (
        int(config.data.train_batch_size) != 24
        or int(config.data.gen_batch_size) != 24
        or int(rollout.n) != 1
        or int(config.data.max_response_length) != 512
        or not str(config.data.train_files).endswith(expected_train_file)
        or (
            is_canary
            and (
                int(config.data.val_batch_size) != 40
                or not str(config.data.val_files).endswith(
                    "val_navigation_joint_id183.yaml"
                )
            )
        )
    ):
        raise ValueError(
            f"ID{experiment_id} integration gate rollout batch mismatch"
        )
    if (
        not bool(actor.freeze_vision_tower)
        or int(actor.ppo_mini_batch_size) != 24
        or int(actor.ppo_micro_batch_size_per_gpu) != 1
        or not bool(rollout.enforce_eager)
        or float(rollout.temperature) != 0.7
        or float(rollout.top_p) != 0.95
    ):
        raise ValueError(
            f"ID{experiment_id} integration gate actor/rollout mismatch"
        )
    if is_canary and (
        int(rollout.val_kwargs.n) != 1
        or float(rollout.val_kwargs.temperature) != 0.7
        or float(rollout.val_kwargs.top_p) != 0.95
        or not bool(rollout.val_kwargs.do_sample)
    ):
        raise ValueError("ID183 canary validation sampling mismatch")
    if rollout.get("engine_kwargs", {}).get("vllm", {}).get(
        "mm_encoder_tp_mode"
    ) != "data":
        raise ValueError(
            f"ID{experiment_id} gate requires mm_encoder_tp_mode=data"
        )
    if trainer.get("concat_multi_turn", True):
        raise ValueError(
            f"ID{experiment_id} integration gate requires no-concat training"
        )


def _configure_joint_actor_extension(config):
    """Install the explicit custom actor without enabling stock PPO fallback."""

    from omegaconf import OmegaConf, open_dict
    from vagen.joint_policy import (
        K4MCTSGuidedPolicyConfig,
        parse_joint_policy_section,
        parse_k4_world_model_training_section,
        validate_k4_joint_training_alignment,
    )
    from vagen.joint_policy.integration_gate import parse_joint_integration_gate
    from vagen.joint_policy.training_contract import parse_joint_training_section

    raw_training = config.get("joint_training", {"enabled": False})
    raw_policy = config.get("joint_policy", {"enabled": False})
    raw_k4_world_model = config.get(
        "k4_world_model_training",
        {"enabled": False},
    )
    if OmegaConf.is_config(raw_training):
        raw_training = OmegaConf.to_container(raw_training, resolve=True)
    if OmegaConf.is_config(raw_policy):
        raw_policy = OmegaConf.to_container(raw_policy, resolve=True)
    if OmegaConf.is_config(raw_k4_world_model):
        raw_k4_world_model = OmegaConf.to_container(
            raw_k4_world_model,
            resolve=True,
        )
    training = parse_joint_training_section(raw_training)
    policy = parse_joint_policy_section(raw_policy)
    k4_world_model = parse_k4_world_model_training_section(raw_k4_world_model)
    raw_gate = config.get("joint_integration_gate", {"enabled": False})
    if OmegaConf.is_config(raw_gate):
        raw_gate = OmegaConf.to_container(raw_gate, resolve=True)
    integration_gate = parse_joint_integration_gate(raw_gate)
    if (training is None) != (policy is None):
        raise ValueError(
            "joint_policy and joint_training must be enabled or disabled together"
        )
    if training is None:
        if integration_gate is not None:
            raise ValueError("joint integration gate requires enabled joint training")
        if k4_world_model is not None:
            raise ValueError("K4 world-model training requires joint training")
        return None
    is_k4 = isinstance(policy, K4MCTSGuidedPolicyConfig)
    if is_k4 != (k4_world_model is not None):
        raise ValueError(
            "K4 joint policy and k4_world_model_training must be enabled together"
        )
    if k4_world_model is not None:
        validate_k4_joint_training_alignment(training, k4_world_model)
    actor = config.actor_rollout_ref.actor
    model = config.actor_rollout_ref.model
    if config.actor_rollout_ref.rollout.name != "nimloth_vllm":
        raise ValueError("joint training requires rollout.name=nimloth_vllm")
    import vagen.rollout.nimloth_vllm  # noqa: F401 -- register driver replica

    if actor.strategy not in {"fsdp", "fsdp2"}:
        raise ValueError("joint training supports only FSDP actor strategy")
    if int(actor.get("ulysses_sequence_parallel_size", 1)) != 1:
        raise ValueError("joint training requires actor DP8 without sequence parallelism")
    if (
        int(config.trainer.nnodes) != 1
        or int(config.trainer.n_gpus_per_node) != 8
        or int(config.actor_rollout_ref.rollout.tensor_model_parallel_size) != 8
        or int(config.actor_rollout_ref.rollout.get("data_parallel_size", 1)) != 1
    ):
        raise ValueError(
            "joint training requires one node, actor DP8, rollout TP8, and rollout DP1"
        )
    if actor.ppo_epochs != 1:
        raise ValueError("joint training requires exactly one PPO epoch")
    if config.trainer.critic_warmup != 0:
        raise ValueError("joint training requires trainer.critic_warmup=0")
    if actor.use_dynamic_bsz:
        raise ValueError("joint training dynamic actor batching is not implemented")
    if actor.shuffle:
        raise ValueError("joint training requires deterministic actor.shuffle=false")
    if actor.use_kl_loss or float(actor.entropy_coeff) != 0.0:
        raise ValueError(
            "joint training requires stock actor KL and entropy to be disabled"
        )
    if bool(model.get("use_fused_kernels", False)) or bool(
        actor.get("use_fused_kernels", False)
    ):
        raise ValueError("joint training requires actor fused kernels disabled")
    if config.algorithm.use_kl_in_reward:
        raise ValueError("joint training forbids stock KL-in-reward")
    if config.filter.get("enable", False):
        raise ValueError("joint training does not support post-return row filtering")
    if config.trainer.default_hdfs_dir is not None:
        raise ValueError("joint exact checkpoint/resume currently requires local shared storage")
    if integration_gate is not None:
        _validate_joint_integration_gate_runtime(
            config,
            training=training,
            policy=policy,
            k4_world_model=k4_world_model,
            gate=integration_gate,
        )
    if config.trainer.get("remove_previous_ckpt_in_save", False):
        raise ValueError("joint checkpointing forbids remove_previous_ckpt_in_save")
    keep = config.trainer.get("max_actor_ckpt_to_keep", None)
    if keep is not None and keep < 2:
        raise ValueError("joint checkpointing must retain at least two actor checkpoints")
    custom_policy = {
        field: value
        for field, value in raw_policy.items()
        if field != "enabled"
    }
    with open_dict(config.critic):
        config.critic.enable = False
    with open_dict(config.trainer):
        config.trainer.save_freq = training.checkpoint_frequency
    actor_optim = training.actor_optimizer
    with open_dict(actor.optim):
        actor.optim.optimizer = "AdamW"
        actor.optim.optimizer_impl = "torch.optim"
        actor.optim.lr = actor_optim.lr
        actor.optim.betas = list(actor_optim.betas)
        actor.optim.weight_decay = actor_optim.weight_decay
        actor.optim.clip_grad = actor_optim.grad_clip
        actor.optim.grad_clip = None
        actor.optim.lr_scheduler_type = actor_optim.lr_scheduler_type
        actor.optim.lr_warmup_steps = actor_optim.lr_warmup_steps
        actor.optim.lr_warmup_steps_ratio = actor_optim.lr_warmup_steps_ratio
        actor.optim.min_lr_ratio = actor_optim.min_lr_ratio
        actor.optim.num_cycles = actor_optim.num_cycles
        actor.optim.override_optimizer_config = {"eps": actor_optim.eps}
    with open_dict(actor):
        actor.grad_clip = actor_optim.grad_clip
        actor.custom_cls = {
            "path": "pkg://vagen.joint_policy.actor",
            "name": "JointDataParallelPPOActor",
        }
        actor.custom_config = {
            "joint_policy": custom_policy,
            "joint_training": raw_training,
            **(
                {"k4_world_model_training": raw_k4_world_model}
                if k4_world_model is not None
                else {}
            ),
        }
    return training


class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy."""
        from verl.single_controller.ray import RayWorkerGroup

        joint_enabled = bool(config.get("joint_training", {}).get("enabled", False))
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            if joint_enabled:
                if config.actor_rollout_ref.rollout.mode != "async":
                    raise ValueError("joint training requires async actor rollout worker")
                from vagen.joint_policy.worker import JointAsyncActorRolloutRefWorker

                actor_rollout_cls = JointAsyncActorRolloutRefWorker
            else:
                actor_rollout_cls = (
                    AsyncActorRolloutRefWorker
                    if config.actor_rollout_ref.rollout.mode == "async"
                    else ActorRolloutRefWorker
                )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)

        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""
        from verl.trainer.ppo.ray_trainer import Role

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # TODO Here you can use the new registration method to support dynamic registration of roles
        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool

        self.mapping[Role.ActorRollout] = global_pool_id
        self.mapping[Role.Critic] = global_pool_id
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def add_reward_model_worker(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward_model.enable:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                    from verl.workers.fsdp_workers import RewardModelWorker
                elif config.reward_model.strategy == "megatron":
                    from verl.workers.megatron_workers import RewardModelWorker
                else:
                    raise NotImplementedError
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import RewardModelWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            if config.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used."""
        from verl.trainer.ppo.ray_trainer import Role

        joint_token_kl = float(
            config.get("joint_training", {}).get("token_kl_coefficient", 0.0)
        )
        if (
            config.algorithm.use_kl_in_reward
            or config.actor_rollout_ref.actor.use_kl_loss
            or joint_token_kl > 0.0
        ):
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        joint_training = _configure_joint_actor_extension(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        if joint_training is None:
            self.add_critic_worker(config)

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        self.add_reward_model_worker(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # Keep stock actor/data validation, but never create its scalar critic
        # when the replicated action-value critic extension is enabled.
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=(
                need_critic(config) if joint_training is None else False
            ),
        )

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()

        # Start the training process.
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        # If a data generation strategy is specified, use the DynamicGenDataset class
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

        dataset_cls = DynamicGenDataset
        print("Using DynamicGenDataset for data generation.")
    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_type(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        seed = data_config.get("seed")
        if seed is not None:
            train_dataloader_generator.manual_seed(seed)
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
