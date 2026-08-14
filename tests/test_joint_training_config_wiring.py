import os
from pathlib import Path
import subprocess
import sys
import unittest


class JointTrainingConfigWiringTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from omegaconf import OmegaConf  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"OmegaConf unavailable: {exc}")

    def _config(self):
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "joint_policy": {
                    "enabled": True,
                    "implementation": "frozen_q_guided_v1",
                    "alpha": 1.0,
                    "beta": 1.0,
                    "prior_temperature": 1.0,
                    "backprop_to_llm": True,
                    "score_dtype": "float32",
                },
                "joint_training": {
                    "enabled": True,
                    "implementation": "replicated_joint_update_v1",
                    "run_seed": 42,
                    "gamma": 0.9,
                    "gae_lambda": 0.8,
                    "ppo_clip_ratio": 0.2,
                    "normalize_advantages": True,
                    "token_kl_coefficient": 0.01,
                    "token_kl_type": "low_var_kl",
                    "guided_entropy_coefficient": 0.02,
                    "checkpoint_frequency": 10,
                    "actor_optimizer": {
                        "name": "adamw",
                        "lr": 1e-6,
                        "betas": [0.9, 0.95],
                        "eps": 1e-8,
                        "weight_decay": 0.01,
                        "grad_clip": 1.0,
                        "lr_scheduler_type": "constant",
                        "lr_warmup_steps": 0,
                        "lr_warmup_steps_ratio": 0.0,
                        "min_lr_ratio": None,
                        "num_cycles": 0.5,
                    },
                    "critic_checkpoint": "/checkpoints/id74",
                    "initial_snapshot_source_step": 776,
                    "critic_qwen_hidden_dim": 2048,
                    "critic_grid_tokens": 16,
                    "critic_state_dim": 1024,
                    "critic_action_count": 8,
                    "critic_huber_delta": 1.0,
                    "critic_grad_clip": 1.0,
                    "critic_optimizer": {
                        "name": "adamw",
                        "lr": 1e-4,
                        "betas": [0.9, 0.95],
                        "eps": 1e-8,
                        "weight_decay": 0.01,
                    },
                },
                "actor_rollout_ref": {
                    "actor": {
                        "strategy": "fsdp",
                        "ulysses_sequence_parallel_size": 1,
                        "ppo_epochs": 1,
                        "use_dynamic_bsz": False,
                        "shuffle": False,
                        "use_kl_loss": False,
                        "entropy_coeff": 0.0,
                        "use_fused_kernels": False,
                        "optim": {
                            "optimizer": "AdamW",
                            "optimizer_impl": "torch.optim",
                            "lr": 9e-9,
                            "betas": [0.9, 0.999],
                            "weight_decay": 0.0,
                            "clip_grad": 9.0,
                            "grad_clip": None,
                            "lr_scheduler_type": "cosine",
                            "lr_warmup_steps": 9,
                            "lr_warmup_steps_ratio": 0.0,
                            "min_lr_ratio": 0.1,
                            "num_cycles": 1.0,
                            "override_optimizer_config": None,
                        },
                    },
                    "model": {"use_fused_kernels": False},
                    "rollout": {
                        "name": "nimloth_vllm",
                        "mode": "async",
                        "tensor_model_parallel_size": 8,
                        "data_parallel_size": 1,
                    },
                },
                "trainer": {
                    "nnodes": 1,
                    "n_gpus_per_node": 8,
                    "critic_warmup": 0,
                    "default_hdfs_dir": None,
                    "remove_previous_ckpt_in_save": False,
                    "max_actor_ckpt_to_keep": 2,
                    "save_freq": -1,
                },
                "algorithm": {"use_kl_in_reward": False},
                "critic": {"enable": None},
                "filter": {"enable": False},
            }
        )

    def _enable_id170_gate(self, config, *, phase="update_1"):
        from omegaconf import open_dict

        model = (
            "/project/peilab/atst/nimloth/outputs/experiments/"
            "vagen_legacy_wm_k16_grid/2026-08-02/sft2/"
            "74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_"
            "ws16n3g844lw844_px100352/train_ws16/epoch_001"
        )
        with open_dict(config):
            config.joint_integration_gate = {
                "enabled": True,
                "implementation": "id170_dp8_resume_smoke_v1",
                "experiment_id": 170,
                "phase": phase,
            }
            config.decision_ledger = {"enabled": True}
            config.data = {
                "train_batch_size": 8,
                "train_files": "/repo/train_navigation_joint_id170.yaml",
            }
        with open_dict(config.joint_training):
            config.joint_training.run_seed = 42001
            config.joint_training.gamma = 0.99
            config.joint_training.gae_lambda = 0.95
            config.joint_training.token_kl_coefficient = 0.01
            config.joint_training.guided_entropy_coefficient = 0.01
            config.joint_training.critic_checkpoint = model
            config.joint_training.initial_snapshot_source_step = 776
            config.joint_training.critic_qwen_hidden_dim = 2048
            config.joint_training.critic_grid_tokens = 16
            config.joint_training.critic_state_dim = 1024
            config.joint_training.critic_action_count = 8
            config.joint_training.checkpoint_frequency = 1
            config.joint_training.actor_optimizer = {
                "name": "adamw",
                "lr": 1.0e-7,
                "betas": [0.9, 0.95],
                "eps": 1.0e-8,
                "weight_decay": 0.01,
                "grad_clip": 1.0,
                "lr_scheduler_type": "constant",
                "lr_warmup_steps": 0,
                "lr_warmup_steps_ratio": 0.0,
                "min_lr_ratio": None,
                "num_cycles": 0.5,
            }
            config.joint_training.critic_optimizer = {
                "name": "adamw",
                "lr": 1.0e-4,
                "betas": [0.9, 0.95],
                "eps": 1.0e-8,
                "weight_decay": 0.01,
            }
        with open_dict(config.actor_rollout_ref.model):
            config.actor_rollout_ref.model.path = model
        with open_dict(config.actor_rollout_ref.actor):
            config.actor_rollout_ref.actor.freeze_vision_tower = True
            config.actor_rollout_ref.actor.ppo_mini_batch_size = 8
            config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
        with open_dict(config.actor_rollout_ref.rollout):
            config.actor_rollout_ref.rollout.n = 1
            config.actor_rollout_ref.rollout.enforce_eager = True
            config.actor_rollout_ref.rollout.engine_kwargs = {
                "vllm": {"mm_encoder_tp_mode": "data"}
            }
        with open_dict(config.trainer):
            config.trainer.total_training_steps = 1 if phase == "update_1" else 2
            config.trainer.total_epochs = 1 if phase == "update_1" else 2
            config.trainer.resume_mode = "disable" if phase == "update_1" else "auto"
            config.trainer.project_name = "vagen"
            config.trainer.experiment_name = (
                "170_smoke_vagenlite_jointupdate_dp8_tp8_gate"
            )
            config.trainer.logger = ["console", "wandb"]
            config.trainer.val_before_train = False
            config.trainer.test_freq = -1
            config.trainer.default_local_dir = "/outputs/id170/checkpoints"
            config.trainer.concat_multi_turn = False
        return config

    def test_external_lib_registers_worker_rollout_class_in_fresh_process(self) -> None:
        script = """
from verl.utils.import_utils import import_external_libs
from verl.workers.rollout.base import get_rollout_class
import_external_libs('vagen.rollout.nimloth_vllm')
cls = get_rollout_class('nimloth_vllm', 'async')
assert cls.__name__ == 'vLLMAsyncRollout', cls
"""
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        subprocess.run([sys.executable, "-c", script], check=True, env=env)

    def test_id170_config_constructs_disabled_hf_upload_manager(self) -> None:
        from omegaconf import OmegaConf
        from vagen.utils.upload_hugging_face import HFUploadManager

        config_path = Path(__file__).parents[1] / "vagen/configs/joint_id170_gate.yaml"
        source_config = OmegaConf.load(config_path)
        runtime_config = OmegaConf.create(
            {
                "huggingface_hub": source_config.huggingface_hub,
                "trainer": {
                    "default_local_dir": "/tmp/id170-checkpoints",
                    "project_name": "vagen",
                    "experiment_name": "id170-config-test",
                },
            }
        )
        manager = HFUploadManager(runtime_config)
        self.assertIsNone(manager._hf_save_freq)

    def test_installs_custom_actor_and_exact_optimizer_values(self) -> None:
        from vagen.main_ppo import _configure_joint_actor_extension

        config = self._config()
        training = _configure_joint_actor_extension(config)
        self.assertEqual(training.initial_snapshot_source_step, 776)
        self.assertFalse(config.critic.enable)
        self.assertEqual(config.trainer.save_freq, 10)
        self.assertEqual(
            config.actor_rollout_ref.actor.custom_cls.name,
            "JointDataParallelPPOActor",
        )
        optim = config.actor_rollout_ref.actor.optim
        self.assertEqual(optim.lr, 1e-6)
        self.assertEqual(tuple(optim.betas), (0.9, 0.95))
        self.assertEqual(optim.override_optimizer_config, {"eps": 1e-8})
        self.assertEqual(config.actor_rollout_ref.actor.grad_clip, 1.0)
        from verl.utils.import_utils import load_extern_type
        from verl.workers.actor.dp_actor import DataParallelPPOActor

        custom = config.actor_rollout_ref.actor.custom_cls
        actor_type = load_extern_type(custom.path, custom.name)
        self.assertTrue(issubclass(actor_type, DataParallelPPOActor))

    def test_allows_only_human_approved_id170_integration_gate(self) -> None:
        from vagen.main_ppo import _configure_joint_actor_extension

        config = self._enable_id170_gate(self._config())
        training = _configure_joint_actor_extension(config)
        self.assertEqual(training.run_seed, 42001)
        self.assertEqual(config.trainer.save_freq, 1)
        config = self._enable_id170_gate(self._config())
        config.joint_integration_gate.experiment_id = 169
        with self.assertRaisesRegex(ValueError, "170"):
            _configure_joint_actor_extension(config)

    def test_rejects_non_target_parallel_layout(self) -> None:
        from vagen.main_ppo import _configure_joint_actor_extension

        config = self._config()
        config.actor_rollout_ref.rollout.tensor_model_parallel_size = 4
        with self.assertRaisesRegex(ValueError, "DP8.*TP8"):
            _configure_joint_actor_extension(config)


if __name__ == "__main__":
    unittest.main()
