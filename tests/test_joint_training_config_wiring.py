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

    def test_rejects_non_target_parallel_layout(self) -> None:
        from vagen.main_ppo import _configure_joint_actor_extension

        config = self._config()
        config.actor_rollout_ref.rollout.tensor_model_parallel_size = 4
        with self.assertRaisesRegex(ValueError, "DP8.*TP8"):
            _configure_joint_actor_extension(config)


if __name__ == "__main__":
    unittest.main()
