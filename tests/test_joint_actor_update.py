import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class JointActorUpdateTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch
            import torch.distributed as dist
        except ImportError as exc:
            self.skipTest(f"torch unavailable: {exc}")
        if dist.is_initialized():
            self.skipTest("test requires ownership of a fresh process group")
        self._temporary = tempfile.TemporaryDirectory()
        self._dist_path = f"{self._temporary.name}/process-group"
        dist.init_process_group(
            "gloo",
            init_method=f"file://{self._dist_path}",
            rank=0,
            world_size=1,
        )

    def tearDown(self) -> None:
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
            self._temporary.cleanup()
        except ImportError:
            pass

    def _training_config(self):
        from vagen.joint_policy.training_contract import JointTrainingConfig

        return JointTrainingConfig.from_mapping(
            {
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
                    "lr": 1e-2,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                    "grad_clip": 10.0,
                    "lr_scheduler_type": "constant",
                    "lr_warmup_steps": 0,
                    "lr_warmup_steps_ratio": 0.0,
                    "min_lr_ratio": None,
                    "num_cycles": 0.5,
                },
                "critic_checkpoint": "/unused",
                "initial_snapshot_source_step": 776,
                "critic_qwen_hidden_dim": 2,
                "critic_grid_tokens": 1,
                "critic_state_dim": 2,
                "critic_action_count": 2,
                "critic_huber_delta": 1.0,
                "critic_grad_clip": 10.0,
                "critic_optimizer": {
                    "name": "adamw",
                    "lr": 1e-2,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                },
            }
        )

    def _policy(self):
        from vagen.joint_policy.contract import FrozenQGuidedPolicyConfig

        return FrozenQGuidedPolicyConfig.from_mapping(
            {
                "implementation": "frozen_q_guided_v1",
                "alpha": 1.0,
                "beta": 0.0,
                "prior_temperature": 1.0,
                "backprop_to_llm": True,
                "score_dtype": "float32",
            }
        )

    def test_one_update_backpropagates_actor_and_allreduces_current_critic(self) -> None:
        import torch
        from torch import nn
        from torch.nn.parallel import DistributedDataParallel

        from nimloth.training.rl.joint_critic import JointActionValueCritic
        from nimloth.wm.grid import SharedSlotProjector
        from nimloth.wm.value_head import ValueHead
        from verl import DataProto
        from vagen.joint_policy.actor import JointDataParallelPPOActor

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.linspace(-0.3, 0.3, 7))

            def forward(self, *, input_ids, **_kwargs):
                logits = self.bias.view(1, 1, -1).expand(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    -1,
                )
                return SimpleNamespace(logits=logits)

        actor = object.__new__(JointDataParallelPPOActor)
        actor.actor_module = TinyLM()
        actor.actor_optimizer = torch.optim.AdamW(
            actor.actor_module.parameters(),
            lr=1e-2,
            weight_decay=0.0,
        )
        actor.scaler = None
        actor.use_remove_padding = False
        actor.use_fused_kernels = False
        actor.device_name = "cpu"
        actor.param_dtype = torch.float32
        actor.config = SimpleNamespace(
            ppo_mini_batch_size=2,
            ppo_micro_batch_size_per_gpu=1,
            use_kl_loss=False,
            entropy_coeff=0.0,
            entropy_checkpointing=False,
            grad_clip=10.0,
        )
        actor.joint_training = self._training_config()
        actor.joint_policy = self._policy()
        actor._joint_world_size = 1
        actor._joint_rank = 0
        actor._joint_completed_updates = 0
        actor._joint_contract_id = None
        actor.current_joint_critic = DistributedDataParallel(
            JointActionValueCritic(
                state_projector=SharedSlotProjector(
                    input_dim=2,
                    output_dim=2,
                    hidden_dim=3,
                    grid_tokens=1,
                ),
                value_head=ValueHead(
                    emb_dim=2,
                    num_actions=2,
                    hidden_dim=3,
                ),
            )
        )
        actor.joint_critic_optimizer = torch.optim.AdamW(
            actor.current_joint_critic.parameters(),
            lr=1e-2,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )

        action_logits = actor.actor_module.bias.detach()[torch.tensor([1, 5])]
        behavior_log_probs = torch.log_softmax(action_logits, dim=-1)
        responses = torch.tensor([[1, 2], [2, 5]], dtype=torch.long)
        data = DataProto.from_dict(
            tensors={
                "input_ids": torch.tensor([[3, 4, 1, 2], [4, 3, 2, 5]]),
                "attention_mask": torch.ones(2, 4, dtype=torch.long),
                "position_ids": torch.arange(4).expand(2, -1),
                "responses": responses,
                "response_mask": torch.ones(2, 2, dtype=torch.long),
                "joint_action_token_ids": torch.tensor([[1, 5], [1, 5]]),
                "joint_prior_token_ids": torch.tensor([1, 5]),
                "joint_prior_response_indices": torch.tensor([0, 1]),
                "joint_guided_action_ids": torch.tensor([0, 1]),
                "joint_behavior_guided_log_probs": behavior_log_probs,
                "joint_frozen_all_action_q": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "joint_advantages": torch.tensor([1.0, -1.0]),
                "joint_valid_mask": torch.tensor([True, True]),
                "joint_critic_hidden": torch.tensor([[[0.2, -0.1]], [[0.5, 0.3]]]),
                "joint_critic_returns": torch.tensor([1.0, 0.0]),
                "joint_reference_token_log_probs": torch.full((2, 2), -2.0),
            },
            meta_info={"temperature": 1.0},
        )
        actor_before = actor.actor_module.bias.detach().clone()
        critic_before = {
            key: value.detach().clone()
            for key, value in actor.current_joint_critic.module.state_dict().items()
        }

        def pure_log_probs(logits, labels, **_kwargs):
            return torch.log_softmax(logits, dim=-1).gather(
                -1,
                labels.unsqueeze(-1),
            ).squeeze(-1)

        with (
            patch("vagen.joint_policy.actor.get_device_id", return_value=torch.device("cpu")),
            patch(
                "verl.workers.actor.dp_actor.logprobs_from_logits",
                side_effect=pure_log_probs,
            ),
        ):
            metrics = actor.update_policy(data)

        self.assertEqual(actor._joint_completed_updates, 1)
        self.assertEqual(metrics["joint/completed_updates"], [1.0])
        self.assertFalse(torch.equal(actor.actor_module.bias.detach(), actor_before))
        self.assertTrue(
            any(
                not torch.equal(value.detach(), critic_before[key])
                for key, value in actor.current_joint_critic.module.state_dict().items()
            )
        )
        exported = actor.export_joint_checkpoint(
            source_step=777,
            contract_id="contract-1",
            score_dtype="float32",
        )
        payload = exported["checkpoint_payload"]
        saved_critic = {
            key: value.detach().clone()
            for key, value in actor.current_joint_critic.module.state_dict().items()
        }
        with torch.no_grad():
            for parameter in actor.current_joint_critic.parameters():
                parameter.add_(1.0)
        restored = actor.load_joint_checkpoint(payload)
        self.assertEqual(restored["source_step"], 777)
        self.assertEqual(restored["completed_updates"], 1)
        self.assertTrue(
            all(
                torch.equal(value.detach(), saved_critic[key])
                for key, value in actor.current_joint_critic.module.state_dict().items()
            )
        )


if __name__ == "__main__":
    unittest.main()
