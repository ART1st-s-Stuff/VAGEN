import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class K4ActorUpdateTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch.distributed as dist
        except ImportError as exc:
            self.skipTest(f"torch unavailable: {exc}")
        if dist.is_initialized():
            self.skipTest("test requires a fresh process group")
        self._temporary = tempfile.TemporaryDirectory()
        dist.init_process_group(
            "gloo",
            init_method=f"file://{self._temporary.name}/process-group",
            rank=0,
            world_size=1,
        )

    def tearDown(self) -> None:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
        self._temporary.cleanup()

    def _training(self):
        from vagen.joint_policy.training_contract import JointTrainingConfig

        return JointTrainingConfig.from_mapping(
            {
                "implementation": "replicated_joint_update_v1",
                "run_seed": 42,
                "gamma": 1.0,
                "gae_lambda": 0.95,
                "ppo_clip_ratio": 0.2,
                "normalize_advantages": True,
                "token_kl_coefficient": 0.01,
                "token_kl_type": "low_var_kl",
                "guided_entropy_coefficient": 0.01,
                "checkpoint_frequency": 1,
                "actor_optimizer": {
                    "name": "adamw", "lr": 1e-2, "betas": [0.9, 0.95],
                    "eps": 1e-8, "weight_decay": 0.0, "grad_clip": 10.0,
                    "lr_scheduler_type": "constant", "lr_warmup_steps": 0,
                    "lr_warmup_steps_ratio": 0.0, "min_lr_ratio": None,
                    "num_cycles": 0.5,
                },
                "critic_checkpoint": "/unused",
                "initial_snapshot_source_step": 776,
                "critic_qwen_hidden_dim": 2,
                "critic_grid_tokens": 16,
                "critic_state_dim": 1024,
                "critic_action_count": 2,
                "critic_huber_delta": 1.0,
                "critic_grad_clip": 10.0,
                "critic_optimizer": {
                    "name": "adamw", "lr": 1e-2, "betas": [0.9, 0.95],
                    "eps": 1e-8, "weight_decay": 0.0,
                },
            }
        )

    def _world_model_config(self):
        from vagen.joint_policy.k4_training_contract import K4WorldModelTrainingConfig

        return K4WorldModelTrainingConfig.from_mapping(
            {
                "implementation": "k4_world_model_update_v1",
                "planning_checkpoint": "/unused",
                "snapshot_transport_root": self._temporary.name,
                "prediction_horizon": 4,
                "minimum_window_depth": 1,
                "maximum_window_depth": 4,
                "state_mse_weight": 1.0,
                "dino_grid_weight": 0.5,
                "sigreg_weight": 0.1,
                "sigreg_knots": 17,
                "sigreg_num_proj": 1024,
                "dino_identity": {
                    "source": "facebook/dinov2-large",
                    "revision": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
                    "processor_fingerprint": "7d65a7de8788e87d",
                    "hidden_size": 1024,
                    "grid_size": 4,
                },
                "selected_action_huber_delta": 1.0,
                "grad_clip": 10.0,
                "optimizer": {
                    "name": "adamw", "projector_lr": 1e-2,
                    "predictor_lr": 1e-2, "value_head_lr": 1e-2,
                    "betas": [0.9, 0.95], "eps": 1e-8,
                    "weight_decay": 0.0,
                },
            }
        )

    def _policy(self):
        from vagen.joint_policy.planning_contract import K4MCTSGuidedPolicyConfig

        return K4MCTSGuidedPolicyConfig.from_mapping(
            {
                "implementation": "k4_mcts_guided_v1",
                "alpha": 1.0,
                "beta": 85.78297006578457,
                "prior_temperature": 1.0,
                "backprop_to_llm": True,
                "score_dtype": "float32",
                "planning_horizon": 4,
                "mcts_num_simulations": 100,
                "mcts_exploration_constant": 1.0,
            }
        )

    def test_sigreg_microbatches_are_count_weighted_once(self) -> None:
        import torch
        from vagen.joint_policy.actor import _weighted_k4_sigreg_loss

        first = _weighted_k4_sigreg_loss(
            torch.tensor(3.0), torch.tensor(2), torch.tensor(4)
        )
        second = _weighted_k4_sigreg_loss(
            torch.tensor(5.0), torch.tensor(2), torch.tensor(4)
        )
        self.assertEqual(float(first + second), 4.0)

    def test_one_update_changes_actor_projector_predictor_and_value_head(self) -> None:
        import numpy as np
        import torch
        from torch import nn
        from torch.nn.parallel import DistributedDataParallel
        from verl import DataProto
        from nimloth.training.rl.joint_planner import JointWorldModelCritic
        from nimloth.wm import SequenceSIGReg
        from nimloth.wm.grid import (
            GridPredictorConfig,
            SharedSlotProjector,
            TemporalSpatialGridPredictor,
        )
        from nimloth.wm.value_head import ValueHead
        from vagen.joint_policy.actor import JointDataParallelPPOActor
        from vagen.joint_policy.k4_world_model_update import (
            K4WorldModelUpdateModule,
            build_k4_planning_optimizer,
        )

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.linspace(-0.3, 0.3, 7))

            def forward(self, *, input_ids, **_kwargs):
                logits = self.bias.view(1, 1, -1).expand(
                    input_ids.shape[0], input_ids.shape[1], -1
                )
                return SimpleNamespace(logits=logits)

        class FakeSIGReg(nn.Module):
            def forward(self, sequence):
                return sequence.square().mean()

        class Teacher:
            def load_images(self, images, *, device):
                return torch.zeros(len(images), 16, 1024, device=device)

        model = JointWorldModelCritic(
            state_projector=SharedSlotProjector(
                input_dim=2, output_dim=1024, hidden_dim=3, grid_tokens=16
            ),
            wm_predictor=TemporalSpatialGridPredictor(
                GridPredictorConfig(
                    grid_tokens=16, emb_dim=1024, action_dim=2,
                    history_size=1, depth=1, heads=1, dim_head=4,
                    mlp_dim=8, dropout=0.0,
                )
            ),
            value_head=ValueHead(emb_dim=1024, num_actions=2, hidden_dim=3),
        )
        planning = K4WorldModelUpdateModule(model, self._world_model_config())
        planning.sigreg = SequenceSIGReg(regularizer=FakeSIGReg())
        actor = object.__new__(JointDataParallelPPOActor)
        actor.actor_module = TinyLM()
        actor.actor_optimizer = torch.optim.AdamW(
            actor.actor_module.parameters(), lr=1e-2, weight_decay=0.0
        )
        actor.scaler = None
        actor.use_remove_padding = False
        actor.use_fused_kernels = False
        actor.device_name = "cpu"
        actor.param_dtype = torch.float32
        actor.config = SimpleNamespace(
            ppo_mini_batch_size=2,
            ppo_micro_batch_size_per_gpu=2,
            use_kl_loss=False,
            entropy_coeff=0.0,
            entropy_checkpointing=False,
            grad_clip=10.0,
        )
        actor.joint_training = self._training()
        actor.joint_policy = self._policy()
        actor.k4_world_model_training = planning.config
        actor._joint_world_size = 1
        actor._joint_rank = 0
        actor._joint_completed_updates = 0
        actor._joint_contract_id = None
        actor._last_k4_transport = None
        actor.current_k4_world_model = DistributedDataParallel(planning)
        actor.joint_planning_optimizer = build_k4_planning_optimizer(planning)
        actor.k4_dino_grid_targets = Teacher()
        actor.current_joint_critic = None
        actor.joint_critic_optimizer = None

        action_logits = actor.actor_module.bias.detach()[torch.tensor([1, 5])]
        behavior_log_probs = torch.log_softmax(action_logits, dim=-1)
        future_mask = torch.tensor(
            [[True, False, False, False], [True, False, False, False]]
        )
        images = np.array(
            [["next-0", None, None, None], ["next-1", None, None, None]],
            dtype=object,
        )
        data = DataProto.from_dict(
            tensors={
                "input_ids": torch.tensor([[3, 4, 1, 2], [4, 3, 2, 5]]),
                "attention_mask": torch.ones(2, 4, dtype=torch.long),
                "position_ids": torch.arange(4).expand(2, -1),
                "responses": torch.tensor([[1, 2], [2, 5]]),
                "response_mask": torch.ones(2, 2, dtype=torch.long),
                "joint_action_token_ids": torch.tensor([[1, 5], [1, 5]]),
                "joint_prior_token_ids": torch.tensor([1, 5]),
                "joint_prior_response_indices": torch.tensor([0, 1]),
                "joint_guided_action_ids": torch.tensor([0, 1]),
                "joint_behavior_guided_log_probs": behavior_log_probs,
                "joint_frozen_planner_root_mean_values": torch.zeros(2, 2),
                "joint_frozen_direct_all_action_q": torch.zeros(2, 2),
                "joint_advantages": torch.tensor([1.0, -1.0]),
                "joint_valid_mask": torch.tensor([True, True]),
                "joint_critic_hidden": torch.randn(2, 16, 2),
                "joint_critic_returns": torch.tensor([1.0, 0.0]),
                "joint_wm_future_hidden": torch.randn(2, 4, 16, 2),
                "joint_wm_future_action_ids": torch.tensor(
                    [[0, 0, 0, 0], [1, 0, 0, 0]]
                ),
                "joint_wm_future_valid_mask": future_mask,
                "joint_reference_token_log_probs": torch.full((2, 2), -2.0),
            },
            non_tensors={"joint_wm_future_images": images},
            meta_info={"temperature": 1.0},
        )
        actor_before = actor.actor_module.bias.detach().clone()
        planning_before = {
            key: value.detach().clone()
            for key, value in planning.model.state_dict().items()
        }

        def pure_log_probs(logits, labels, **_kwargs):
            return torch.log_softmax(logits, dim=-1).gather(
                -1, labels.unsqueeze(-1)
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
        self.assertFalse(torch.equal(actor.actor_module.bias.detach(), actor_before))
        for prefix in ("state_proj.", "wm_predictor.", "value_head."):
            self.assertTrue(
                any(
                    key.startswith(prefix)
                    and not torch.equal(value.detach(), planning_before[key])
                    for key, value in planning.model.state_dict().items()
                ),
                prefix,
            )
        self.assertEqual(metrics["wm/window_count"], [2.0])
        self.assertGreater(metrics["wm/sigreg_loss_sum"][0], 0.0)
        self.assertIn("planning/grad_norm", metrics)


if __name__ == "__main__":
    unittest.main()
