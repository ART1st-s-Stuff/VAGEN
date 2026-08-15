import unittest


def _config(snapshot_transport_root="/tmp/snapshots"):
    from vagen.joint_policy.k4_training_contract import K4WorldModelTrainingConfig

    return K4WorldModelTrainingConfig.from_mapping(
        {
            "implementation": "k4_world_model_update_v1",
            "planning_checkpoint": "/tmp/id74",
            "snapshot_transport_root": snapshot_transport_root,
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
            "grad_clip": 1.0,
            "optimizer": {
                "name": "adamw",
                "projector_lr": 1e-4,
                "predictor_lr": 1e-4,
                "value_head_lr": 1e-4,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        }
    )


class K4WorldModelUpdateTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"torch unavailable: {exc}")

    def _module(self):
        import torch
        from torch import nn

        from nimloth.training.rl.joint_planner import JointWorldModelCritic
        from nimloth.wm import SequenceSIGReg
        from nimloth.wm.grid import (
            GridPredictorConfig,
            SharedSlotProjector,
            TemporalSpatialGridPredictor,
        )
        from nimloth.wm.value_head import ValueHead
        from vagen.joint_policy.k4_world_model_update import K4WorldModelUpdateModule

        model = JointWorldModelCritic(
            state_projector=SharedSlotProjector(
                input_dim=2,
                output_dim=1024,
                hidden_dim=3,
                grid_tokens=16,
            ),
            wm_predictor=TemporalSpatialGridPredictor(
                GridPredictorConfig(
                    grid_tokens=16,
                    emb_dim=1024,
                    action_dim=2,
                    history_size=1,
                    depth=1,
                    heads=1,
                    dim_head=4,
                    mlp_dim=8,
                    dropout=0.0,
                )
            ),
            value_head=ValueHead(emb_dim=1024, num_actions=2, hidden_dim=3),
        )
        module = K4WorldModelUpdateModule(model, _config())

        class FakeSIGReg(nn.Module):
            def forward(self, sequence):
                return sequence.square().mean()

        module.sigreg = SequenceSIGReg(regularizer=FakeSIGReg())
        return module

    def test_all_valid_prefix_windows_and_three_modules_receive_gradients(self) -> None:
        import torch

        module = self._module()
        batch_size = 2
        current = torch.randn(batch_size, 16, 2)
        future = torch.randn(batch_size, 4, 16, 2)
        valid = torch.tensor(
            [[True, True, False, False], [True, False, False, False]]
        )
        outputs = module(
            current_hidden=current,
            future_hidden=future,
            future_action_ids=torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0]]),
            future_valid_mask=valid,
            valid_row_mask=torch.tensor([True, True]),
            guided_action_ids=torch.tensor([0, 1]),
            critic_returns=torch.tensor([1.0, -0.5]),
            future_dino_grid_targets=torch.randn(batch_size, 4, 16, 1024),
            sigreg_seed=123,
        )
        self.assertEqual(int(outputs.window_count), 3)
        self.assertEqual(int(outputs.critic_valid_count), 2)
        self.assertEqual(int(outputs.sigreg_valid_count), 2)
        loss = (
            outputs.state_window_loss_sum / outputs.window_count
            + 0.5 * outputs.dino_window_loss_sum / outputs.window_count
            + outputs.critic_loss_sum / outputs.critic_valid_count
            + 0.1 * outputs.sigreg_loss
        )
        loss.backward()
        for child in (
            module.model.state_proj,
            module.model.wm_predictor,
            module.model.value_head,
        ):
            self.assertTrue(
                any(
                    parameter.grad is not None
                    and torch.isfinite(parameter.grad).all()
                    and float(parameter.grad.abs().sum()) > 0.0
                    for parameter in child.parameters()
                )
            )

    def test_one_optimizer_has_exact_named_groups(self) -> None:
        from vagen.joint_policy.k4_world_model_update import (
            build_k4_planning_optimizer,
        )

        optimizer = build_k4_planning_optimizer(self._module())
        self.assertEqual(
            [group["name"] for group in optimizer.param_groups],
            ["state_projector", "wm_predictor", "value_head"],
        )
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [1e-4, 1e-4, 1e-4],
        )
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.95))
        self.assertEqual(optimizer.defaults["eps"], 1e-8)
        self.assertEqual(optimizer.defaults["weight_decay"], 0.01)

    def test_actor_exports_rank_zero_full_k4_transport(self) -> None:
        import tempfile
        from types import SimpleNamespace

        from vagen.joint_policy.actor import JointDataParallelPPOActor
        from vagen.joint_policy.k4_world_model_update import (
            build_k4_planning_optimizer,
        )
        from vagen.joint_policy.planning_contract import (
            K4MCTSGuidedPolicyConfig,
        )

        with tempfile.TemporaryDirectory() as temporary:
            module = self._module()
            module.config = _config(temporary)
            actor = object.__new__(JointDataParallelPPOActor)
            actor.joint_policy = K4MCTSGuidedPolicyConfig.from_mapping(
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
            actor.joint_training = SimpleNamespace(
                initial_snapshot_source_step=776
            )
            actor.k4_world_model_training = module.config
            actor.current_k4_world_model = SimpleNamespace(module=module)
            actor.joint_planning_optimizer = build_k4_planning_optimizer(module)
            actor._joint_rank = 0
            actor._joint_world_size = 1
            actor._joint_completed_updates = 1
            actor._joint_contract_id = None
            actor._last_k4_transport = None
            export = actor.export_joint_critic_snapshot(
                source_step=777,
                contract_id="contract-1",
                score_dtype="float32",
            )
            state = export["snapshot_state"]
            self.assertEqual(state["snapshot_source_step"], 777)
            self.assertEqual(state["planning_horizon"], 4)
            self.assertTrue(__import__("pathlib").Path(state["transport_path"]).is_file())
            repeated = actor.export_joint_critic_snapshot(
                source_step=777,
                contract_id="contract-1",
                score_dtype="float32",
            )
            self.assertEqual(repeated["snapshot_id"], export["snapshot_id"])
            checkpoint = actor.export_joint_checkpoint(
                source_step=777,
                contract_id="contract-1",
                score_dtype="float32",
            )
            payload = checkpoint["checkpoint_payload"]
            self.assertEqual(
                payload["schema"],
                "vagen_joint_k4_actor_planning_checkpoint_v1",
            )
            self.assertEqual(
                payload["snapshot_transport"]["snapshot_id"],
                export["snapshot_id"],
            )
            self.assertEqual(
                payload["planning_optimizer_fingerprint"],
                export["optimizer_fingerprint"],
            )

    def test_actor_dino_table_encodes_only_valid_future_images(self) -> None:
        import numpy as np
        import torch

        from vagen.joint_policy.actor import _k4_dino_target_tensor

        class Teacher:
            def __init__(self):
                self.images = None

            def load_images(self, images, *, device):
                self.images = list(images)
                return torch.arange(len(images) * 2, dtype=torch.float32).reshape(
                    len(images), 1, 2
                ).to(device)

        teacher = Teacher()
        images = np.array(
            [["a", "b", None, None], ["c", None, None, None]],
            dtype=object,
        )
        result = _k4_dino_target_tensor(
            teacher,
            images,
            torch.tensor(
                [[True, True, False, False], [True, False, False, False]]
            ),
            device=torch.device("cpu"),
            grid_tokens=1,
            state_dim=2,
        )
        self.assertEqual(teacher.images, ["a", "b", "c"])
        self.assertEqual(result.shape, (2, 4, 1, 2))
        self.assertTrue(torch.equal(result[0, 0], torch.tensor([[0.0, 1.0]])))
        self.assertTrue(torch.equal(result[1, 0], torch.tensor([[4.0, 5.0]])))
        self.assertEqual(float(result[1, 1:].abs().sum()), 0.0)

    def test_rejects_nonprefix_future_mask(self) -> None:
        import torch

        module = self._module()
        with self.assertRaisesRegex(ValueError, "contiguous prefix"):
            module(
                current_hidden=torch.randn(1, 16, 2),
                future_hidden=torch.randn(1, 4, 16, 2),
                future_action_ids=torch.zeros(1, 4, dtype=torch.long),
                future_valid_mask=torch.tensor([[True, False, True, False]]),
                valid_row_mask=torch.tensor([True]),
                guided_action_ids=torch.tensor([0]),
                critic_returns=torch.tensor([0.0]),
                future_dino_grid_targets=torch.randn(1, 4, 16, 1024),
                sigreg_seed=1,
            )


if __name__ == "__main__":
    unittest.main()
