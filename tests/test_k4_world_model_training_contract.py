import unittest


def _config(**updates):
    raw = {
        "enabled": True,
        "implementation": "k4_world_model_update_v1",
        "planning_checkpoint": "/checkpoints/id74",
        "snapshot_transport_root": "/outputs/planning_snapshots",
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
    raw.update(updates)
    return raw


class K4WorldModelTrainingContractTest(unittest.TestCase):
    def test_parses_every_approved_value_and_one_optimizer(self) -> None:
        from vagen.joint_policy.k4_training_contract import (
            k4_world_model_training_contract_id,
            parse_k4_world_model_training_section,
        )

        config = parse_k4_world_model_training_section(_config())
        self.assertEqual(config.prediction_horizon, 4)
        self.assertEqual(config.minimum_window_depth, 1)
        self.assertEqual(config.maximum_window_depth, 4)
        self.assertEqual(config.state_mse_weight, 1.0)
        self.assertEqual(config.dino_grid_weight, 0.5)
        self.assertEqual(config.sigreg_weight, 0.1)
        self.assertEqual(config.optimizer.projector_lr, 1e-4)
        self.assertEqual(config.optimizer.predictor_lr, 1e-4)
        self.assertEqual(config.optimizer.value_head_lr, 1e-4)
        self.assertRegex(
            k4_world_model_training_contract_id(config),
            r"^sha256:[0-9a-f]{64}$",
        )

    def test_disabled_section_has_no_hidden_defaults(self) -> None:
        from vagen.joint_policy.k4_training_contract import (
            parse_k4_world_model_training_section,
        )

        self.assertIsNone(parse_k4_world_model_training_section({"enabled": False}))
        with self.assertRaisesRegex(ValueError, "populated"):
            parse_k4_world_model_training_section(
                {"enabled": False, "state_mse_weight": 1.0}
            )

    def test_enabled_requires_every_field_and_exact_id74_shapes(self) -> None:
        from vagen.joint_policy.k4_training_contract import (
            parse_k4_world_model_training_section,
        )

        for field in tuple(_config()):
            raw = _config()
            raw.pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse_k4_world_model_training_section(raw)
        for field, value in {
            "prediction_horizon": 3,
            "minimum_window_depth": 2,
            "maximum_window_depth": 3,
            "sigreg_knots": 16,
            "sigreg_num_proj": 512,
            "grad_clip": 0.0,
        }.items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse_k4_world_model_training_section(_config(**{field: value}))
        bad_dino = dict(_config()["dino_identity"])
        bad_dino["revision"] = "main"
        with self.assertRaisesRegex(ValueError, "DINO identity"):
            parse_k4_world_model_training_section(_config(dino_identity=bad_dino))

    def test_contract_id_binds_optimizer_groups_and_transport_root(self) -> None:
        from vagen.joint_policy.k4_training_contract import (
            k4_world_model_training_contract_id,
            parse_k4_world_model_training_section,
        )

        first = parse_k4_world_model_training_section(_config())
        changed_optim = dict(_config()["optimizer"])
        changed_optim["predictor_lr"] = 2e-4
        second = parse_k4_world_model_training_section(
            _config(optimizer=changed_optim)
        )
        third = parse_k4_world_model_training_section(
            _config(snapshot_transport_root="/different")
        )
        ids = {
            k4_world_model_training_contract_id(value)
            for value in (first, second, third)
        }
        self.assertEqual(len(ids), 3)


if __name__ == "__main__":
    unittest.main()
