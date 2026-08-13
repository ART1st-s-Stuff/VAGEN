from __future__ import annotations

import inspect
import unittest

try:
    import torch
except ImportError:
    torch = None

from vagen.joint_policy import selected_action_huber_loss


@unittest.skipIf(torch is None, "torch is not installed")
class SelectedActionHuberLossTest(unittest.TestCase):
    def test_only_selected_actions_receive_gradient_and_targets_are_detached(self) -> None:
        action_values = torch.tensor(
            [[1.0, float("nan")], [float("nan"), 3.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        targets = torch.tensor(
            [2.0, 5.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        output = selected_action_huber_loss(
            action_values,
            torch.tensor([0, 1]),
            targets,
            delta=1.0,
            reduction="mean",
        )
        torch.testing.assert_close(
            output.selected_action_values,
            torch.tensor([1.0, 3.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            output.detached_targets,
            torch.tensor([2.0, 5.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            output.per_sample_loss,
            torch.tensor([0.5, 1.5], dtype=torch.float64),
        )
        torch.testing.assert_close(output.loss, torch.tensor(1.0, dtype=torch.float64))

        output.loss.backward()
        torch.testing.assert_close(
            action_values.grad,
            torch.tensor([[-0.5, 0.0], [0.0, -0.5]], dtype=torch.float64),
        )
        self.assertIsNone(targets.grad)

    def test_delta_and_reduction_are_required_and_sum_none_are_explicit(self) -> None:
        signature = inspect.signature(selected_action_huber_loss)
        self.assertIs(signature.parameters["delta"].default, inspect.Parameter.empty)
        self.assertIs(signature.parameters["reduction"].default, inspect.Parameter.empty)

        action_values = torch.tensor([[0.0, 2.0], [4.0, 1.0]])
        actions = torch.tensor([1, 0])
        targets = torch.tensor([0.0, 1.0])
        none_output = selected_action_huber_loss(
            action_values,
            actions,
            targets,
            delta=0.5,
            reduction="none",
        )
        torch.testing.assert_close(
            none_output.per_sample_loss,
            torch.tensor([0.875, 1.375]),
        )
        torch.testing.assert_close(none_output.loss, none_output.per_sample_loss)
        sum_output = selected_action_huber_loss(
            action_values,
            actions,
            targets,
            delta=0.5,
            reduction="sum",
        )
        torch.testing.assert_close(sum_output.loss, torch.tensor(2.25))

    def test_rejects_invalid_shapes_actions_delta_reduction_and_selected_values(self) -> None:
        valid_q = torch.zeros((2, 2))
        valid_actions = torch.tensor([0, 1])
        valid_targets = torch.zeros(2)
        cases = (
            (torch.zeros(2), valid_actions, valid_targets, 1.0, "mean", "shape"),
            (valid_q, torch.tensor([[0], [1]]), valid_targets, 1.0, "mean", "shape"),
            (valid_q, valid_actions, torch.zeros((2, 1)), 1.0, "mean", "shape"),
            (valid_q, torch.tensor([0.0, 1.0]), valid_targets, 1.0, "mean", "integer"),
            (valid_q, torch.tensor([0, 2]), valid_targets, 1.0, "mean", "outside"),
            (valid_q, valid_actions, valid_targets, 0.0, "mean", "delta"),
            (valid_q, valid_actions, valid_targets, float("inf"), "mean", "delta"),
            (valid_q, valid_actions, valid_targets, 1.0, "batch", "reduction"),
            (
                torch.tensor([[float("nan"), 0.0], [0.0, 0.0]]),
                valid_actions,
                valid_targets,
                1.0,
                "mean",
                "selected action values",
            ),
            (
                valid_q,
                valid_actions,
                torch.tensor([0.0, float("nan")]),
                1.0,
                "mean",
                "targets",
            ),
        )
        for q_values, actions, targets, delta, reduction, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    selected_action_huber_loss(
                        q_values,
                        actions,
                        targets,
                        delta=delta,
                        reduction=reduction,
                    )

    def test_rejects_non_floating_q_and_target_dtypes(self) -> None:
        for dtype in (torch.int64, torch.bool, torch.complex64):
            with self.subTest(q_dtype=dtype):
                with self.assertRaisesRegex(ValueError, "real floating dtype"):
                    selected_action_huber_loss(
                        torch.zeros((1, 2), dtype=dtype),
                        torch.tensor([0]),
                        torch.tensor([0.0]),
                        delta=0.5,
                        reduction="mean",
                    )
        for dtype in (torch.int64, torch.bool, torch.complex64):
            with self.subTest(target_dtype=dtype):
                with self.assertRaisesRegex(ValueError, "real floating dtype"):
                    selected_action_huber_loss(
                        torch.zeros((1, 2), dtype=torch.float32),
                        torch.tensor([0]),
                        torch.zeros((1,), dtype=dtype),
                        delta=0.5,
                        reduction="mean",
                    )

    def test_large_fp16_linear_error_has_finite_fp32_loss_and_gradient(self) -> None:
        action_values = torch.tensor(
            [[60000.0, float("nan")]],
            dtype=torch.float16,
            requires_grad=True,
        )
        output = selected_action_huber_loss(
            action_values,
            torch.tensor([0]),
            torch.tensor([0.0], dtype=torch.float16),
            delta=1.0,
            reduction="mean",
        )
        self.assertEqual(output.loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertTrue(torch.isfinite(action_values.grad).all())
        torch.testing.assert_close(
            action_values.grad,
            torch.tensor([[1.0, 0.0]], dtype=torch.float16),
        )

    def test_delta_must_remain_positive_in_loss_dtype(self) -> None:
        with self.assertRaisesRegex(ValueError, "loss dtype"):
            selected_action_huber_loss(
                torch.zeros((1, 2), dtype=torch.float32),
                torch.tensor([0]),
                torch.tensor([0.0], dtype=torch.float32),
                delta=1e-50,
                reduction="mean",
            )

    def test_converts_floating_targets_to_q_dtype(self) -> None:
        output = selected_action_huber_loss(
            torch.tensor([[1.0, 2.0]], dtype=torch.float64),
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([0.5], dtype=torch.float32),
            delta=0.5,
            reduction="mean",
        )
        self.assertEqual(output.detached_targets.dtype, torch.float64)
        torch.testing.assert_close(
            output.detached_targets,
            torch.tensor([0.5], dtype=torch.float64),
        )
        torch.testing.assert_close(output.loss, torch.tensor(0.625, dtype=torch.float64))

    def test_rejects_empty_batch_and_non_tensor_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            selected_action_huber_loss(
                torch.zeros((0, 2)),
                torch.zeros((0,), dtype=torch.long),
                torch.zeros((0,)),
                delta=1.0,
                reduction="mean",
            )
        with self.assertRaisesRegex(ValueError, "torch Tensor"):
            selected_action_huber_loss(
                [[0.0, 1.0]],
                torch.tensor([0]),
                torch.tensor([0.0]),
                delta=1.0,
                reduction="mean",
            )


if __name__ == "__main__":
    unittest.main()
