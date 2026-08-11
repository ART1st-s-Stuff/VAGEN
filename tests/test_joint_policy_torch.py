from __future__ import annotations

import math
import unittest

try:
    import torch
except ImportError:
    torch = None

from vagen.joint_policy import FrozenQGuidedPolicyConfig
from vagen.joint_policy.contract import guided_log_probs_reference
from vagen.joint_policy.torch_policy import frozen_q_guided_log_probs


def _config(backprop_to_llm: bool, score_dtype: str = "float64"):
    return FrozenQGuidedPolicyConfig.from_mapping(
        {
            "implementation": "frozen_q_guided_v1",
            "alpha": 1.0,
            "beta": 0.5,
            "prior_temperature": 0.7,
            "backprop_to_llm": backprop_to_llm,
            "score_dtype": score_dtype,
        }
    )


@unittest.skipIf(torch is None, "torch is not installed")
class FrozenQGuidedTorchTest(unittest.TestCase):
    def test_matches_reference_and_detaches_q(self) -> None:
        prior = torch.tensor([[0.2, -0.3]], dtype=torch.float64, requires_grad=True)
        q_values = torch.tensor([[1.5, -0.5]], dtype=torch.float64, requires_grad=True)
        config = _config(True)
        output = frozen_q_guided_log_probs(prior, q_values, config)
        expected_prior, expected_guided = guided_log_probs_reference(
            prior.detach().tolist()[0],
            q_values.detach().tolist()[0],
            config,
        )
        torch.testing.assert_close(
            output["prior_log_probs"],
            torch.tensor([expected_prior], dtype=torch.float64),
        )
        torch.testing.assert_close(
            output["guided_log_probs"],
            torch.tensor([expected_guided], dtype=torch.float64),
        )
        loss = output["guided_log_probs"][0, 0]
        loss.backward()
        self.assertIsNotNone(prior.grad)
        self.assertGreater(float(prior.grad.abs().sum()), 0.0)
        self.assertIsNone(q_values.grad)

    def test_detached_prior_has_no_guided_gradient(self) -> None:
        prior = torch.tensor([[0.2, -0.3]], requires_grad=True)
        q_values = torch.tensor([[1.5, -0.5]], requires_grad=True)
        output = frozen_q_guided_log_probs(
            prior,
            q_values,
            _config(False, score_dtype="float32"),
        )
        self.assertFalse(output["guided_log_probs"].requires_grad)

    def test_rejects_overflow_after_scaling(self) -> None:
        prior = torch.tensor([[1e30, -1e30]], dtype=torch.float32)
        q_values = torch.zeros_like(prior)
        config = FrozenQGuidedPolicyConfig.from_mapping(
            {
                "implementation": "frozen_q_guided_v1",
                "alpha": 1e30,
                "beta": 0.0,
                "prior_temperature": 1.0,
                "backprop_to_llm": True,
                "score_dtype": "float32",
            }
        )
        with self.assertRaisesRegex(ValueError, "guided logits must be finite"):
            frozen_q_guided_log_probs(prior, q_values, config)


if __name__ == "__main__":
    unittest.main()
