from __future__ import annotations

import inspect
import unittest
from dataclasses import replace

try:
    import torch
except ImportError:
    torch = None

from vagen.joint_policy import (
    FrozenQGuidedPolicyConfig,
    GuidedPolicyBehaviorRecord,
    replay_guided_behavior_log_probs as _replay_guided_behavior_log_probs,
)
from vagen.joint_policy.contract import guided_log_probs_reference


def replay_guided_behavior_log_probs(
    current_prior_logits,
    behavior_records,
    *,
    expected_snapshot_id: str,
    expected_contract_id: str | None = None,
):
    if expected_contract_id is None:
        expected_contract_id = (
            behavior_records[0].contract_id if behavior_records else "empty-batch"
        )
    return _replay_guided_behavior_log_probs(
        current_prior_logits,
        behavior_records,
        expected_contract_id=expected_contract_id,
        expected_snapshot_id=expected_snapshot_id,
    )


def _config(*, beta: float = 0.5) -> FrozenQGuidedPolicyConfig:
    return FrozenQGuidedPolicyConfig.from_mapping(
        {
            "implementation": "frozen_q_guided_v1",
            "alpha": 1.0,
            "beta": beta,
            "prior_temperature": 0.7,
            "backprop_to_llm": True,
            "score_dtype": "float64",
        }
    )


def _record(
    *,
    prior_logits: tuple[float, ...],
    frozen_q: tuple[float, ...],
    prior_action_id: int,
    guided_action_id: int,
    snapshot_id: str = "snapshot-7",
    config: FrozenQGuidedPolicyConfig | None = None,
    action_names: tuple[str, ...] = ("left", "right"),
    action_token_ids: tuple[int, ...] = (101, 102),
) -> GuidedPolicyBehaviorRecord:
    policy_config = config or _config()
    prior_log_probs, guided_log_probs = guided_log_probs_reference(
        prior_logits,
        frozen_q,
        policy_config,
    )
    return GuidedPolicyBehaviorRecord.build(
        action_space="navigation_v1",
        action_space_names=action_names,
        action_token_ids=action_token_ids,
        snapshot_id=snapshot_id,
        prior_token_id=action_token_ids[prior_action_id],
        prior_action_id=prior_action_id,
        prior_response_idx=22,
        behavior_llm_prior_logprob=prior_log_probs[prior_action_id],
        prior_logits=prior_logits,
        frozen_all_action_q=frozen_q,
        guided_action_id=guided_action_id,
        behavior_guided_logprob=guided_log_probs[guided_action_id],
        config=policy_config,
    )


@unittest.skipIf(torch is None, "torch is not installed")
class GuidedBehaviorReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = (
            _record(
                prior_logits=(0.2, -0.3),
                frozen_q=(1.5, -0.5),
                prior_action_id=0,
                guided_action_id=0,
            ),
            _record(
                prior_logits=(-0.4, 0.7),
                frozen_q=(-0.2, 0.9),
                prior_action_id=1,
                guided_action_id=1,
            ),
        )

    def test_replays_persisted_q_and_preserves_current_prior_gradient(self) -> None:
        current = torch.tensor(
            [[0.1, 0.8], [0.6, -0.2]],
            dtype=torch.float64,
            requires_grad=True,
        )
        output = replay_guided_behavior_log_probs(
            current,
            self.records,
            expected_snapshot_id="snapshot-7",
        )

        persisted_q = torch.tensor(
            [record.frozen_all_action_q for record in self.records],
            dtype=torch.float64,
        )
        guided_logits = current / 0.7 + 0.5 * persisted_q
        expected_all = torch.log_softmax(guided_logits, dim=-1)
        expected_selected = expected_all[
            torch.arange(2),
            torch.tensor([0, 1]),
        ]
        torch.testing.assert_close(output["current_guided_log_probs"], expected_selected)
        torch.testing.assert_close(
            output["all_current_guided_log_probs"],
            expected_all,
        )
        torch.testing.assert_close(
            output["behavior_guided_log_probs"],
            torch.tensor(
                [record.behavior_guided_logprob for record in self.records],
                dtype=torch.float64,
            ),
        )

        (-output["current_guided_log_probs"].mean()).backward()
        self.assertIsNotNone(current.grad)
        self.assertGreater(float(current.grad.abs().sum()), 0.0)
        self.assertNotIn(
            "current_q",
            inspect.signature(_replay_guided_behavior_log_probs).parameters,
        )

    def test_current_logits_change_replayed_probability_without_mutating_records(self) -> None:
        before = tuple(record.to_mapping() for record in self.records)
        first = replay_guided_behavior_log_probs(
            torch.tensor([[0.1, 0.8], [0.6, -0.2]], dtype=torch.float64),
            self.records,
            expected_snapshot_id="snapshot-7",
        )["current_guided_log_probs"]
        second = replay_guided_behavior_log_probs(
            torch.tensor([[1.1, -0.8], [-0.6, 1.2]], dtype=torch.float64),
            self.records,
            expected_snapshot_id="snapshot-7",
        )["current_guided_log_probs"]
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(before, tuple(record.to_mapping() for record in self.records))

    def test_rejects_wrong_shape_and_empty_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            replay_guided_behavior_log_probs(
                torch.zeros((2, 3), dtype=torch.float64),
                self.records,
                expected_snapshot_id="snapshot-7",
            )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            replay_guided_behavior_log_probs(
                torch.zeros((0, 2), dtype=torch.float64),
                (),
                expected_snapshot_id="snapshot-7",
            )

    def test_rejects_snapshot_contract_and_action_table_mixing(self) -> None:
        other_snapshot = replace(self.records[1], snapshot_id="snapshot-8")
        with self.assertRaisesRegex(ValueError, "snapshot"):
            replay_guided_behavior_log_probs(
                torch.zeros((2, 2), dtype=torch.float64),
                (self.records[0], other_snapshot),
                expected_snapshot_id="snapshot-7",
            )

        other_contract = _record(
            prior_logits=(-0.4, 0.7),
            frozen_q=(-0.2, 0.9),
            prior_action_id=1,
            guided_action_id=1,
            config=_config(beta=0.8),
        )
        with self.assertRaisesRegex(ValueError, "contract"):
            replay_guided_behavior_log_probs(
                torch.zeros((2, 2), dtype=torch.float64),
                (self.records[0], other_contract),
                expected_snapshot_id="snapshot-7",
            )

        other_table = _record(
            prior_logits=(-0.4, 0.7),
            frozen_q=(-0.2, 0.9),
            prior_action_id=1,
            guided_action_id=1,
            action_names=("back", "forward"),
        )
        with self.assertRaisesRegex(ValueError, "contract"):
            replay_guided_behavior_log_probs(
                torch.zeros((2, 2), dtype=torch.float64),
                (self.records[0], other_table),
                expected_snapshot_id="snapshot-7",
            )

    def test_revalidates_records_and_expected_snapshot(self) -> None:
        malformed = replace(self.records[1], behavior_guided_logprob=0.0)
        with self.assertRaisesRegex(ValueError, "behavior guided log-prob"):
            replay_guided_behavior_log_probs(
                torch.zeros((2, 2), dtype=torch.float64),
                (self.records[0], malformed),
                expected_snapshot_id="snapshot-7",
            )
        with self.assertRaisesRegex(ValueError, "expected_snapshot_id"):
            replay_guided_behavior_log_probs(
                torch.zeros((2, 2), dtype=torch.float64),
                self.records,
                expected_snapshot_id="",
            )
        with self.assertRaisesRegex(ValueError, "snapshot"):
            replay_guided_behavior_log_probs(
                torch.zeros((2, 2), dtype=torch.float64),
                self.records,
                expected_snapshot_id="snapshot-missing",
            )
        with self.assertRaisesRegex(ValueError, "contract"):
            replay_guided_behavior_log_probs(
                torch.zeros((2, 2), dtype=torch.float64),
                self.records,
                expected_contract_id="sha256:wrong-contract",
                expected_snapshot_id="snapshot-7",
            )


if __name__ == "__main__":
    unittest.main()
