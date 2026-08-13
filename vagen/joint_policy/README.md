# `vagen.joint_policy`

This package owns versioned policy and behavior-record contracts that are not
part of stock response-token PPO.

Current scope:

- the custom Nimloth async rollout captures identity-bound K-slot hidden states
  and raw action-boundary logits from the same generation and transports them to
  `DataProto`; capture schema v2 separates sticky episode `request_id` from a
  unique per-forward `generation_id`; this transport does not score Q or change
  the environment action;
- `contract.py` defines the provisional frozen-Q guided policy configuration,
  versioned behavior record, contract identity, and numerical audit reference;
- `torch_policy.py` implements the tensor distribution while always detaching Q
  and preserving the confirmed prior-to-LLM actor gradient;
- `replay.py` revalidates an immutable behavior-record batch and replays the
  selected guided-action log-prob from current prior logits plus only the
  rollout-persisted frozen Q; callers must provide the expected contract and
  snapshot identities;
- `critic_loss.py` defines a pure selected-action Huber objective: it gathers
  only the actually executed action, detaches return targets, and requires the
  caller to specify `delta` and reduction rather than choosing experiment
  defaults;
- the behavior contract binds action names, action token ids, score dtype,
  frozen-Q snapshot identity, selected action, and recorded log-probabilities.

The parent Nimloth layer has a pure `SharedSlotProjector`/`ValueHead` snapshot
and capture-to-Q scorer, but this package does not yet own a production critic
service, snapshot refresh lifecycle, guided rollout sampler, integrated PPO
loss, or checkpoint lifecycle. The pure scorer and replay helper are not wired
to the actor worker or trainer. Enabling `joint_policy`
therefore fails closed instead of silently running stock PPO. Remaining integration steps
are described in `docs/joint_policy_scaffold.md`.
