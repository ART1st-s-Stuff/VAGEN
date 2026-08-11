# `vagen.joint_policy`

This package owns versioned policy and behavior-record contracts that are not
part of stock response-token PPO.

Current scope:

- `contract.py` defines the provisional frozen-Q guided policy configuration,
  versioned behavior record, contract identity, and numerical audit reference;
- `torch_policy.py` implements the tensor distribution while always detaching Q
  and requiring an explicit decision for prior-to-LLM gradients;
- the behavior contract binds action names, action token ids, score dtype,
  frozen-Q snapshot identity, selected action, and recorded log-probabilities.

The package does not yet own a ValueHead, critic snapshot service, rollout
sampler, PPO loss, or checkpoint lifecycle. Enabling `joint_policy` therefore
fails closed instead of silently running stock PPO. Remaining integration steps
are described in `docs/joint_policy_scaffold.md`.
